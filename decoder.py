import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from build_swin_unet import build_swin_unet
import cv2 as cv
import copy
import warnings
warnings.filterwarnings("ignore")

def get_junctions(jloc, joff, topk = 300, th=0):
    height, width = jloc.size(1), jloc.size(2)
    jloc_flat = jloc.flatten()
    joff_flat = joff.flatten(start_dim=1)

    scores, index = torch.topk(jloc_flat, k=topk)
    y = (index / width).float() + torch.gather(joff_flat[1], 0, index) + 0.5
    x = (index % width).float() + torch.gather(joff_flat[0], 0, index) + 0.5

    junctions = torch.stack((x, y)).t()

    score_mask = scores>th

    return junctions[score_mask], scores[score_mask], index[score_mask]

def non_maximum_suppression(a):
    ap = F.max_pool2d(a, 3, stride=1, padding=1)
    mask = (a == ap).float().clamp(min=0.0)
    return a * mask


def sigmoid_l1_loss(logits, targets, offset=0.0, mask=None):
    logp = torch.sigmoid(logits) + offset
    a = logp
    b = targets
    c = a-b
    loss = torch.abs(logp - targets)

    if mask is not None:
        w = mask.mean(3, keepdim=True).mean(2, keepdim=True)
        w[w == 0] = 1
        loss = loss * (mask / w)

    return loss.mean()


def cross_entropy_loss_for_junction(logits, positive):
    nlogp = -F.log_softmax(logits, dim=1)
    loss = (positive * nlogp[:, None, 1] + (1 - positive) * nlogp[:, None, 0])
    return loss.mean()


class EvRoomDetector(nn.Module):
    def __init__(self,):
        super().__init__()
        # self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.loss = nn.CrossEntropyLoss(reduction='none', weight=None)
        self.use_gt_lines = False
        self.use_residual = True
        self.use_gt_junctions = False
        self.require_valid_junctions = False
        junction_label_weight = torch.tensor([0.4, 100, 100], dtype=torch.float32)
        self.junction_label_loss = nn.CrossEntropyLoss(weight=junction_label_weight)
        self.pool1d = nn.MaxPool1d(32 // 8, stride=32 // 8)
        self.use_residual = True
        self.n_dyn_othr2 = 300
        self.n_pts0 = 32
        self.n_pts1 = 8
        self.dim_loi = 128
        self.dim_fc = 1024
        self.n_dyn_junc = 50
        self.n_dyn_posl = 300
        self.junc_thresh = 0.008
        self.n_out_junc = 80
        self.max_distance = float('inf')
        self.fc0 = nn.Conv2d(96, 1, 1)
        self.fc1 = nn.Conv2d(256, self.dim_loi, 1)
        self.fc = nn.Sequential(
            nn.Linear(self.dim_loi * self.n_pts1, self.dim_fc),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim_fc, self.dim_fc),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim_fc, 1)
        )
        self.register_buffer('tspan', torch.linspace(0, 1, self.n_pts0)[None, None, :])
        self.backbone = build_swin_unet()
        self.DIS_TH = 5
        self.ANG_TH = 0.1
        self.NUM_STATIC_POS_LINES = 300
        self.NUM_STATIC_NEG_LINES = 40

        self.nbr_line_labels = 6
        last_fc = nn.Linear(self.dim_fc, self.nbr_line_labels)
        self.fc2 = nn.Sequential(
            nn.Linear(self.dim_loi * self.n_pts1, self.dim_fc),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim_fc, self.dim_fc),
            nn.ReLU(inplace=True),
            last_fc
        )
    def pooling(self, features_per_image, lines_per_im):
        h, w = features_per_image.size(1), features_per_image.size(2)
        U, V = lines_per_im[:, :2], lines_per_im[:, 2:]
        sampled_points = U[:, :, None] * self.tspan + V[:, :, None] * (1 - self.tspan) - 0.5
        sampled_points = sampled_points.permute((0, 2, 1)).reshape(-1, 2)
        px, py = sampled_points[:, 0], sampled_points[:, 1]
        px0 = px.floor().clamp(min=0, max=w - 1)
        py0 = py.floor().clamp(min=0, max=h - 1)
        px1 = (px0 + 1).clamp(min=0, max=w - 1)
        py1 = (py0 + 1).clamp(min=0, max=h - 1)
        px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()

        xp = ((features_per_image[:, py0l, px0l] * (py1 - py) * (px1 - px) + features_per_image[:, py1l, px0l] * (
                    py - py0) * (px1 - px) + features_per_image[:, py0l, px1l] * (py1 - py) * (
                           px - px0) + features_per_image[:, py1l, px1l] * (py - py0) * (px - px0)).reshape(128, -1, 32)
              ).permute(1, 0, 2)

        # if self.pool1d is not None:
        xp = self.pool1d(xp)
        features_per_line = xp.view(-1, self.n_pts1 * self.dim_loi)
        # features_per_line = self.fc2(features_per_line)

        return features_per_line

    def proposal_lines(self, md_maps, dis_maps, scale=5.0):
        """

        :param md_maps: 3xhxw, the range should be (0,1) for every element
        :param dis_maps: 1xhxw
        :return:
        """
        device = md_maps.device
        height, width = md_maps.size(1), md_maps.size(2)
        _y = torch.arange(0, height, device=device).float()
        _x = torch.arange(0, width, device=device).float()

        y0, x0 = torch.meshgrid(_y, _x)
        md_ = (md_maps[0] - 0.5) * np.pi * 2
        st_ = md_maps[1] * np.pi / 2
        ed_ = -md_maps[2] * np.pi / 2

        cs_md = torch.cos(md_)
        ss_md = torch.sin(md_)

        cs_st = torch.cos(st_).clamp(min=1e-3)
        ss_st = torch.sin(st_).clamp(min=1e-3)

        cs_ed = torch.cos(ed_).clamp(min=1e-3)
        ss_ed = torch.sin(ed_).clamp(max=-1e-3)

        x_standard = torch.ones_like(cs_st)

        y_st = ss_st / cs_st
        y_ed = ss_ed / cs_ed

        x_st_rotated = (cs_md - ss_md * y_st) * dis_maps[0] * scale
        y_st_rotated = (ss_md + cs_md * y_st) * dis_maps[0] * scale

        x_ed_rotated = (cs_md - ss_md * y_ed) * dis_maps[0] * scale
        y_ed_rotated = (ss_md + cs_md * y_ed) * dis_maps[0] * scale

        x_st_final = (x_st_rotated + x0).clamp(min=0, max=width - 1)
        y_st_final = (y_st_rotated + y0).clamp(min=0, max=height - 1)

        x_ed_final = (x_ed_rotated + x0).clamp(min=0, max=width - 1)
        y_ed_final = (y_ed_rotated + y0).clamp(min=0, max=height - 1)

        lines = torch.stack((x_st_final, y_st_final, x_ed_final, y_ed_final)).permute((1, 2, 0))

        return lines  # , normals

    def forward(self, data, label, label2):
        return self.forward_train(data, label, label2)

    def gen_line_map(self, label2):
        batch_lines = []
        for i in range(len(label2)):
            this_line = label2[i]['lines'].detach().cpu().numpy()
            this_line_map = np.zeros((128, 128, 1), dtype=np.float32)
            radius = 1
            for l in this_line:
                x0 = int(round(l[0]))
                y0 = int(round(l[1]))
                x1 = int(round(l[2]))
                y1 = int(round(l[3]))
                cv.line(this_line_map, (x0, y0), (x1, y1), (255, 255, 255), radius)
            this_line_map = np.array(this_line_map, dtype=np.float32) / 255.0
            this_line_map = torch.tensor(this_line_map.transpose(2, 0, 1)).unsqueeze(0)
            batch_lines.append(this_line_map)
        out_lines_map = torch.cat(batch_lines, dim=0)
        return out_lines_map

    def weighted_bce_with_logits(self, out, gt, pos_w=1.0, neg_w=30.0):
        pos_mask = torch.where(gt != 0, torch.ones_like(gt), torch.zeros_like(gt))
        neg_mask = torch.ones_like(pos_mask) - pos_mask

        loss = F.binary_cross_entropy_with_logits(out, gt, reduction='none')
        loss_pos = (loss * pos_mask).sum() / torch.sum(pos_mask)
        loss_neg = (loss * neg_mask).sum() / torch.sum(neg_mask)

        loss = loss_pos * pos_w + loss_neg * neg_w
        return loss
    def forward_train(self, data, label, label2):
        # np.save('/home/sduu2/userspace-18T-2/gxc/project/swin_hawpv7/images.npy', data['img'])
        # np.save('/home/sduu2/userspace-18T-2/gxc/project/swin_hawpv7/gt1.npy', label)
        # np.save('/home/sduu2/userspace-18T-2/gxc/project/swin_hawpv7/gt2.npy', label2)
        # import pdb
        # pdb.set_trace()
        batch_size = data['img'].shape[0]
        this_device = data['img'].device
        outputs, features = self.backbone(data)

        mask = label['mask']

        loss_map = torch.mean(F.l1_loss(outputs[:, :3].sigmoid(), label['md'], reduction='none'), dim=1, keepdim=True)
        loss_md = torch.mean(loss_map * mask) / torch.mean(mask)
        loss_map = F.l1_loss(outputs[:, 3:4].sigmoid(), label['dis'], reduction='none')
        loss_dis = torch.mean(loss_map * mask) / torch.mean(mask)
        loss_residual_map = F.l1_loss(outputs[:, 4:5].sigmoid(), loss_map, reduction='none')
        loss_res = torch.mean(loss_residual_map * mask) / torch.mean(mask)
        # loss_jloc = cross_entropy_loss_for_junction(outputs[:, 5:7], label_new['jloc'])
        loss_jlabel = self.junction_label_loss(outputs[:, 5:8].flatten(start_dim=2), label['jlabel'].flatten(start_dim=1))
        loss_joff = sigmoid_l1_loss(outputs[:, 8:10], label['joff'], -0.5, label['jloc'])

        # 求后2个loss
        loss_pos = 0
        loss_neg = 0
        loss_cons_result = 0

        loi_features = self.fc1(features)
        md_preds = outputs[:, :3].sigmoid()
        dis_preds = outputs[:, 3:4].sigmoid()
        res_preds = outputs[:, 4:5].sigmoid()
        # jloc_preds = outputs[:, 5:7].softmax(1)[:, 1:]
        jlabel_prob = outputs[:, 5:8].softmax(1)
        jlabel = jlabel_prob.argmax(1)
        jloc_preds = 1 - jlabel_prob[:, 0, None]
        joff_preds = outputs[:, 8:10].sigmoid() - 0.5

        # 前期同步达成
        for i, (md_pred_per_im, dis_pred_per_im, res_pred_per_im, meta) in enumerate(zip(md_preds, dis_preds, res_preds, label2)):
            lines_pred = []
            if self.use_residual:
                for scale in [-1.0, 0.0, 1.0]:
                    _ = self.proposal_lines(md_pred_per_im, dis_pred_per_im + scale * res_pred_per_im).view(-1, 4)
                    lines_pred.append(_)
            else:
                lines_pred.append(self.proposal_lines(md_pred_per_im, dis_pred_per_im).view(-1, 4))
            lines_pred = torch.cat(lines_pred)
            junction_gt = meta['junc'].to(this_device)
            N = junction_gt.size(0)

            juncs_pred, juncs_valid_score, flat_index = get_junctions(non_maximum_suppression(jloc_preds[i]), joff_preds[i], topk=min(N * 2 + 2, self.n_dyn_junc))
            junction_features = loi_features[i].flatten(start_dim=1)[:, flat_index].T
            juncs_logits = (outputs[:, 5:8].flatten(start_dim=2)[i, :, flat_index]).T

            if juncs_pred.size(0) < 2:
                logits = self.pooling(loi_features[i],meta['lpre'].to(this_device))
                loss_ = self.loss(logits, meta['lpre_label'].to(this_device))

                loss_positive = loss_[meta['lpre_label']>0].mean()
                loss_negative = loss_[meta['lpre_label']==0].mean()

                loss_pos += loss_positive/batch_size
                loss_neg += loss_negative/batch_size

                continue

            dis_junc_to_end1, idx_junc_to_end1 = torch.sum((lines_pred[:, :2] - juncs_pred[:, None]) ** 2, dim=-1).min(0)
            dis_junc_to_end2, idx_junc_to_end2 = torch.sum((lines_pred[:, 2:] - juncs_pred[:, None]) ** 2, dim=-1).min(0)

            idx_junc_to_end_min = torch.min(idx_junc_to_end1, idx_junc_to_end2)
            idx_junc_to_end_max = torch.max(idx_junc_to_end1, idx_junc_to_end2)
            iskeep = idx_junc_to_end_min < idx_junc_to_end_max
            idx_lines_for_junctions = torch.cat((idx_junc_to_end_min[iskeep, None], idx_junc_to_end_max[iskeep, None]),
                                                dim=1).unique(dim=0)
            # idx_lines_for_junctions_mirror = torch.cat((idx_lines_for_junctions[:,1,None],idx_lines_for_junctions[:,0,None]),dim=1)
            # idx_lines_for_junctions = torch.cat((idx_lines_for_junctions, idx_lines_for_junctions_mirror))
            lines_adjusted = torch.cat(
                (juncs_pred[idx_lines_for_junctions[:, 0]], juncs_pred[idx_lines_for_junctions[:, 1]]), dim=1)

            cost_, match_ = torch.sum((juncs_pred - junction_gt[:, None]) ** 2, dim=-1).min(0)
            match_[cost_ > 1.5 * 1.5] = N
            Lpos = meta['Lpos'].to(this_device)
            labels = Lpos[match_[idx_lines_for_junctions[:, 0]], match_[idx_lines_for_junctions[:, 1]]]

            iskeep = torch.zeros_like(labels, dtype=torch.bool)
            cdx = labels.nonzero().flatten()

            if len(cdx) > self.n_dyn_posl:
                perm = torch.randperm(len(cdx), device=this_device)[:self.n_dyn_posl]
                cdx = cdx[perm]

            iskeep[cdx] = 1

            if self.n_dyn_othr2 >0 :
                cdx = (labels==0).nonzero().flatten()
                if len(cdx) > self.n_dyn_othr2:
                    perm = torch.randperm(len(cdx), device=this_device)[:self.n_dyn_othr2]
                    cdx = cdx[perm]
                iskeep[cdx] = 1

            all_lines = torch.cat((lines_adjusted, meta['lpre'].to(this_device)))
            all_labels = torch.cat((labels, meta['lpre_label'].to(this_device)))

            pooled_line_features = self.pooling(loi_features[i], all_lines)
            line_logits = self.fc2(pooled_line_features)
            all_iskeep = torch.cat((iskeep, torch.ones_like(meta['lpre_label'].to(this_device), dtype=torch.bool)))
            selected_labels = all_labels[all_iskeep]
            selected_logits_nognn = line_logits[all_iskeep]
            loss_no_gnn = self.loss(selected_logits_nognn, selected_labels)

            loss_pos += loss_no_gnn[selected_labels > 0].mean() / batch_size
            loss_neg += loss_no_gnn[selected_labels == 0].mean() / batch_size


            # lines_selected = lines_adjusted[iskeep]
            # idx_lines_for_junctions = idx_lines_for_junctions[iskeep]
            # log_prob_valid = 1 - line_logits[:lines_selected.size(0)].softmax(1)[:, 0]
            # loss_cons = log_prob_valid.unsqueeze(1) * (1 - juncs_valid_score[idx_lines_for_junctions])
            # loss_cons_result += loss_cons.mean() / batch_size

        return loss_md, loss_dis, loss_res, loss_jlabel, loss_joff, loss_pos, loss_neg

    def forward_test(self, data, annotations):
        extra_info = {
        }
        batch_size = data['img'].shape[0]
        device = data['img'].device
        outputs, features = self.backbone(data)
        outpath = '/home/sduu2/userspace-18T-2/gxc/project/process_all_need_data/check_featuremap/'
        loi_features = self.fc1(features)
        # np.save(outpath + 'loi.npy', loi_features.detach().cpu().numpy())
        md_preds = outputs[:, :3].sigmoid()
        # np.save(outpath + 'md.npy',md_preds.detach().cpu().numpy())
        dis_preds = outputs[:, 3:4].sigmoid()
        # np.save(outpath + 'dis.npy', dis_preds.detach().cpu().numpy())
        res_preds = outputs[:, 4:5].sigmoid()
        # np.save(outpath + 'res.npy', res_preds.detach().cpu().numpy())
        jlabel_prob = outputs[:, 5:8].softmax(1)
        # np.save(outpath + 'jlabel.npy', jlabel_prob.detach().cpu().numpy())
        jloc_pred = 1 - jlabel_prob[:, 0, None]
        # np.save(outpath + 'jloc.npy', jloc_pred.detach().cpu().numpy())
        joff_preds = outputs[:, 8:10].sigmoid() - 0.5
        # np.save(outpath + 'joff.npy', joff_preds.detach().cpu().numpy())

        # in_folder = '/home/sduu2/userspace-18T-2/gxc/project/process_all_need_data/check_featuremap/orig_feature/'
        # md_preds = torch.tensor(np.load(in_folder + 'md.npy')).to(device)
        # dis_preds = torch.tensor(np.load(in_folder + 'dis.npy')).to(device)
        # res_preds = torch.tensor(np.load(in_folder + 'res.npy')).to(device)
        # jlabel_prob = torch.tensor(np.load(in_folder + 'jlabel.npy')).to(device)
        # jloc_pred = torch.tensor(np.load(in_folder + 'jloc.npy')).to(device)
        # joff_preds = torch.tensor(np.load(in_folder + 'joff.npy')).to(device)


        batch_size = md_preds.size(0)
        assert batch_size == 1
        ann = annotations[0]

        if self.use_gt_lines:
            junctions = ann['junctions']
            junctions[:,0] *= 128/float(ann['width'])
            junctions[:,1] *= 128/float(ann['height'])
            edges_positive = ann['edges_positive']
            lines_pred = torch.cat((junctions[edges_positive[:,0]], junctions[edges_positive[:,1]]),dim=-1).to(device)
        elif self.use_residual:
            lines_pred = self.proposal_lines_new(md_preds[0],dis_preds[0],res_preds[0]).view(-1,4)
        else:
            lines_pred = self.proposal_lines_new(md_preds[0], dis_preds[0], None).view(-1, 4)

        jloc_pred_nms = non_maximum_suppression(jloc_pred[0])
        topK = min(self.n_out_junc, int((jloc_pred_nms>0.008).float().sum().item()))

        if self.use_gt_junctions:
            juncs_pred = ann['junctions'].to(device)
            juncs_pred[:,0] *= 128/float(ann['width'])
            juncs_pred[:,1] *= 128/float(ann['height'])
            juncs_label = ann['junctions_semantic']
            juncs_score = torch.zeros([juncs_pred.size(0), jlabel_prob.size(1)])
            juncs_score[range(juncs_label.size(0)), juncs_label] = 1
            juncs_logits = juncs_score
        else:
            juncs_pred, juncs_valid_score, flat_index = get_junctions(jloc_pred_nms, joff_preds[0], topk=topK)
            juncs_logits = (outputs[:, 5:8].flatten(start_dim=2)[0,:,flat_index]).T
            juncs_score = (jlabel_prob.flatten(start_dim=2)[0,:,flat_index]).T
            juncs_label = juncs_score.argmax(dim=1)
            junction_features = loi_features[0].flatten(start_dim=1)[:,flat_index].T

        if self.require_valid_junctions:
            keep_mask = juncs_label > 0
            juncs_pred = juncs_pred[keep_mask]
            juncs_score = juncs_score[keep_mask]
            juncs_label = juncs_label[keep_mask]
            flat_index = flat_index[keep_mask]
        if juncs_pred.size(0) > 1:
            dis_junc_to_end1, idx_junc_to_end1 = torch.sum((lines_pred[:,:2]-juncs_pred[:,None])**2,dim=-1).min(0)
            dis_junc_to_end2, idx_junc_to_end2 = torch.sum((lines_pred[:,2:] - juncs_pred[:, None]) ** 2, dim=-1).min(0)

            idx_junc_to_end_min = torch.min(idx_junc_to_end1,idx_junc_to_end2)
            idx_junc_to_end_max = torch.max(idx_junc_to_end1,idx_junc_to_end2)

            # iskeep = (idx_junc_to_end_min < idx_junc_to_end_max)# * (dis_junc_to_end1< 10*10)*(dis_junc_to_end2<10*10)  # *(dis_junc_to_end2<100)
            iskeep = (idx_junc_to_end_min < idx_junc_to_end_max)*(dis_junc_to_end1< self.max_distance**2)*(dis_junc_to_end2<self.max_distance**2)
        else:
            iskeep = torch.zeros(1, dtype=torch.bool)

        some_lines_valid = iskeep.count_nonzero() > 0
        if some_lines_valid:
            idx_lines_for_junctions = torch.unique(
                torch.cat((idx_junc_to_end_min[iskeep,None],idx_junc_to_end_max[iskeep,None]),dim=1),
                dim=0)

            lines_adjusted = torch.cat((juncs_pred[idx_lines_for_junctions[:,0]], juncs_pred[idx_lines_for_junctions[:,1]]),dim=1)


            pooled_line_features = self.pooling(loi_features[0],lines_adjusted)

            # Filter lines
            line_logits = self.fc2(pooled_line_features)

            scores = line_logits.softmax(1)
            # TODO: Why is this done? And why not also filter the junctions?
            lines_score_valid = 1-scores[:,0]
            valid_mask = lines_score_valid > 0.05
            lines_final = lines_adjusted[valid_mask]
            pooled_line_features = pooled_line_features[valid_mask]
            line_logits = line_logits[valid_mask]

            # TODO: Supply edges for the junctions?
            unique_j_idx, l2j_idx = idx_lines_for_junctions[valid_mask].unique(return_inverse=True)
            juncs_final = juncs_pred[unique_j_idx]
            junction_features = junction_features[unique_j_idx]
            juncs_logits = juncs_logits[unique_j_idx]

            scores = line_logits.softmax(1)  # <<<<
            lines_score_valid = 1 - scores[:, 0]  # <<<<
            lines_label = scores.argmax(1)  # <<<<
            lines_score_label = torch.gather(scores, 1, lines_label.unsqueeze(1)).squeeze(1)  # <<<<

            juncs_score = juncs_logits.softmax(1)  # <<<<
            juncs_label = juncs_score.argmax(1)  # <<<<
            juncs_valid_score = 1 - juncs_score[:, 0]  # <<<<
            juncs_label_score = torch.gather(juncs_score, 1, juncs_label.unsqueeze(1)).squeeze(1)  # <<<<

            width = annotations[0]['width']
            height = annotations[0]['height']

            sx = width / jloc_pred.size(3)  # <<<<
            sy = height / jloc_pred.size(2)  # <<<<

            juncs_pred[:, 0] *= sx  # <<<<
            juncs_pred[:, 1] *= sy  # <<<<
            extra_info['junc_prior_ver'] = juncs_pred  # <<<<
            lines_pred[:, 0] *= sx  # <<<<
            lines_pred[:, 1] *= sy  # <<<<
            lines_pred[:, 2] *= sx  # <<<<
            lines_pred[:, 3] *= sy  # <<<<
            extra_info['lines_prior_ver'] = lines_pred  # <<<<

            lines_adjusted[:, 0] *= sx  # <<<<
            lines_adjusted[:, 1] *= sy  # <<<<
            lines_adjusted[:, 2] *= sx  # <<<<
            lines_adjusted[:, 3] *= sy  # <<<<
            extra_info['lines_prior_scoring'] = lines_adjusted  # <<<<

            extra_info['gnn_line_features'] = pooled_line_features  # <<<<
            extra_info['gnn_junction_features'] = junction_features  # <<<<

            output = {
                'num_proposals': 0,
                'filename': annotations[0]['filename'] if annotations else None,
                'width': width,
                'height': height,
            }  # <<<<

            lines_final[:, 0] *= sx  # <<<<
            lines_final[:, 1] *= sy  # <<<<
            lines_final[:, 2] *= sx  # <<<<
            lines_final[:, 3] *= sy  # <<<<

            juncs_final[:, 0] *= sx  # <<<<
            juncs_final[:, 1] *= sy  # <<<<

            output.update({
                'lines_pred': lines_final,
                'lines_label': lines_label,
                'lines_valid_score': lines_score_valid,
                'lines_label_score': lines_score_label,
                'lines_score': scores,
                'juncs_pred': juncs_final,
                'juncs_label': juncs_label,
                'juncs_valid_score': juncs_valid_score,
                'juncs_label_score': juncs_label_score,
                'juncs_score': juncs_score,
                'line2junc_idx': l2j_idx,
                'num_proposals': lines_adjusted.size(0),
            })

        return lines_final, line_logits, juncs_final, juncs_logits, output, extra_info


    def proposal_lines_new(self, md_maps, dis_maps, residual_maps, scale=5.0):
        """

        :param md_maps: 3xhxw, the range should be (0,1) for every element
        :param dis_maps: 1xhxw
        :return:
        """
        device = md_maps.device
        sign_pad     = torch.tensor([-1,0,1],device=device,dtype=torch.float32).reshape(3,1,1)

        if residual_maps is None:
            dis_maps_new = dis_maps.repeat((1,1,1))
        else:
            dis_maps_new = dis_maps.repeat((3,1,1))+sign_pad*residual_maps.repeat((3,1,1))
        height, width = md_maps.size(1), md_maps.size(2)
        _y = torch.arange(0,height,device=device).float()
        _x = torch.arange(0,width, device=device).float()

        y0,x0 = torch.meshgrid(_y,_x)
        md_ = (md_maps[0]-0.5)*np.pi*2
        st_ = md_maps[1]*np.pi/2
        ed_ = -md_maps[2]*np.pi/2

        cs_md = torch.cos(md_)
        ss_md = torch.sin(md_)

        cs_st = torch.cos(st_).clamp(min=1e-3)
        ss_st = torch.sin(st_).clamp(min=1e-3)

        cs_ed = torch.cos(ed_).clamp(min=1e-3)
        ss_ed = torch.sin(ed_).clamp(max=-1e-3)

        y_st = ss_st/cs_st
        y_ed = ss_ed/cs_ed

        x_st_rotated = (cs_md-ss_md*y_st)[None]*dis_maps_new*scale
        y_st_rotated =  (ss_md + cs_md*y_st)[None]*dis_maps_new*scale

        x_ed_rotated =  (cs_md - ss_md*y_ed)[None]*dis_maps_new*scale
        y_ed_rotated = (ss_md + cs_md*y_ed)[None]*dis_maps_new*scale

        x_st_final = (x_st_rotated + x0[None]).clamp(min=0,max=width-1)
        y_st_final = (y_st_rotated + y0[None]).clamp(min=0,max=height-1)

        x_ed_final = (x_ed_rotated + x0[None]).clamp(min=0,max=width-1)
        y_ed_final = (y_ed_rotated + y0[None]).clamp(min=0,max=height-1)

        lines = torch.stack((x_st_final,y_st_final,x_ed_final,y_ed_final)).permute((1,2,3,0))

        # normals = torch.stack((cs_md,ss_md)).permute((1,2,0))

        return  lines#, normals