import torch
import random
import numpy as np

from parsing.config import cfg
from parsing.utils.comm import to_device
from parsing.dataset import build_train_dataset
from parsing.dataset import build_test_dataset
from parsing.detector import WireframeDetector
from parsing.gnn import WireframeGNNClassifier
from parsing.solver import make_lr_scheduler, make_optimizer
from parsing.utils.logger import setup_logger, wandb_init
from parsing.utils.metric_logger import MetricLogger
from parsing.utils.miscellaneous import save_config
from parsing.utils.checkpoint import DetectronCheckpointer
from parsing.encoder.hafm import HAFMencoder
import os
import os.path as osp
import time
from datetime import datetime, timedelta
import argparse
import logging
from torch.utils.tensorboard import SummaryWriter
import sys
import glob
import torch.optim as optim
from decoder import EvRoomDetector
from tqdm import tqdm
from utils.tester import eval_sap
torch.multiprocessing.set_sharing_strategy('file_system')


def train(cfg):
    resume = False
    lr = 0.0004
    epochs = 500

    device = 'cuda:0'
    train_dataset = build_train_dataset(cfg)
    test_dataset = build_test_dataset(cfg, validation=False)
    hafm_encoder = HAFMencoder(cfg)

    model = EvRoomDetector().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if resume==True:
        print('Attention! pretrain load=========================================================')
        checkpoint = torch.load('./outputs/latest_swin_hawp_48.pth',  map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(epochs):
        epoch_loss = 0
        losst_md = 0
        losst_dis = 0
        losst_res = 0
        losst_jlabel = 0
        losst_joff = 0
        losst_pos = 0
        losst_neg = 0
        model.train()
        for images, annotations in tqdm(train_dataset):
            images = to_device(images, 'cuda:0')
            annotations = to_device(annotations, 'cuda:0')
            targets, metas = hafm_encoder(annotations)
            images = to_device(images, device)
            annotations = to_device(annotations, device)
            targets = to_device(targets, device)
            metas = to_device(metas, device)

            data = {}
            data['img'] = images
            map_orig = torch.cat([annotations[item]['map_orig'].unsqueeze(0) for item in range(images.shape[0])], dim=0)
            data['map_orig'] = to_device(map_orig, device)
            map_shift = torch.cat([annotations[item]['map_shift'].unsqueeze(0) for item in range(images.shape[0])], dim=0)
            data['map_shift'] = to_device(map_shift, device)
            loss_md, loss_dis, loss_res, loss_jlabel, loss_joff, loss_pos, loss_neg = model(data, targets, metas)
            loss = loss_md + loss_dis + loss_res + 0.25 * loss_joff + loss_pos + loss_neg + loss_jlabel
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss / len(train_dataset)
            losst_md += loss_md / len(train_dataset)
            losst_dis += loss_dis / len(train_dataset)
            losst_res += loss_res / len(train_dataset)
            losst_jlabel += loss_jlabel / len(train_dataset)
            losst_joff += loss_joff / len(train_dataset)
            losst_pos += loss_pos / len(train_dataset)
            losst_neg += loss_neg / len(train_dataset)

        current_lr = optimizer.param_groups[0]['lr']
        with open('output_log.txt', 'a') as f:
            print(
                f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f}   \n", file=f
            )
            print(
                f"loss_md : {losst_md} - loss_dis : {losst_dis:.4f} - loss_res : {losst_res:.4f}  - loss_joff : {losst_joff} - loss_jlabel : {losst_jlabel}\n", file=f
            )
            print(
                f"loss_pos : {losst_pos:.4f} - loss_neg : {losst_neg:.4f} - lr : {current_lr:.4f} \n", file=f
            )

            if epoch % 2 and epoch>50:
                model.eval()
                with torch.no_grad():
                    for name, dataset in test_dataset:
                        results = []
                        annotations_dict = {}
                        for i, (images, annotations) in enumerate(tqdm(dataset)):
                            ann = annotations if hasattr(annotations, 'keys') else annotations[0]
                            img = {}
                            img['img'] = images
                            map_orig = ann['map_orig'].unsqueeze(0)
                            img['map_orig'] = to_device(map_orig, device)
                            map_shift = ann['map_shift'].unsqueeze(0)
                            img['map_shift'] = to_device(map_shift, device)
                            annotations_dict[ann['filename']] = ann
                            img = to_device(img, device)
                            annotations = to_device(annotations, device)
                            lines_final, line_logits, juncs_final, juncs_logits, output, extra_info = \
                                model.forward_test(img, annotations)
                            output = to_device(output, 'cpu')
                            for k in output.keys():
                                if isinstance(output[k], torch.Tensor):
                                    output[k] = output[k].tolist()
                            results.append(output)

                        sAP, jAP = eval_sap(results, annotations_dict, epoch)
                        out_sAP = sAP['label']
                        out_jAP = jAP['label']
                        print(out_sAP['mean'], file=f)
                        print(out_jAP['mean'], file=f)



        save_path = f'./outputs/checkpoint/latest_swin_hawp_{epoch}.pth'
        checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, save_path)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Layout SRW Training')

    parser.add_argument("--config-file",
                        default='./config-files/Pred-SRW-S3D.yaml',
                        type=str,
                        )
    parser.add_argument("--seed",
                        default=2,
                        type=int)
    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER
                        )

    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    with open('output_log.txt', 'w') as file:
        file.write("==="*10)
    train(cfg)

