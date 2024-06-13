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
    epoch = 999
    device = 'cuda:0'
    test_dataset = build_test_dataset(cfg, validation=False)

    model = EvRoomDetector().to(device)
    cpkt_name = './outputs/checkpoint/stage_4/latest_swin_hawp_50.pth'
    checkpoint = torch.load(cpkt_name)
    print('=======================')
    print(cpkt_name)
    print('=======================')
    model.load_state_dict(checkpoint['model'])

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

                np.save('img.npy', images)
                np.save('line.npy', lines_final)
                np.save('line_logti.npy', line_logits)

                if i == 455:
                    break



            # sAP, jAP = eval_sap(results, annotations_dict, epoch)
            # out_sAP = sAP['label']
            # out_jAP = jAP['label']
            # print(out_sAP['mean'])
            # print(out_jAP['mean'])


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
    train(cfg)
