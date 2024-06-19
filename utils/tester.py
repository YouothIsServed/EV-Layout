import utils.metric_evaluation as me
from utils.labels import LabelMapper
import os.path as osp
from tabulate import tabulate
import torch
import matplotlib.pyplot as plt
from utils.visualization import ImagePlotter

LINE_LABELS = ['invalid', 'wall', 'floor', 'ceiling', 'window', 'door']
JUNCTION_LABELS = ['invalid', 'false', 'proper']
DISABLE_CLASSES = False

lm = LabelMapper(LINE_LABELS, JUNCTION_LABELS, disable=DISABLE_CLASSES)
img_viz = ImagePlotter(lm.get_line_labels(), lm.get_junction_labels())
plot_dir = './outputs'

def eval_sap(results, annotations_dict, epoch):
    thresholds = [5, 10, 15]  # <<<<<
    rcs, pcs, sAP = me.evalulate_sap(results, annotations_dict, thresholds, lm.get_line_labels())
    for m_type, thres_dict in sAP.items():
        for t in thres_dict:
            try:
                fig = img_viz.plot_ap(rcs[m_type][t], pcs[m_type][t], sAP[m_type][t], t,
                                           AP_string=fr'\mathrm{{sAP}}')
                fig_path = osp.join(plot_dir, 'E{:02}_sAP_{}_{}.pdf'.format(epoch, m_type, t))
                plt.savefig(fig_path)
            except KeyError:
                fig = None
    thresholds = [0.5, 1.0, 2.0]
    rcs, pcs, jAP = me.evalulate_jap(results, annotations_dict, thresholds, lm.get_junction_labels())
    ap_str = {'valid': r'\mathrm{{j}}_1\mathrm{{AP}}',
              'label': r'\mathrm{{j}}_3\mathrm{{AP}}',
              'label_line_valid': r'\mathrm{{j}}_2\mathrm{{AP}}'}
    for m_type, thres_dict in jAP.items():
        dstr = ap_str[m_type]
        for t in thres_dict:
            try:
                fig = img_viz.plot_ap(rcs[m_type][t], pcs[m_type][t], jAP[m_type][t], t, AP_string=dstr)
                fig_path = osp.join(plot_dir, 'E{:02}_jAP_{}_{}.pdf'.format(epoch, m_type, t))
                plt.savefig(fig_path)
            except KeyError:
                fig = None
    return sAP, jAP
