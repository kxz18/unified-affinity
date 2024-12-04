#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse

from tqdm import tqdm
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from visualize.lineplot import lineplot


def parse():
    parser = argparse.ArgumentParser(description='positive rate curve')
    parser.add_argument('--datasets', type=str, nargs='+', required=True)
    parser.add_argument('--out_dir', type=str, default=os.path.join(os.path.dirname(__file__), 'images', 'curve'))
    parser.add_argument('--model_results', type=str, default=None, nargs='+')
    return parser.parse_args()


def load_labels(path):
    with open(os.path.join(path, 'summary.tsv'), 'r') as fin:
        lines = fin.read().strip().split('\n')
    id2labels = {}
    head = lines[0].split('\t')
    for line in lines[1:]:
        line = line.split('\t')
        id, kd, rosetta_dg, foldx_dg = line[0:4]
        rosetta_dg = float(rosetta_dg)
        foldx_dg = float(foldx_dg)
        try: kd = float(kd)
        except ValueError: kd = None
        id2labels[id] = {
            'rosetta': rosetta_dg,
            'foldx': foldx_dg,
            'kd': kd
        }
    return id2labels


def normalize_dg(dgs):
    norm = min(dgs)
    return [dg / norm for dg in dgs]


def load_model_data(labels, res_path):
    with open(res_path, 'r') as fin:
        lines = fin.readlines()
    kds, dgs, confs = [], [], []
    for line in lines[1:]:
        id, kd_pred, conf = line.strip().split('\t')
        kd_pred, conf = float(kd_pred), float(conf)
        kds.append(labels[id]['kd'])
        dgs.append(kd_pred)
        confs.append(conf)
    return kds, dgs, confs


def get_data(kds, confidence, pred_kd, min_val=0.0, max_val=1.0):

    cutoff = np.linspace(min_val, max_val, 100)
    data = {
        'positive rate (%)': [],
        'kd corr': [],
        'number': [],
        'cutoff': cutoff.tolist()
    }

    # positive rate and correlation
    for th in cutoff:
        idx = [i for i in range(len(kds)) if confidence[i] > th]
        pos_kds = [kds[i] for i in idx if kds[i] is not None]
        pos_pred = [pred_kd[i] for i in idx if kds[i] is not None]
        
        pos_number = len(pos_kds)
        pos_rate = float('nan') if len(idx) == 0 else 100 * pos_number / len(idx)

        corr = float('nan') if len(pos_kds) < 3 else pearsonr(pos_kds, pos_pred).correlation

        data['positive rate (%)'].append(pos_rate)
        data['kd corr'].append(corr)
        data['number'].append(len(idx))

    return data


def draw_curve(data, name, out_dir, hue='target'):

    min_val, max_val = min(data['cutoff']), max(data['cutoff'])

    def post_edit(ax):
        ax.set_xlim(min_val - max_val * 0.05, max_val + max_val * 0.05)

    plt.clf()
    lineplot(data, x='cutoff', y='positive rate (%)', hue=hue, save_path=os.path.join(out_dir, name + '_pos_rate.png'), post_edit_func=post_edit)
    plt.clf()
    lineplot(data, x='cutoff', y='kd corr', hue=hue, save_path=os.path.join(out_dir, name + '_kd_corr.png'), post_edit_func=post_edit)
    plt.clf()
    lineplot(data, x='cutoff', y='number', hue=hue, save_path=os.path.join(out_dir, name + '_num.png'), post_edit_func=post_edit)
    plt.clf()


def main(args):
    
    # input/output
    # name = os.path.basename(args.dataset.strip(os.path.sep))
    # out_dir = os.path.join(args.out_dir, name)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    all_data = {
        'rosetta': { 'target': [] },
        'foldx': { 'target': [] }
    }
    struct_types = ['raw', 'openmm_relax', 'rosetta_relax', 'openmm_foldx_relax']
    if args.model_results is not None:
        for path in args.model_results:
            model_name = os.path.basename(path.strip(os.path.sep))
            for stype in struct_types:
                all_data[model_name + '_' + stype] = { 'target': [] }

    def update(data, metric_name, target_name):
        for key in data:
            if key not in all_data[metric_name]: all_data[metric_name][key] = []
            all_data[metric_name][key].extend(data[key])
        all_data[metric_name]['target'].extend([target_name for _ in data[key]])

    for path in tqdm(args.datasets):
        name = os.path.basename(path.strip(os.path.sep))

        # load labels
        labels = load_labels(path)

        kds = [labels[id]['kd'] for id in labels]

        sub_out_dir = os.path.join(out_dir, name)
        os.makedirs(sub_out_dir, exist_ok=True)

        metric_data = { 'metric': [] }

        # rosetta
        dgs = normalize_dg([labels[id]['rosetta'] for id in labels])
        data = get_data(kds, dgs, dgs)
        metric_data.update(data)
        metric_data['metric'].extend(['rosetta' for _ in data['cutoff']])
        update(data, 'rosetta', name)
        
        # foldx
        dgs = normalize_dg([labels[id]['foldx'] for id in labels])
        data = get_data(kds, dgs, dgs)
        for key in data: metric_data[key].extend(data[key])
        metric_data['metric'].extend(['foldx' for _ in data['cutoff']])
        update(data, 'foldx', name)

        # your metric
        for model_path in args.model_results:
            model_name = os.path.basename(model_path.strip(os.path.sep))
            for stype in struct_types:
                res_path = os.path.join(model_path, name, stype + '.txt')
                kds, dgs, confs = load_model_data(labels, res_path)
                data = get_data(kds, confs, dgs)
                for key in data: metric_data[key].extend(data[key])
                metric_name = model_name + '_' + stype
                metric_data['metric'].extend([metric_name for _ in data['cutoff']])
                update(data, metric_name, name)
        
        draw_curve(metric_data, '', sub_out_dir, hue='metric')


    # draw figure
    for metric_name in all_data:
        draw_curve(all_data[metric_name], metric_name, out_dir)


if __name__ == '__main__':
    main(parse())