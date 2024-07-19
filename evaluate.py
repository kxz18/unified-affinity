#!/usr/bin/python
# -*- coding:utf-8 -*-
import json
import argparse
from functools import partial
from easydict import EasyDict

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression

from utils.logger import print_log


def rmse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(((y_pred - y_true) ** 2).mean())


def mae(y_true, y_pred, reduction=True):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    val = np.abs(y_true - y_pred)
    if reduction:
        return val.mean()
    return val


def minimized_rmse(y_true, y_pred):
    # from https://github.com/luost26/RDE-PPI/blob/main/rde/utils/skempi.py#L135
    y_true, y_pred = np.array(y_true), np.array(y_pred)[:, None]
    reg = LinearRegression().fit(y_pred, y_true)
    pred_corrected = reg.predict(y_pred)
    return rmse(y_true, pred_corrected)


def minimized_mae(y_true, y_pred):
    # from https://github.com/luost26/RDE-PPI/blob/main/rde/utils/skempi.py#L135
    y_true, y_pred = np.array(y_true), np.array(y_pred)[:, None]
    reg = LinearRegression().fit(y_pred, y_true)
    pred_corrected = reg.predict(y_pred)
    return mae(y_true, pred_corrected)


def continuous_auroc(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = y_true > 0
    return roc_auc_score(y_true, y_pred)


def perstruct_corr(y_true, y_pred, struct_ids, _type='pearson'):
    assert len(y_pred) == len(struct_ids)
    results = {}
    for i, _id in enumerate(struct_ids):
        if _id not in results:
            results[_id] = [[], []]
        results[_id][0].append(y_true[i])
        results[_id][1].append(y_pred[i])

    corr = pearsonr if _type == 'pearson' else spearmanr
    valid_ids = []
    for _id in results:
        if len(results[_id][0]) < 10:  # from https://github.com/luost26/RDE-PPI/blob/main/rde/utils/skempi.py
            continue
        results[_id] = corr(results[_id][0], results[_id][1]).statistic
        valid_ids.append(_id)

    return np.mean([results[_id] for _id in valid_ids])


def avg_rank(bin_labels, confidence, ids):
    id2res = {}
    for gt, c, i in zip(bin_labels, confidence, ids):
        _id = i.split('-')[0]
        if _id not in id2res: id2res[_id] = { 'label': [], 'conf': [] }
        id2res[_id]['label'].append(gt)
        id2res[_id]['conf'].append(c)
    pool_size = max([len(id2res[_id]['label']) for _id in id2res])
    ranks = []
    for _id in id2res:
        cur_ranks = []
        idxs = list(range(len(id2res[_id]['label'])))
        sorted_label = sorted(idxs, key=lambda i: id2res[_id]['conf'][i], reverse=True)
        for r, i in enumerate(sorted_label):
            if id2res[_id]['label'][i] > 0:
                cur_ranks.append((r + 1) / pool_size)
        ranks.append(sum(cur_ranks) / (len(cur_ranks) + 1e-10))
    return sum(ranks) / len(ranks)


def parse():
    parser = argparse.ArgumentParser(description='Calculate evaluation metrics')
    parser.add_argument('--predictions', type=str, required=True, help='Path to the predicted results')
    return parser.parse_args()


def main(args):
    with open(args.predictions, 'r') as fin:
        preds = [EasyDict(json.loads(s)) for s in fin.readlines()]
    
    # categorized by data type
    type2preds = {}
    for pred in preds:
        _type = pred.type
        if _type not in type2preds: type2preds[_type] = []
        type2preds[_type].append(pred)

    # calculate metrics
    for _type in type2preds:
        print(f'Results for {_type}:')
        pred_pkd, label, conf, ids = [], [], [], []
        for pred in type2preds[_type]:
            pred_pkd.append(pred.pred)
            label.append(pred.label)
            conf.append(pred.confidence)
            ids.append(pred.id)
        pred_pkd, label, conf = np.array(pred_pkd), np.array(label), np.array(conf)
        # positive complexes
        pos_mask = label > 0
        metrics = {
            'Pearson': pearsonr,
            'Spearman': spearmanr,
            'RMSE': rmse,
            'MAE': mae
        }
        for met_name in metrics:
            print(f'\t{met_name}: {metrics[met_name](label[pos_mask], pred_pkd[pos_mask])}')

        # correlation of confidence with respect to labels
        metrics = {
            'Pearson': pearsonr,
            'Spearman': spearmanr,
        }
        for met_name in metrics:
            print(f'\tConfidence {met_name}: {metrics[met_name](label[pos_mask], conf[pos_mask])}')
        for met_name in metrics:
            print(f'\tConfidence vs pKd {met_name}: {metrics[met_name](pred_pkd[pos_mask], conf[pos_mask])}')

        # negative complexes
        binary_label = pos_mask.astype(np.int32) 
        if binary_label.sum() == binary_label.shape[0]: continue # no negative data
        metrics = {
            'AUROC': roc_auc_score,
            'AUPRC': average_precision_score,
        }
        for met_name in metrics:
            print(f'\t{met_name}: {metrics[met_name](binary_label, conf)}')
        
        # average ranks
        print(f'\tAvgRank: {avg_rank(binary_label, conf, ids)}')

        metrics = {
            'Precision': precision_score,
            'Recall': recall_score,
            'F1': f1_score
        }
        th = 0.7
        pred_binary_label = (conf > th).astype(np.int32)
        for met_name in metrics:
            print(f'\t{met_name}: {metrics[met_name](binary_label, pred_binary_label)}')



if __name__ == '__main__':
    main(parse())