#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import spearmanr, pearsonr

from utils import register as R

from .abs_trainer import Trainer



@R.register('ConfidenceTrainer')
class ConfidenceTrainer(Trainer):

    ########## Override start ##########
    def __init__(self, model, train_loader, valid_loader, config: dict,  save_config: dict):
        super().__init__(model, train_loader, valid_loader, config, save_config)
        self.max_step = self.config.max_epoch * len(self.train_loader)

    def train_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx, val=False)

    def valid_step(self, batch, batch_idx):
        if hasattr(self.model, 'infer'): model = self.model
        else: model = self.model.module
        confidence = model.infer(batch)

        self.val_pred_target.append((
            confidence.detach(), confidence.detach(), batch['agg_label'], ['all' for _ in confidence]
        ))
        
        return self.share_step(batch, batch_idx, val=True)

    def _train_epoch_begin(self, device):
        # reform batch, with new random batches
        self.train_loader.dataset.update_epoch()
        self.train_loader.dataset._form_batch()
        self.val_pred_target = []
        return super()._train_epoch_begin(device)

    def _aggregate_val_metric(self, metric_arr):
        name2pred, name2label, name2conf = {}, {}, {}
        for batch_p, batch_conf, batch_gt, batch_name in self.val_pred_target:
            for p, c, gt, name in zip(batch_p, batch_conf, batch_gt, batch_name):
                if name not in name2pred: name2pred[name] = []
                if name not in name2label: name2label[name] = []
                if name not in name2conf: name2conf[name] = []
                name2pred[name].append(p)
                name2label[name].append(gt)
                name2conf[name].append(c)

        for name in name2pred:
            pred, target = torch.stack(name2pred[name]), torch.stack(name2label[name])
            binding_mask = target < 5

            # on positive data
            pred, target = pred[binding_mask], target[binding_mask]
            pcc = pearsonr(pred.cpu().numpy(), target.cpu().numpy())[0]
            spm = spearmanr(pred.cpu().numpy(), target.cpu().numpy()).correlation

            self.log(f'PCC/{name}', pcc, None, val=True)
            self.log(f'SPCC/{name}', spm, None, val=True)
            
            # distinguishing negative data
            if torch.sum(~binding_mask)  == 0:
                continue # no negative data
            th = 0.5 # 5A
            confidence = 1 - torch.stack(name2conf[name]) / 10.0
            bin_label = binding_mask.long()
            auroc = roc_auc_score(bin_label.cpu().numpy(), confidence.cpu().numpy())
            f1 = f1_score(bin_label.cpu().numpy(), (confidence > th).long().cpu().numpy())
            acc = ((confidence > th) == bin_label).sum() / len(bin_label)
            pred_pos_label = bin_label[confidence > th]
            label_pos_pred = confidence[bin_label.bool()] > th
            prec = pred_pos_label.sum() / (len(pred_pos_label) + 1e-10)
            recall = label_pos_pred.sum() / (len(label_pos_pred) + 1e-10)
            self.log(f'auroc/{name}', auroc, None, val=True)
            self.log(f'f1/{name}', f1, None, val=True)
            self.log(f'acc/{name}', acc, None, val=True)
            self.log(f'precision/{name}', prec, None, val=True)
            self.log(f'recall/{name}', recall, None, val=True)

        # return super()._aggregate_val_metric(metric_arr)
        return prec

    ########## Override end ##########

    def share_step(self, batch, batch_idx, val=False):
        loss = self.model(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            atom_positions=batch['atom_positions'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
            label=batch['label'])

        log_type = 'Validation' if val else 'Train'

        self.log(f'Loss/{log_type}', loss, batch_idx, val, batch_size=len(batch['label']))

        if not val:
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.log('lr', lr, batch_idx, val)

        return loss