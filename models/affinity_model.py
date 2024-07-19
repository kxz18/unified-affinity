#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum

from utils import register as R

from .predictor import PredictionModel, ReturnValue


@R.register('AffinityModel')
class AffinityModel(PredictionModel):

    def __init__(self, model_type, hidden_size, n_channel, **kwargs) -> None:
        self.alpha = kwargs.pop('alpha', 0.183) # 0.9 probability - 1nm
        self.loss_weights = kwargs.pop('loss_weights', {
            'regression': 1.0,
            'classification': 1.0
        })
        super().__init__(model_type, hidden_size, n_channel, **kwargs)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label, return_noise=False) -> ReturnValue:
        return_value = super().forward(Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label, return_noise)
        pkd = -return_value.energy

        confidence = self.confidence_head(return_value.graph_repr)
        confidence = torch.sigmoid(confidence).squeeze(-1) # [bs]

        return self.get_loss(pkd, confidence, label)
    
    def infer(self, batch):
        return_value = super().forward(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            atom_positions=batch['atom_positions'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
            label=None
        )
        confidence = self.confidence_head(return_value.graph_repr)
        confidence = torch.sigmoid(confidence).squeeze(-1) # [bs]

        return -return_value.energy, confidence

    def get_loss(self, pred, confidence, label):
        binding_mask = label > 0
        reg_loss = F.mse_loss(pred[binding_mask], label[binding_mask]) # since we are supervising pK=-log_10(Kd), whereas the energy is RTln(Kd)
        if binding_mask.sum() == 0: reg_loss = 0

        target_prob = torch.clamp(torch.pow(label * (1.0 / 16.0), self.alpha), max=1.0) # affinity max 0.1pm, otherwise treat as probability = 1.0
        conf_loss = F.binary_cross_entropy(confidence, target_prob)
        
        loss_dict = {
            'regression': reg_loss,
            'classification': conf_loss,
        }

        total_loss = 0
        for key in loss_dict:
            total_loss = total_loss + self.loss_weights[key] * loss_dict[key]
        loss_dict['total_loss'] = total_loss
        return loss_dict
    

class NoisedAffinityModel(AffinityModel):
    def __init__(self, model_type, hidden_size, n_channel, sigma: float=0.1, **kwargs) -> None:
        super().__init__(model_type, hidden_size, n_channel, **kwargs)
        self.sigma = sigma

    def add_noise(self, X):
        return X + torch.randn_like(X) * self.sigma

    def forward(self, Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label, return_noise=False) -> ReturnValue:
        Z = self.add_noise(Z)
        return super().forward(Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label, return_noise)

    def infer(self, batch):
        batch['X'] = self.add_noise(batch['X'])
        return super().infer(batch)