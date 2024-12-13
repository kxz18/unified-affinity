#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum

from utils import register as R

from .predictor import PredictionModel, ReturnValue



@R.register('BinaryModel')
class BinaryModel(PredictionModel):

    def __init__(self, model_type, hidden_size, n_channel, **kwargs) -> None:
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

        confidence = self.confidence_head(return_value.graph_repr)
        confidence = torch.sigmoid(confidence).squeeze(-1) # [bs]

        return F.binary_cross_entropy(confidence, label) + 0 * return_value.energy.sum()
    
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

        return confidence