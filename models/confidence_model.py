#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean

from utils import register as R

from .predictor import PredictionModel, ReturnValue


def discretize(values, n_bins, min_val, max_val):
    """
    Discretize float values into n_bins with two extra bins for extremes.
    
    Args:
        values (torch.Tensor): Input tensor of float values.
        n_bins (int): Number of equal-span bins.
        min_val (float): Minimum value for binning range.
        max_val (float): Maximum value for binning range.

    Returns:
        torch.Tensor: Tensor of discretized values (bin indices).
    """
    # Compute the bin edges
    bin_edges = torch.linspace(min_val, max_val, n_bins + 1, device=values.device)  # Equal-span bins

    # Initialize the bin indices to 0 (extremely small values)
    bin_indices = torch.zeros_like(values, dtype=torch.long)

    # Assign extremely large values to the last bin index (n_bins + 1)
    bin_indices[values > max_val] = n_bins + 1

    # Assign values within the range to appropriate bins
    within_range_mask = (values >= min_val) & (values <= max_val)
    bin_indices[within_range_mask] = torch.bucketize(values[within_range_mask], bin_edges, right=False)

    bin_vals = F.pad(bin_edges, pad=(1, 0), value=0.0)
    return bin_indices, bin_vals


@R.register('ConfidenceModel')
class ConfidenceModel(PredictionModel):

    def __init__(self, model_type, hidden_size, n_channel, n_bins=50, min_val=0, max_val=10.0, residue_level=False, **kwargs) -> None:
        super().__init__(model_type, hidden_size, n_channel, **kwargs)
        self.n_bins = n_bins
        self.min_val = min_val
        self.max_val = max_val
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_bins + 2) # 1 for <min, 1 for > max
        )
        _, bin_vals = discretize(
            torch.zeros(1), self.n_bins, self.min_val, self.max_val
        )
        self.register_buffer('bin_vals', bin_vals)
        self.residue_level = residue_level
    
    def forward(self, Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label, return_noise=False) -> ReturnValue:
        return_value = super().forward(Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label, return_noise)
        label, _ = discretize(label, self.n_bins, self.min_val, self.max_val)
        if self.residue_level:
            confidence = self.confidence_head(return_value.block_repr)
            confidence = confidence[segment_ids == 1]
        else:
            confidence = self.confidence_head(return_value.graph_repr) # [n_bins + 2]
        return F.cross_entropy(confidence, label) + 0 * return_value.energy.sum()
    
    def infer(self, batch):
        return_value = super().forward(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            atom_positions=batch['atom_positions'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
            label=None
        )
        if self.residue_level:
            lig_mask = batch['segment_ids'] == 1
            confidence = self.confidence_head(return_value.block_repr)
            confidence = confidence[lig_mask] # [Nlig, n_bins + 2]
            confidence = F.softmax(confidence, dim=-1) # [Nlig, n_bins + 2]
            confidence = (self.bin_vals.unsqueeze(0) * confidence).sum(-1) # [Nlig]
            confidence = scatter_mean(confidence, return_value.batch_id[lig_mask], dim=0)
        else:
            confidence = self.confidence_head(return_value.graph_repr)
            confidence = F.softmax(confidence, dim=-1) # [bs, n_bins + 2]
            confidence = (self.bin_vals.unsqueeze(0) * confidence).sum(-1)

        return confidence


if __name__ == '__main__':
    # Example usage
    values = torch.tensor([-1.5, 0.2, 1.5, 3.5, 4.2])  # Input float values
    n_bins = 50  # Number of equal-span bins
    min_val = 0.0  # Minimum value for binning range
    max_val = 10.0  # Maximum value for binning range

    bin_indices = discretize(values, n_bins, min_val, max_val)
    print(bin_indices)  # Output: [0, 1, 2, 4, 4]