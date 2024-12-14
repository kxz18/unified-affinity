#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import ast
import json
from typing import Optional

from easydict import EasyDict
import torch
import numpy as np

from data.format import Block
from utils import register as R
from data.converter.blocks_to_data import blocks_to_data_simple

from .mmap_dataset import MMAPDataset
from .interface_dataset import InterfaceDataset

'''
Suitable for:
    - PDBbind
    - PPAB
    - SAbDab
'''


def resample_with_rmsd(rmsds, n_bins=50, min_val=0, max_val=10):
    rmsds = np.array(rmsds)
    bins = np.linspace(min_val, max_val, num=n_bins+1)
    rmsds = np.digitize(rmsds, bins, right=False)
    cnts = np.bincount(rmsds)
    p = 1 / (cnts + 1e-5)
    p[cnts == 0] = 0
    p = p / p.sum()
    idx = np.arange(len(rmsds))
    p = p[rmsds]
    p = p / p.sum()

    return np.random.choice(idx, size=len(rmsds), replace=True, p=p).tolist()


def resample_with_binary(rmsds, th):
    rmsds = np.array(rmsds)
    label = rmsds < th
    pos_cnt = label.sum()
    p = np.zeros_like(rmsds)
    p[label] = 1.0 / pos_cnt
    p[~label] = 1.0 / (len(rmsds) - pos_cnt)
    p = p / p.sum()
    idx = np.arange(len(rmsds))
    return np.random.choice(idx, size=len(rmsds), replace=True, p=p).tolist()


@R.register('StructDataset')
class StructDataset(InterfaceDataset):

    def __init__(self, mmap_dir, specify_data = None, specify_index = None, resample = False, residue_level = False):
        super().__init__(mmap_dir, specify_data, specify_index)
        self.dynamic_idx = [i for i in range(len(self))]
        self.resample = resample
        self.residue_level = residue_level
        self.update_epoch()

    def get_id(self, idx):
        idx = self.dynamic_idx[idx]
        return self._indexes[idx][0]
    
    def get_len(self, idx: int):
        idx = self.dynamic_idx[idx]
        return int(self._properties[idx][0])

    def get_summary(self, idx: int):
        idx = self.dynamic_idx[idx]
        return self.metadata[idx]

    def update_epoch(self):
        if not self.resample: return
        rmsds = [self._total_rmsd(metadata.rmsd) for metadata in self.metadata]
        self.dynamic_idx = resample_with_rmsd(rmsds)
    
    def _total_rmsd(self, rmsd):
        if isinstance(rmsd, float) or rmsd is None: return rmsd
        return rmsd[0] # (rmsd, residue_rmsd)

    def __getitem__(self, idx: int):
        '''
        an example of the returned data
        {
            'X': [Natom, n_channel, 3],
            'B': [Nblock],
            'A': [Natom],
            'atom_positions': [Natom],
            'block_lengths': [Natom]
            'segment_ids': [Nblock]
            'label': [1]
        }
        '''
        idx = self.dynamic_idx[idx]
        rec_blocks_tuple, lig_blocks_tuple = super(InterfaceDataset, self).__getitem__(idx)
        rec_blocks = [Block.from_tuple(tup) for tup in rec_blocks_tuple]
        lig_blocks = [Block.from_tuple(tup) for tup in lig_blocks_tuple]
        item = blocks_to_data_simple(rec_blocks, lig_blocks)
        
        rmsd = self.metadata[idx].rmsd
        if self.residue_level:
            item['label'] = torch.tensor(rmsd[1], dtype=torch.float)
            assert len(item['label']) == len(lig_blocks)
            item['agg_label'] = rmsd[0]
        else:
            # if not isinstance(rmsd, float): rmsd = 100.0 # negative
            if rmsd is None: rmsd = 100.0 # negative
            elif isinstance(rmsd, tuple): rmsd = rmsd[0]
            item['label'] = rmsd
            item['agg_label'] = rmsd
        return item
    
    def collate_fn(self, batch):
        results = {}
        for key in batch[0]:
            values = [item[key] for item in batch]
            if values[0] is None:
                results[key] = None
                continue
            if key == 'lengths':
                results[key] = torch.tensor(values, dtype=torch.long)
            elif key == 'label':
                if self.residue_level:
                    results[key] = torch.cat(values, dim=0)
                else:
                    results[key] = torch.tensor(values, dtype=torch.float)
            elif key == 'agg_label':
                results[key] = torch.tensor(values, dtype=torch.float)
            else:
                try:
                    results[key] = torch.cat(values, dim=0)
                except RuntimeError:
                    print(key, [v.shape for v in values])
        return results


@R.register('BinaryDataset')
class BinaryDataset(InterfaceDataset):

    def __init__(self, mmap_dir, specify_data = None, specify_index = None, resample = False, pos_th = 5.0):
        super().__init__(mmap_dir, specify_data, specify_index)
        self.dynamic_idx = [i for i in range(len(self))]
        self.resample = resample
        self.pos_th = pos_th

        self.update_epoch()

    def get_id(self, idx):
        idx = self.dynamic_idx[idx]
        return self._indexes[idx][0]
    
    def get_len(self, idx: int):
        idx = self.dynamic_idx[idx]
        return int(self._properties[idx][0])

    def get_summary(self, idx: int):
        idx = self.dynamic_idx[idx]
        return self.metadata[idx]

    def update_epoch(self):
        if not self.resample: return
        rmsds = [self._total_rmsd(metadata.rmsd) for metadata in self.metadata]
        self.dynamic_idx = resample_with_binary(rmsds, self.pos_th)

    def _total_rmsd(self, rmsd):
        if isinstance(rmsd, float) or rmsd is None: return rmsd
        return rmsd[0] # (rmsd, residue_rmsd)

    def __getitem__(self, idx: int):
        '''
        an example of the returned data
        {
            'X': [Natom, n_channel, 3],
            'B': [Nblock],
            'A': [Natom],
            'atom_positions': [Natom],
            'block_lengths': [Natom]
            'segment_ids': [Nblock]
            'label': [1]
        }
        '''
        idx = self.dynamic_idx[idx]
        rec_blocks_tuple, lig_blocks_tuple = super(InterfaceDataset, self).__getitem__(idx)
        rec_blocks = [Block.from_tuple(tup) for tup in rec_blocks_tuple]
        lig_blocks = [Block.from_tuple(tup) for tup in lig_blocks_tuple]
        item = blocks_to_data_simple(rec_blocks, lig_blocks)
        
        rmsd = self._total_rmsd(self.metadata[idx].rmsd)
        if not isinstance(rmsd, float): rmsd = 100.0 # negative
        item['label'] = rmsd < self.pos_th
        return item


if __name__ == '__main__':
    import sys
    dataset = BinaryDataset(sys.argv[1], resample=True)
    print(dataset[0])
    print(dataset.get_id(0))