#!/usr/bin/python
# -*- coding:utf-8 -*-
import ast

import numpy as np

from data.format import Block
from utils import register as R
from data.converter.blocks_to_data import blocks_to_data_simple
from .interface_dataset import InterfaceDataset


@R.register('WetlabDataset')
class WetlabDataset(InterfaceDataset):
    def __init__(self, mmap_dir, specify_data = None, specify_index = None, stype = 'raw', neg_ratio = 10.0, test_mode = False):
        super().__init__(mmap_dir, specify_data, specify_index)
        self.stype = stype
        self.neg_ratio = neg_ratio
        self.test_mode = test_mode
        self.update_epoch()

    def __getitem__(self, idx):
        idx = self.dynamic_idxs[idx]
        rec_blocks_tuple, lig_blocks_tuple = super(InterfaceDataset, self).__getitem__(idx)[self.stype]
        rec_blocks = [Block.from_tuple(tup) for tup in rec_blocks_tuple]
        lig_blocks = [Block.from_tuple(tup) for tup in lig_blocks_tuple]
        item = blocks_to_data_simple(rec_blocks, lig_blocks)
        item['label'] = self.metadata[idx].pkd if self.metadata[idx].pkd is not None else 0.0
        return item

    def get_len(self, idx):
        idx = self.dynamic_idxs[idx]
        return int(ast.literal_eval(self._properties[idx][0])[self.stype])

    def update_epoch(self):
        if self.test_mode:
            self.dynamic_idxs = [i for i in range(len(self))]
            return
        
        # split positive data and negative data
        pos_idxs, neg_idxs = [], []
        for i, metadata in enumerate(self.metadata):
            if metadata.pkd > 0: pos_idxs.append(i)
            else: neg_idxs.append(i)

        n_pos = int(len(self) * 1.0 / (1.0 + self.neg_ratio))
        n_neg = len(self) - n_pos
        print(f'pos: {n_pos}, neg: {n_neg}')
    
        # random sample
        pos_idxs = np.random.choice(pos_idxs, size=n_pos, replace=True)
        neg_idxs = np.random.choice(neg_idxs, size=n_neg, replace=True)

        self.dynamic_idxs = pos_idxs.tolist() + neg_idxs.tolist()
        np.random.shuffle(self.dynamic_idxs)
        