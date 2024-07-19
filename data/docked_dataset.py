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

'''
Suitable for:
    - docked PDBbind
    - docked SAbDab
'''

@R.register('DockedDataset')
class DockedDataset(MMAPDataset):

    def __init__(
            self,
            mmap_dir: str,
            specify_data: Optional[str] = None,
            specify_index: Optional[str] = None,
            negative_rate: float=0.0,
            pos_rmsd_th: float=None,
            test_mode: bool=False # flatten dataset with all samples
        ) -> None:
        super().__init__(mmap_dir, specify_data, specify_index)
        self.mmap_dir = mmap_dir
        self._lengths = [ast.literal_eval(props[0]) for props in self._properties]
        self.metadata = [EasyDict(ast.literal_eval(props[1])) for props in self._properties]
        self.negative_rate = negative_rate
        self.pos_rmsd_th = pos_rmsd_th # positive pair with rmsd above th will also be seen as negative ones
        if self.pos_rmsd_th is not None:
            assert 'rmsd' in self.metadata[0]
            cnt = 0
            for metadata in self.metadata:
                if metadata['rmsd'] <= self.pos_rmsd_th:
                    cnt += 1
            print(f'{cnt}/{len(self.metadata)} satisfied RMSD threshold (<{self.pos_rmsd_th})')

        self.test_mode = test_mode
        self.dynamic_selections = [] # only for training
        self.update_epoch()
        
        # for test mode
        self.flat_idx = []
        for i, len_dict in enumerate(self._lengths):
            for key in sorted(list(len_dict.keys()), reverse=True): # pos, neg9, neg8, ...
                if len_dict[key] > 0: self.flat_idx.append((i, key))

    def __len__(self):
        if self.test_mode:
            return len(self.flat_idx)
        return super().__len__()

    def get_id(self, idx: int):
        if self.test_mode:
            i, key = self.flat_idx[idx]
            _id = self.metadata[i].id
            return f'{_id}-{key}'
        item_key = self.dynamic_selections[idx]
        _id = self.metadata[idx].id
        return f'{_id}-{item_key}'

    def get_len(self, idx: int):
        if self.test_mode:
            i, key = self.flat_idx[idx]
            return self._lengths[i][key]
        item_key = self.dynamic_selections[idx]
        return self._lengths[idx][item_key]

    def get_summary(self, idx: int):
        if self.test_mode:
            idx = self.flat_idx[idx][0]
        return self.metadata[idx]

    def update_epoch(self):
        if self.test_mode:
            return
        self.dynamic_selections = []
        for idx in range(len(self)):
            len_dict = self._lengths[idx]
            neg_names = [name for name in len_dict if 'neg' in name and len_dict[name] > 0]
            if (np.random.rand() < self.negative_rate and len(neg_names) > 0) or len_dict['pos'] == 0:
                # sample negatives
                neg = np.random.choice(neg_names)
                self.dynamic_selections.append(neg)
            else:
                self.dynamic_selections.append('pos')

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
        if self.test_mode:
            i, item_key = self.flat_idx[idx]
            data_dict = super().__getitem__(i)
        else:
            data_dict = super().__getitem__(idx)
            item_key = self.dynamic_selections[idx]
        rec_blocks_tuple, lig_blocks_tuple = data_dict[item_key]
        rec_blocks = [Block.from_tuple(tup) for tup in rec_blocks_tuple]
        lig_blocks = [Block.from_tuple(tup) for tup in lig_blocks_tuple]
        item = blocks_to_data_simple(rec_blocks, lig_blocks)

        metadata = self.get_summary(idx)
        if item_key == 'pos':
            if self.pos_rmsd_th is not None and metadata.rmsd > self.pos_rmsd_th:
                label = 0.0
            else:
                label = metadata.pKd
        else:
            label = 0.0
        item['label'] = label
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
                results[key] = torch.tensor(values, dtype=torch.float)
            else:
                results[key] = torch.cat(values, dim=0)
        return results


if __name__ == '__main__':
    import sys
    dataset = DockedDataset(sys.argv[1])
    print(dataset[0])