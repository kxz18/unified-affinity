#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import ast
import json
from typing import Optional

from easydict import EasyDict
import torch

from data.format import Block
from utils import register as R
from data.converter.blocks_to_data import blocks_to_data_simple

from .mmap_dataset import MMAPDataset

'''
Suitable for:
    - PDBbind
    - PPAB
    - SAbDab
'''

@R.register('InterfaceDataset')
class InterfaceDataset(MMAPDataset):

    def __init__(
            self,
            mmap_dir: str,
            specify_data: Optional[str] = None,
            specify_index: Optional[str] = None,
        ) -> None:
        super().__init__(mmap_dir, specify_data, specify_index)
        self.mmap_dir = mmap_dir
        self.metadata = [EasyDict(ast.literal_eval(props[1])) for props in self._properties]

    def get_id(self, idx: int):
        return self.metadata[idx].id

    def get_len(self, idx: int):
        return int(self._properties[idx][0])

    def get_summary(self, idx: int):
        return self.metadata[idx]

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
        rec_blocks_tuple, lig_blocks_tuple = super().__getitem__(idx)
        rec_blocks = [Block.from_tuple(tup) for tup in rec_blocks_tuple]
        lig_blocks = [Block.from_tuple(tup) for tup in lig_blocks_tuple]
        item = blocks_to_data_simple(rec_blocks, lig_blocks)
        item['label'] = self.metadata[idx].pkd
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
                try:
                    results[key] = torch.cat(values, dim=0)
                except RuntimeError:
                    print(key, [v.shape for v in values])
        return results


if __name__ == '__main__':
    import sys
    dataset = InterfaceDataset(sys.argv[1])
    print(dataset[0])