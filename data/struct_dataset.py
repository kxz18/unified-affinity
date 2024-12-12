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
from .interface_dataset import InterfaceDataset

'''
Suitable for:
    - PDBbind
    - PPAB
    - SAbDab
'''

@R.register('StructDataset')
class StructDataset(InterfaceDataset):

    def get_id(self, idx):
        return self._indexes[idx][0]

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
        rec_blocks_tuple, lig_blocks_tuple = super(InterfaceDataset, self).__getitem__(idx)
        rec_blocks = [Block.from_tuple(tup) for tup in rec_blocks_tuple]
        lig_blocks = [Block.from_tuple(tup) for tup in lig_blocks_tuple]
        item = blocks_to_data_simple(rec_blocks, lig_blocks)
        
        rmsd = self.metadata[idx].rmsd
        if not isinstance(rmsd, float): rmsd = 100.0 # negative
        item['label'] = rmsd
        return item


if __name__ == '__main__':
    import sys
    dataset = StructDataset(sys.argv[1])
    print(dataset[0])
    print(dataset.get_id(0))