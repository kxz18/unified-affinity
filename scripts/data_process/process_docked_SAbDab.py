#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import json
import shutil
import argparse
from functools import partial

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from utils.logger import print_log
from data.converter.blocks_to_data import blocks_to_data
from data.converter.sdf_to_list_blocks import sdf_to_list_blocks
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.converter.blocks_interface import blocks_interface
from data.mmap_dataset import create_mmap


def parse():
    parser = argparse.ArgumentParser(description='Process PDBBind')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory of raw data of general set and refined set')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--interface_dist', type=float, default=8.0,
                        help='Residues who has atoms with distance below this threshold are considered in the complex interface')
    return parser.parse_args()


def parse_index(data_dir):
    data = {}

    for folder in os.listdir(data_dir):
        cur_data_dir = os.path.join(data_dir, folder)
        metadata_path = os.path.join(cur_data_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            continue
        metadata = json.load(open(metadata_path, 'r'))
        if 'pos' not in metadata['dock']:
            continue # the positive sample failed
        data[folder] = (metadata, cur_data_dir)
    
    return data


def process_iterator(indexes, if_th):
    for cnt, pdb_id in enumerate(indexes):
        metadata, data_dir = indexes[pdb_id]
        
        all_data, len_dict = {}, {}
        rec_chains, lig_chains = [], []
        for i, _ in enumerate(metadata['antigen_chains']):
            rec_chains.append(chr(ord('A') + i))
        for name in sorted(list(metadata['dock'].keys())):
            lig_chains = []
            seq_key = 'antibody_seqs' if name == 'pos' else 'seqs'
            for i, seq in enumerate(metadata['dock'][name][seq_key]):
                lig_chains.append(chr(ord(rec_chains[-1]) + i + 1))
            pdb_fname = os.path.join(data_dir, name + '.pdb')
            chain2blocks = pdb_to_list_blocks(pdb_fname, dict_form=True)
            rec_blocks, lig_blocks = [], []
            for c in rec_chains: rec_blocks.extend(chain2blocks[c])
            for c in lig_chains: lig_blocks.extend(chain2blocks[c])

            (rec_blocks, lig_blocks), _ = blocks_interface(rec_blocks, lig_blocks, if_th)
            if len(rec_blocks) == 0:
                print_log(f'{pdb_id}, {name} no interaction detected', level='WARN')
            
            data = (
                [block.to_tuple() for block in rec_blocks],
                [block.to_tuple() for block in lig_blocks],
            )
            all_data[name] = data
            n_atoms = 0
            for block in rec_blocks: n_atoms += len(block)
            for block in lig_blocks: n_atoms += len(block)
            len_dict[name] = n_atoms
        
        if len_dict['pos'] == 0:
            continue

        yield pdb_id, all_data, [len_dict, metadata], cnt + 1


def main(args):

    #if not os.path.exists(args.out_dir):
    if True:
        print_log(f'Generating data from {args.data_dir}')
        entries = parse_index(args.data_dir)
        print_log(f'{len(entries)} entries')
        create_mmap(
            process_iterator(
                entries, args.interface_dist
            ), args.out_dir, len(entries)
        )
    else: print_log(f'{args.out_dir} exists, skip')

    print_log('Finished!')


if __name__ == '__main__':
    np.random.seed(12)
    main(parse())