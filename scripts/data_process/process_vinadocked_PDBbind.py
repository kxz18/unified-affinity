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
    parser.add_argument('--interface_dist_th', type=float, default=8.0,
                        help='Residues who has atoms with distance below this threshold are considered in the complex interface')
    return parser.parse_args()


def parse_index(fpath, quality):
    pdb_dir = os.path.dirname(fpath)
    with open(fpath, 'r') as fin:
        lines = fin.readlines()
    
    data = {}
    for line in lines:
        name, status = line.strip().split('\t')
        if status == 'failed':
            continue
        metadata = json.load(open(os.path.join(pdb_dir, name, 'metadata.json'), 'r'))
        if 'pos' not in metadata['dock']:
            continue # the positive sample failed to dock
        metadata['quality'] = quality
        metadata['pKd'] = metadata['pK']
        del metadata['pK']
        data[name] = (metadata, os.path.join(pdb_dir, name))
    return data


def process_iterator_PL(indexes, if_th):
    for cnt, pdb_id in enumerate(indexes):
        metadata, data_dir = indexes[pdb_id]
        prot_fname = os.path.join(data_dir, 'receptor.pdb')
        sm_fname = os.path.join(data_dir, 'ligands.sdf')

        prot_list_blocks = pdb_to_list_blocks(prot_fname)
        sms = sdf_to_list_blocks(sm_fname, dict_form=False, silent=True)
        sm_dicts = {}
        for name, mol in zip(metadata['dock'], sms):
            sm_dicts[name] = mol
        rec_blocks = []
        for blocks in prot_list_blocks:
            rec_blocks.extend(blocks)

        all_data, len_dict = {}, {}
        for name in sorted(list(sm_dicts.keys())):
            (pocket_blocks, _), _ = blocks_interface(rec_blocks, sm_dicts[name][0], if_th)
            if len(pocket_blocks) == 0:
                print_log(f'{pdb_id}, {name} no interaction detected', level='WARN')
            
            data = (
                [block.to_tuple() for block in pocket_blocks],
                [block.to_tuple() for block in sm_dicts[name][0]],
            )
            all_data[name] = data
            n_atoms = 0
            for block in pocket_blocks: n_atoms += len(block)
            for block in sm_dicts[name][0]: n_atoms += len(block)
            len_dict[name] = n_atoms

        yield pdb_id, all_data, [len_dict, metadata], cnt + 1


def main(args):

    if not os.path.exists(args.out_dir):
        print_log(f'Generating data from {args.data_dir}')
        # refined set
        indexes = parse_index(os.path.join(args.data_dir, 'processed_refined_set', 'done.log'), 'refined')
        # indexes2 = parse_index(os.path.join(args.data_dir, 'processed_general_set', 'done.log'), 'general')
        # for name in indexes2:
        #     assert name not in indexes, name
        #     indexes[name] = indexes2[name]
        create_mmap(
            process_iterator_PL(
                indexes, args.interface_dist_th
            ), args.out_dir, len(indexes)
        )
    else: print_log(f'{args.out_dir} exists, skip')

    print_log('Finished!')


if __name__ == '__main__':
    np.random.seed(12)
    main(parse())