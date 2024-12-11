#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import math
import ast
import argparse
from dataclasses import dataclass
from typing import List

from data.format import VOCAB
from utils.logger import print_log
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.converter.blocks_interface import blocks_interface
from data.mmap_dataset import create_mmap

def parse():
    parser = argparse.ArgumentParser(description='construct dataset')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--interface_dist', type=float, default=8.0,
                        help='Residues who has atoms with distance below this threshold are considered in the complex interface')
    return parser.parse_args()


@dataclass
class Entry:
    id: str
    kd: float
    pkd: float
    rosetta_dg: float
    foldx_dg: float
    sequence: str
    raw_id: str
    target: str


def parse_index(data_dir):
    entries = []
    for target_name in os.listdir(data_dir):
        with open(os.path.join(data_dir, target_name, 'summary.tsv'), 'r') as fin:
            lines = fin.readlines()
        for line in lines[1:]:
            line = line.strip().split('\t')
            entries.append(Entry(
                id=line[0],
                kd=(float(line[1]) * 1e-9) if line[1] != 'None' else None,
                pkd=-math.log((float(line[1]) * 1e-9), 10) if line[1] != 'None' else 0.0,
                rosetta_dg=float(line[2]),
                foldx_dg=float(line[3]),
                sequence=line[4],
                raw_id=line[5],
                target=target_name
            ))
    return entries


def process_iterator(data_dir: str, entries: List[Entry], interface_dist: float):
    structure_types = ['raw', 'openmm_relax', 'rosetta_relax', 'openmm_foldx_relax']
    for i, entry in enumerate(entries):

        data_map = {}
        n_atom_map = {}

        for stype in structure_types:
            pdb_path = os.path.join(data_dir, entry.target, stype, entry.id + '.pdb')
            chain2blocks = pdb_to_list_blocks(pdb_path, dict_form=True)

            _, rec_chains, lig_chains = entry.id.split('_')
            rec_chains, lig_chains = list(rec_chains), list(lig_chains)

            rec_blocks, lig_blocks = [], []
            for c in rec_chains: rec_blocks.extend(chain2blocks[c])
            for c in lig_chains: lig_blocks.extend(chain2blocks[c])

            (rec_blocks, lig_blocks), _ = blocks_interface(
                rec_blocks, lig_blocks, interface_dist
            )

            data = (
                [block.to_tuple() for block in rec_blocks],
                [block.to_tuple() for block in lig_blocks],
            )

            n_atoms = 0
            for block in rec_blocks: n_atoms += len(block)
            for block in lig_blocks: n_atoms += len(block)

            data_map[stype] = data
            n_atom_map[stype] = n_atoms

        properties = {
            'id': entry.id,
            'pkd': entry.pkd,
            'rec_chains': rec_chains,
            'lig_chains': lig_chains,
            'lig_seqs': [entry.sequence],
            'rosetta_dg': entry.rosetta_dg,
            'foldx_dg': entry.foldx_dg,
            'target': entry.target
        }

        yield entry.target + '_' + entry.id, data_map, [n_atom_map, properties], i + 1


def split(mmap_dir, valid_targets, test_targets):
    with open(os.path.join(mmap_dir, 'index.txt'), 'r') as fin:
        lines = fin.readlines()
    train_lines, valid_lines, test_lines = [], [], []
    for line in lines:
        item = ast.literal_eval(line.split('\t')[-1])
        if item['target'] in valid_targets: valid_lines.append(line.strip())
        elif item['target'] in test_targets: test_lines.append(line.strip())
        else: train_lines.append(line.strip())
    for name, lines in zip(['train', 'valid', 'test'], [train_lines, valid_lines, test_lines]):
        fout = open(os.path.join(mmap_dir, name + '.txt'), 'w')
        for line in lines: fout.write(line + '\n')
        fout.close()


def main(args):

    # parse entries
    entries = parse_index(args.data_dir)

    # create dataset
    create_mmap(
        process_iterator(args.data_dir, entries, args.interface_dist),
        args.out_dir, len(entries)
    )

    # split
    valid_targets = ['PD-L1']
    test_targets = ['CD38_nocplx', 'trop2']
    split(args.out_dir, valid_targets, test_targets)

    print_log('Finished!')


if __name__ == '__main__':
    main(parse())