#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import math
import json
import ast
import argparse
from dataclasses import dataclass
from typing import List

import ray
import numpy as np

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
    parser.add_argument('--n_cpus', type=int, default=8)
    return parser.parse_args()


# a: [N, 3], b: [N, 3]
def compute_rmsd(a, b, aligned=False):  # amino acids level rmsd
    dist = np.sum((a - b) ** 2, axis=-1)
    rmsd = np.sqrt(dist.sum() / a.shape[0])
    return float(rmsd)


@dataclass
class Entry:
    id: str
    pdb_path: str
    rec_chains: List[str]
    lig_chain: str
    sequence: str


def parse_index(data_dir, visited):

    entries = []

    for d in os.listdir(data_dir):
        sub_dir = os.path.join(data_dir, d)
        if not os.path.isdir(sub_dir): continue
        for n in os.listdir(sub_dir):
            summary = os.path.join(sub_dir, n, 'summary.jsonl')
            if not os.path.exists(summary): continue
            with open(summary, 'r') as fin: lines = fin.readlines()
            for line in lines:
                item = json.loads(line)
                _id = n + '/' + item['id']
                if _id in visited: continue
                visited[_id] = True
                entries.append(Entry(
                    id=_id,
                    pdb_path=os.path.join(sub_dir, n, item['id'] + '.pdb'),
                    rec_chains=item['rec_chains'],
                    lig_chain=item['pep_chain'],
                    sequence=item['pep_seq']
                ))

    np.random.shuffle(entries)

    return entries


# @ray.remote(num_cpus=1)
def worker(inputs):
    entry, interface_dist = inputs

    # pdb path
    try:
        gen_chain2blocks = pdb_to_list_blocks(entry.pdb_path, dict_form=True)
        lig_blocks = gen_chain2blocks[entry.lig_chain]
    except Exception as e:
        print_log(f'entry failed: {entry}. Reason: {e}')
        return None

    rec_blocks = []
    for c in entry.rec_chains: rec_blocks.extend(gen_chain2blocks[c])

    (rec_blocks, _), _ = blocks_interface(
        rec_blocks, lig_blocks, interface_dist
    )

    data = (
        [block.to_tuple() for block in rec_blocks],
        [block.to_tuple() for block in lig_blocks],
    )

    n_atoms = 0
    for block in rec_blocks: n_atoms += len(block)
    for block in lig_blocks: n_atoms += len(block)

    return entry, data, n_atoms


def process_iterator(data_dir: str, interface_dist: float, n_cpus: int=8):

    cnt = 0

    visited = {}

    entries = parse_index(data_dir, visited)

    while len(entries) > 0:
        for entry in entries:
            res = worker((entry, interface_dist))
            cnt += 1
            if res is None: continue
            entry, data, n_atoms = res
            props = {
                'rmsd': None,
                'rec_chains': entry.rec_chains,
                'lig_chain': entry.lig_chain,
                'sequence': entry.sequence
            }
            yield entry.id, data, [n_atoms, props], cnt
        entries = parse_index(data_dir, visited)

    # ray.init(num_cpus=n_cpus)
    
    # cnt = 0
    
    # visited = {}

    # entries = parse_index(data_dir)
    # for entry in entries: visited[entry.id] = True

    # while len(entries) > 0:
    #     futures = [worker.remote((entry, interface_dist)) for entry in entries]
    #     while len(futures) > 0:
    #         done_ids, futures = ray.wait(futures, num_returns=1)
    #         for done_id in done_ids:
    #             res = ray.get(done_id)
    #             cnt += 1
    #             if res is None: continue
    #             entry, data, n_atoms = res
    #             props = {
    #                 'rmsd': None,
    #                 'rec_chains': entry.rec_chains,
    #                 'lig_chain': entry.lig_chain,
    #                 'sequence': entry.sequence
    #             }
    #             yield entry.id, data, [n_atoms, props],  cnt
    #     entries = parse_index(data_dir)
    #     entries = [entry for entry in entries if entry.id not in visited]


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

    # create dataset
    create_mmap(
        process_iterator(args.data_dir, args.interface_dist, n_cpus=args.n_cpus),
        args.out_dir, 4612 * 50 * 20, commit_batch=1000
    )

    # # split
    # valid_targets = ['PD-L1']
    # test_targets = ['CD38_nocplx', 'trop2']
    # split(args.out_dir, valid_targets, test_targets)

    print_log('Finished!')


if __name__ == '__main__':
    main(parse())