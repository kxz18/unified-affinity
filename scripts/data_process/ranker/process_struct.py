#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import math
import json
import ast
import argparse
from dataclasses import dataclass
from typing import List
from time import sleep

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
    parser.add_argument('--ref_index', type=str, required=True)
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
    ref_pdb_path: str
    rec_chains: List[str]
    lig_chain: str
    ref_lig_chain: str
    sequence: str


def refid2ligchain(index):
    with open(index, 'r') as fin: lines = fin.read().strip().split('\n')
    id2chain = {}
    for line in lines:
        _id, _, pep_chain, _ = line.split('\t')
        id2chain[_id] = pep_chain
    return id2chain


def parse_index(data_dir, index, visited):

    with open(os.path.join(data_dir, 'log.txt'), 'r') as fin:
        lines = fin.read().strip().split('\n')
    finish_dirs = { line.split('\t')[0]: True for line in lines }

    id2chain = refid2ligchain(index)
    ref_dir = os.path.dirname(index)

    entries = []

    for d in os.listdir(data_dir):
        if d not in finish_dirs: continue
        summary = os.path.join(data_dir, d, 'candidates', 'summary.jsonl')
        if not os.path.exists(summary): continue
        with open(summary, 'r') as fin: lines = fin.readlines()
        for line in lines:
            item = json.loads(line)
            _id = item['id']
            if _id in visited: continue
            visited[_id] = True
            entries.append(Entry(
                id=item['id'],
                pdb_path=os.path.join(data_dir, d, 'candidates', item['id'] + '.pdb'),
                ref_pdb_path=os.path.join(ref_dir, 'pdbs', d + '.pdb'),
                rec_chains=item['rec_chains'],
                lig_chain=item['pep_chain'],
                ref_lig_chain=id2chain[d],
                sequence=item['pep_seq']
            ))

    np.random.shuffle(entries)

    return entries


# @ray.remote(num_cpus=1)
def worker(inputs):
    entry, interface_dist = inputs

    # pdb path and ref pdb path
    try:
        gen_chain2blocks = pdb_to_list_blocks(entry.pdb_path, dict_form=True)
        ref_chain2blocks = pdb_to_list_blocks(entry.ref_pdb_path, dict_form=True)

        lig_blocks = gen_chain2blocks[entry.lig_chain]
        ref_lig_blocks = ref_chain2blocks[entry.ref_lig_chain]
    except Exception as e:
        print_log(f'entry failed: {entry}. Reason: {e}')
        return None

    # rmsd
    gen_all_x, ref_all_x = [], []
    for gen_block, ref_block in zip(lig_blocks, ref_lig_blocks):
        for ref_atom in ref_block:
            if gen_block.has_unit(ref_atom.name):
                ref_all_x.append(ref_atom.get_coord())
                gen_all_x.append(gen_block.get_unit_by_name(ref_atom.name).get_coord())
    rmsd = compute_rmsd(np.array(gen_all_x), np.array(ref_all_x), aligned=True)

    # aa level rmsd
    aa_rmsd = []
    for gen_block, ref_block in zip(lig_blocks, ref_lig_blocks):
        gen_all_x, ref_all_x = [], []
        for ref_atom in ref_block:
            if gen_block.has_unit(ref_atom.name):
                ref_all_x.append(ref_atom.get_coord())
                gen_all_x.append(gen_block.get_unit_by_name(ref_atom.name).get_coord())
        if len(gen_all_x):
            rmsd = compute_rmsd(np.array(gen_all_x), np.array(ref_all_x), aligned=True)
        else: rmsd = None
        aa_rmsd.append(rmsd)

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

    return entry, data, (rmsd, aa_rmsd), n_atoms


def process_iterator(data_dir, ref_index, interface_dist: float, n_cpus: int=8):

    cnt = 0

    visited = {}

    entries = parse_index(data_dir, ref_index, visited)

    while len(entries) > 0:
        print_log(f'submitted {len(entries)} entries')
        for entry in entries:
            res = worker((entry, interface_dist))
            cnt += 1
            if res is None: continue
            entry, data, rmsd, n_atoms = res
            props = {
                'rmsd': rmsd,
                'rec_chains': entry.rec_chains,
                'lig_chain': entry.lig_chain,
                'sequence': entry.sequence
            }
            yield entry.id, data, [n_atoms, props], cnt
        
        sleep(30 * 60)
        entries = parse_index(data_dir, ref_index, visited)
    # cnt = 0
    # for entry in entries:
    #     res = worker((entry, interface_dist))
    #     cnt += 1
    #     if res is None: continue
    #     entry, data, rmsd, n_atoms = res
    #     props = {
    #         'rmsd': rmsd,
    #         'rec_chains': entry.rec_chains,
    #         'lig_chain': entry.lig_chain,
    #         'sequence': entry.sequence
    #     }
    #     yield entry.id, data, [n_atoms, props], cnt
       

    # ray.init(num_cpus=n_cpus)
    # futures = [worker.remote((entry, interface_dist)) for entry in entries]
    
    # cnt = 0
    # while len(futures) > 0:
    #     done_ids, futures = ray.wait(futures, num_returns=1)
    #     for done_id in done_ids:
    #         res = ray.get(done_id)
    #         cnt += 1
    #         if res is None: continue
    #         entry, data, rmsd, n_atoms = res
    #         props = {
    #             'rmsd': rmsd,
    #             'rec_chains': entry.rec_chains,
    #             'lig_chain': entry.lig_chain,
    #             'sequence': entry.sequence
    #         }
    #         yield entry.id, data, [n_atoms, props],  cnt


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

    # # parse entries
    # entries = parse_index(args.data_dir, args.ref_index)

    # create dataset
    create_mmap(
        process_iterator(args.data_dir, args.ref_index, args.interface_dist, n_cpus=args.n_cpus),
        args.out_dir, 4612 * 50, commit_batch=1000
    )

    # # split
    # valid_targets = ['PD-L1']
    # test_targets = ['CD38_nocplx', 'trop2']
    # split(args.out_dir, valid_targets, test_targets)

    print_log('Finished!')


if __name__ == '__main__':
    main(parse())