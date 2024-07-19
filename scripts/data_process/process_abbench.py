#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import math
import argparse
from typing import List, Optional
from dataclasses import dataclass

from data.format import VOCAB
from utils.logger import print_log
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.converter.blocks_interface import blocks_interface
from data.mmap_dataset import create_mmap


def parse():
    parser = argparse.ArgumentParser(description='Process antibody benchmark')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory of affinity benchmark')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--interface_dist', type=float, default=8.0,
                        help='Residues who has atoms with distance below this threshold are considered in the complex interface')
    return parser.parse_args()

@dataclass
class Entry:
    id: str
    pdb_path: str
    heavy_chain: Optional[str]
    light_chain: Optional[str]
    antigen_chains: List[str]
    year: Optional[int]
    resolution: Optional[float]
    kd: float
    pkd: float


def format_chain_name(chain_name):
    if chain_name == '' or chain_name == 'NA':
        return ''
    return chain_name


def parse_index(data_dir):
    with open(os.path.join(data_dir, 'antibody_antigen_cases.txt'), 'r') as fin:
        lines = fin.readlines()
    id2chains = {}
    for line in lines[1:]:
        cplx_id = line.split('\t')[0].strip()
        pdb_id, chains = cplx_id.split('_')
        ab_chains, ag_chains = chains.split(':')
        if len(ab_chains) == 2:
            heavy_chain, light_chain = ab_chains
        else:
            heavy_chain, light_chain = ab_chains[0], ''
        new_id = f'{pdb_id}_{ag_chains}_{heavy_chain}_{light_chain}'
        id2chains[new_id] = (list(ag_chains), heavy_chain, light_chain)
    
    with open(os.path.join(data_dir, 'antibody_antigen_affinities.txt'), 'r') as fin:
        lines = fin.readlines()
    pdb_id2kd = {}
    for line in lines[1:]:
        line = line.split('\t')
        pdb_id2kd[line[0]] = float(line[2]) * 1e-9

    entries = []
    for _id in id2chains:
        pdb_id = _id[:4].upper()
        if pdb_id not in pdb_id2kd:
            print_log(f'{_id} affinity not recorded')
            continue
        kd = pdb_id2kd[pdb_id]
        pkd = -math.log(kd, 10)
        antigen_chains, heavy_chain, light_chain = id2chains[_id]
        resolution = None
        year = None
        ag_file = os.path.join(data_dir, 'pdbs', f'{pdb_id}_l_b.pdb')
        ab_file = os.path.join(data_dir, 'pdbs', f'{pdb_id}_r_b.pdb')
        entries.append(Entry(
            id=_id,
            pdb_path=(ag_file, ab_file),
            heavy_chain=None if heavy_chain == '' else heavy_chain,
            light_chain=None if light_chain == '' else light_chain,
            antigen_chains=antigen_chains,
            year=year,
            resolution=resolution,
            kd=kd,
            pkd=pkd
        ))
    return entries


def process_iterator(entries: List[Entry], interface_dist: float):
    for i, entry in enumerate(entries):
        ag_chain2blocks = pdb_to_list_blocks(entry.pdb_path[0], dict_form=True)
        ab_chain2blocks = pdb_to_list_blocks(entry.pdb_path[1], dict_form=True)
        rec_chains = entry.antigen_chains
        lig_chains = []
        if entry.heavy_chain is not None: lig_chains.append(entry.heavy_chain)
        if entry.light_chain is not None: lig_chains.append(entry.light_chain)

        rec_blocks, lig_blocks = [], []
        for c in rec_chains: rec_blocks.extend(ag_chain2blocks[c])
        for c in lig_chains: lig_blocks.extend(ab_chain2blocks[c])

        (rec_blocks, lig_blocks), _ = blocks_interface(
            rec_blocks, lig_blocks, interface_dist
        )

        data = (
            [block.to_tuple() for block in rec_blocks],
            [block.to_tuple() for block in lig_blocks],
        )
        properties = {
            'id': entry.id,
            'resolution': entry.resolution,
            'year': entry.year,
            'pkd': entry.pkd,
            'rec_chains': rec_chains,
            'lig_chains': lig_chains,
            'rec_seqs': [''.join(VOCAB.abrv_to_symbol(block.abrv) for block in ag_chain2blocks[c]) for c in rec_chains],
            'lig_seqs': [''.join(VOCAB.abrv_to_symbol(block.abrv) for block in ab_chain2blocks[c]) for c in lig_chains],
            'heavy_chain': entry.heavy_chain,
            'light_chain': entry.light_chain
        }
        n_atoms = 0
        for block in rec_blocks: n_atoms += len(block)
        for block in lig_blocks: n_atoms += len(block)

        yield entry.id, data, [n_atoms, properties], i + 1


def main(args):
    entries = parse_index(args.data_dir)
    create_mmap(
        process_iterator(entries, args.interface_dist),
        os.path.join(args.out_dir), len(entries)
    )

    print_log('Finished!')


if __name__ == '__main__':
    main(parse())