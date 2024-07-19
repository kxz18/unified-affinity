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
    parser = argparse.ArgumentParser(description='Process PDBBind')
    parser.add_argument('--summary', type=str, required=True,
                        help='Path to the summary of SAbDab data')
    parser.add_argument('--struct_dir', type=str, required=True,
                        help='Directory of SAbDab structures (e.g. IMGT)')
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


def parse_index(summary, struct_dir):
    with open(summary, 'r') as fin:
        lines = fin.readlines()
    heads = lines[0].strip('\n').split('\t')
    names = ['pdb', 'Hchain', 'Lchain', 'antigen_chain',
             'antigen_type', 'date', 'resolution', 'affinity'
    ]
    name2idx = { n: heads.index(n) for n in names }
    entries = []
    for line in lines[1:]:
        line = line.strip('\n').split('\t')
        kd = line[name2idx['affinity']]
        try:
            kd = float(kd)
        except ValueError:
            continue
        pkd = -math.log(kd, 10)
        pdb_id = line[name2idx['pdb']]
        antigen_chains = line[name2idx['antigen_chain']].replace(' ', '')
        if antigen_chains == 'NA' or antigen_chains == '':
            continue
        antigen_chains = antigen_chains.split('|')
        antigen_type = line[name2idx['antigen_type']]
        if 'protein' not in antigen_type and 'peptide' not in antigen_type and 'nucleic-acid' not in antigen_type:
            continue
        heavy_chain = format_chain_name(line[name2idx['Hchain']])
        light_chain = format_chain_name(line[name2idx['Lchain']])
        if heavy_chain in antigen_chains or light_chain in antigen_chains:
            continue
        if light_chain.lower() == heavy_chain.lower():
            print_log(f'{pdb_id}: heavy chain {heavy_chain}, light chain {light_chain}', level='WARN')
            heavy_chain, light_chain = heavy_chain.upper(), None
        resolution = line[name2idx['resolution']]
        try:
            resolution = float(resolution)
        except ValueError:
            resolution = None
        year = int(line[name2idx['date']].split('/')[-1])
        if year > 24:
            year = 1900 + year
        else:
            year = 2000 + year
        entries.append(Entry(
            id=pdb_id,
            pdb_path=os.path.join(struct_dir, pdb_id + '.pdb'),
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
        chain2blocks = pdb_to_list_blocks(entry.pdb_path, dict_form=True)
        rec_chains = entry.antigen_chains
        lig_chains = []
        if entry.heavy_chain is not None: lig_chains.append(entry.heavy_chain)
        if entry.light_chain is not None: lig_chains.append(entry.light_chain)

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
        properties = {
            'id': entry.id,
            'resolution': entry.resolution,
            'year': entry.year,
            'pkd': entry.pkd,
            'rec_chains': rec_chains,
            'lig_chains': lig_chains,
            'rec_seqs': [''.join(VOCAB.abrv_to_symbol(block.abrv) for block in chain2blocks[c]) for c in rec_chains],
            'lig_seqs': [''.join(VOCAB.abrv_to_symbol(block.abrv) for block in chain2blocks[c]) for c in lig_chains],
            'heavy_chain': entry.heavy_chain,
            'light_chain': entry.light_chain
        }
        n_atoms = 0
        for block in rec_blocks: n_atoms += len(block)
        for block in lig_blocks: n_atoms += len(block)

        yield entry.id, data, [n_atoms, properties], i + 1


def main(args):
    entries = parse_index(args.summary, args.struct_dir)
    create_mmap(
        process_iterator(entries, args.interface_dist),
        os.path.join(args.out_dir), len(entries)
    )

    print_log('Finished!')


if __name__ == '__main__':
    main(parse())