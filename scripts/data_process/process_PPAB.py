#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import sys
import math
import argparse
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

PROJ_DIR = os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
)
sys.path.append(PROJ_DIR)

from utils.logger import print_log
from data.format import VOCAB
from data.mmap_dataset import create_mmap
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.converter.blocks_interface import blocks_interface


@dataclass
class Entry:
    id: str
    pdb_path: str
    rec_chains: List[str]
    lig_chains: List[str]
    resolution: Optional[float]
    year: Optional[int]
    kd: float
    pkd: float
    I_rmsd: float


def parse():
    parser = argparse.ArgumentParser(description='Process protein-protein binding affinity data from the structural benchmark')
    parser.add_argument('--index_file', type=str, required=True,
                        help='Path to the index file: PPAB_V2.csv')
    parser.add_argument('--pdb_dir', type=str, required=True,
                        help='Directory of pdbs: benchmark5.5')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--interface_dist', type=float, default=8.0,
                        help='Residues who has atoms with distance below this threshold are considered in the complex interface')
    return parser.parse_args()
    

def process_line(line, pdb_dir, interface_dist_th):
    line = line.split(',')
    assert len(line) == 14
    pdb, (lig_chains, rec_chains) = line[0][:4], line[0][5:].split(':')  # e.g. 1A2K_C:AB
    try:
        Kd, dG = float(line[7]), float(line[8])
    except ValueError as e:
        print_log(f'{pdb} Kd not correct: {e}.', level='ERROR')
        return None

    _id = pdb.lower() + '_' + rec_chains + '_' + lig_chains
    lig_chains, rec_chains = list(lig_chains), list(rec_chains)
    I_rmsd = float(line[9])
    # _id, _class, I_rmsd = pdb.lower() + line[0][4:], line[1], float(line[9])
    # item = {
    #         'id': pdb.lower() + line[0][4:],
    #         'class': line[1],
    #         'I_rmsd': float(line[9])
    # }
    # item['affinity'] = {
    #         'Kd': Kd,
    #         'dG': dG,
    #         'neglog_aff': -math.log(Kd, 10),
    # }
    #assert item['I_rmsd'] > 0
    assert I_rmsd > 0

    lig_path = os.path.join(pdb_dir, 'structures', f'{pdb}_l_b.pdb')
    rec_path = os.path.join(pdb_dir, 'structures', f'{pdb}_r_b.pdb')

    if os.path.exists(lig_path) and os.path.exists(rec_path):
    
        return Entry(
            id=_id,
            pdb_path=(rec_path, lig_path),
            rec_chains=rec_chains,
            lig_chains=lig_chains,
            resolution=None,
            year=None,
            kd=Kd,
            pkd=-math.log(Kd, 10),
            I_rmsd=I_rmsd
        )
    return None

    if os.path.exists(lig_path) and os.path.exists(rec_path):
        lig_prot = Protein.from_pdb(lig_path)
        rec_prot = Protein.from_pdb(rec_path)
        for c in rec_chains:
            all_hit = True
            if c not in rec_prot.peptides:
                all_hit = False
                break
        if not all_hit:
            lig_chains, rec_chains = rec_chains, lig_chains
        for c in rec_chains:
            if c not in rec_prot.peptides:
                print_log(f'Chain {c} not in {pdb} receptor: {rec_prot.get_chain_names()}', level='ERROR')
                return None
        for c in lig_chains:
            if c not in lig_prot.peptides:
                print_log(f'Chain {c} not in {pdb} ligand: {lig_prot.get_chain_names()}', level='ERROR')
                return None
        peptides = lig_prot.peptides
        peptides.update(rec_prot.peptides)
        cplx = Complex(item['id'], peptides, rec_chains, lig_chains)
    else:
        print_log(f'{pdb} not found in the local files, fetching it from remote server.', level='WARN')
        cplx_path = os.path.join(pdb_dir, 'structures', f'{pdb}_cplx.pdb')
        fetch_from_pdb(pdb.upper(), cplx_path)
        cplx = Complex.from_pdb(cplx_path, rec_chains, lig_chains)


    # Protein1 is receptor, protein2 is ligand
    item['seq_protein1'] = ''.join([cplx.get_chain(c).get_seq() for c in rec_chains])
    item['chains_protein1'] = rec_chains
    item['seq_protein2'] = ''.join([cplx.get_chain(c).get_seq() for c in lig_chains])
    item['chains_protein2'] = lig_chains

    # construct pockets
    interface1, interface2 = cplx.get_interacting_residues(dist_th=interface_dist_th)
    if len(interface1) == 0:  # no interface (if len(interface1) == 0 then we must have len(interface2) == 0)
        print_log(f'{pdb} has no interface', level='ERROR')
        return None
    columns = ['chain', 'insertion_code', 'residue', 'resname', 'x', 'y', 'z', 'element', 'name']
    for i, interface in enumerate([interface1, interface2]):
        data = []
        for chain, residue in interface:
            data.extend(residue_to_pd_rows(chain, residue))
        item[f'atoms_interface{i + 1}'] = pd.DataFrame(data, columns=columns)
            
    # construct DataFrame of coordinates
    for i, chains in enumerate([rec_chains, lig_chains]):
        data = []
        for chain in chains:
            chain_obj = cplx.get_chain(chain)
            if chain_obj is None:
                print_log(f'{chain} not in {pdb}: {cplx.get_chain_names()}. Skip this chain.', level='WARN')
                continue
            for residue in chain_obj:
                data.extend(residue_to_pd_rows(chain, residue))                
        item[f'atoms_protein{i + 1}'] = pd.DataFrame(data, columns=columns)

    return item



def process_iterator_PP(entries: List[Entry], interface_dist: float):
    for i, entry in enumerate(entries):
        rec_chain2blocks = pdb_to_list_blocks(entry.pdb_path[0], dict_form=True)
        lig_chain2blocks = pdb_to_list_blocks(entry.pdb_path[1], dict_form=True)

        rec_chains, lig_chains = entry.rec_chains, entry.lig_chains
        if rec_chains[0] not in rec_chain2blocks:
            rec_chains, lig_chains = lig_chains, rec_chains
        failed = False
        for c in rec_chains:
            if c not in rec_chain2blocks:
                failed = True
        for c in lig_chains:
            if c not in lig_chain2blocks:
                failed = True
        if failed:
            print_log(str(entry))
            print_log(f'only have chains: {rec_chain2blocks.keys()}, {lig_chain2blocks.keys()}')
            continue
        rec_blocks, lig_blocks = [], []
        for c in rec_chains: rec_blocks.extend(rec_chain2blocks[c])
        for c in lig_chains: lig_blocks.extend(lig_chain2blocks[c])

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
            'rec_seqs': [''.join(VOCAB.abrv_to_symbol(block.abrv) for block in rec_chain2blocks[c]) for c in rec_chains],
            'lig_seqs': [''.join(VOCAB.abrv_to_symbol(block.abrv) for block in lig_chain2blocks[c]) for c in lig_chains],
        }
        n_atoms = 0
        for block in rec_blocks: n_atoms += len(block)
        for block in lig_blocks: n_atoms += len(block)

        yield entry.id, data, [n_atoms, properties], i + 1

def main(args):

    with open(args.index_file, 'r') as fin:
        lines = fin.readlines()
    lines = lines[1:]  # the first one is head

    print_log('Preprocessing entries')
    entries = []
    cnt = 0
    for i, line in enumerate(lines):
        entry = process_line(line, args.pdb_dir, args.interface_dist)
        cnt += 1
        if entry is None:
            continue
        entries.append(entry)
    
    print_log('Processing data')
    create_mmap(
        process_iterator_PP(entries, args.interface_dist),
        args.out_dir, len(entries)
    )
    
    print_log('Finished')


if __name__ == '__main__':
    main(parse())