#!/usr/bin/python
# -*- coding:utf-8 -*-
import re
import os
import sys
import math
import pickle
import argparse
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

PROJ_DIR = os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
)
print(f'Project directory: {PROJ_DIR}')
sys.path.append(PROJ_DIR)

from data.format import VOCAB, Block
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.converter.sdf_to_list_blocks import sdf_to_list_blocks
from data.converter.blocks_interface import blocks_interface
from data.mmap_dataset import create_mmap
from utils.network import url_get
from utils.logger import print_log
from utils import const


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
    other_info: any = None


def parse():
    parser = argparse.ArgumentParser(description='Process PDBbind')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory of PDBbind')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--interface_dist', type=float, default=8.0,
                        help='Residues who has atoms with distance below this threshold are considered in the complex interface')
    return parser.parse_args()


def _group_strategy1(proteins):
    # group by protein type
    protein_map = {}
    for _, prot_type, chains, chains2, seq in proteins:
        if prot_type not in protein_map:
            protein_map[prot_type] = { 'chains': [], 'chains2': [], 'seq': '' }
        protein_map[prot_type]['chains'].extend(chains)
        protein_map[prot_type]['seq'] += seq
        protein_map[prot_type]['chains2'].extend(chains2)
    proteins = []
    for prot_type in protein_map:
        proteins.append(protein_map[prot_type])
    return proteins

def _group_strategy2(proteins):
    '''
    group by keywords in the protein name
    e.g.
    >1I51_1|Chains A, C|CASPASE-7 SUBUNIT P20|Homo sapiens (9606)
    >1I51_2|Chains B, D|CASPASE-7 SUBUNIT P11|Homo sapiens (9606)
    >1I51_3|Chains E, F|X-LINKED INHIBITOR OF APOPTOSIS PROTEIN|Homo sapiens (9606)
    1 and 2 should be grouped together
    '''
    keywords = ['SUBUNIT', 'RECEPTOR']
    protein_map = {}
    for prot_name, _, chains, chains2, seq in proteins:
        prot_name = prot_name.upper()
        for keyword in keywords:
            if keyword in prot_name:
                prot_name = keyword
                break
        if prot_name not in protein_map:
            protein_map[prot_name] = { 'chains': [], 'chains2': [], 'seq': '' }
        protein_map[prot_name]['chains'].extend(chains)
        protein_map[prot_name]['seq'] += seq
        protein_map[prot_name]['chains2'].extend(chains2)
    proteins = []
    for prot_name in protein_map:
        proteins.append(protein_map[prot_name])
    return proteins

def _group_strategy3(proteins):
    '''
    group by antibody (the rest is antigen)
    e.g.
    >1YYM_1|Chains A[auth G], E[auth P]|Exterior membrane glycoprotein(GP120),Exterior membrane glycoprotein(GP120),Exterior membrane glycoprotein(GP120)|Human immunodeficiency virus 1 (11676)
    >1YYM_2|Chains B[auth L], F[auth Q]|antibody 17b light chain|Homo sapiens (9606)
    >1YYM_3|Chains C[auth H], G[auth R]|antibody 17b heavy chain|Homo sapiens (9606)
    >1YYM_4|Chains D[auth M], H[auth S]|F23, scorpion-toxin mimic of CD4|synthetic construct (32630)
    2 and 3 should be grouped, while 1 and 4 compose the antigen
    '''
    protein_map = {}
    antibody_detected = False
    for prot_name, _, _, _, _ in proteins:
        if 'ANTIBODY' in prot_name.upper():
            antibody_detected = True
            break
    if not antibody_detected:
        return proteins

    for prot_name, _, chains, chains2, seq in proteins:
        prot_name = prot_name.upper()
        prot_name = 'ANTIBODY' if 'ANTIBODY' in prot_name else 'ANTIGEN'
        if prot_name not in protein_map:
            protein_map[prot_name] = { 'chains': [], 'chains2': [], 'seq': '' }
        protein_map[prot_name]['chains'].extend(chains)
        protein_map[prot_name]['seq'] += seq
        protein_map[prot_name]['chains2'].extend(chains2)
    proteins = []
    for prot_name in protein_map:
        proteins.append(protein_map[prot_name])

    return proteins

def _group_strategy4(proteins):
    '''
    group by modified name (get rid of pairing difference like alpha/beta, heavy/light
    e.g.
    >4CNI_1|Chains A, E[auth H]|OLOKIZUMAB HEAVY CHAIN, FAB PORTION|HOMO SAPIENS (9606)
    >4CNI_2|Chains B, F[auth L]|OLOKIZUMAB LIGHT CHAIN, FAB PORTION|HOMO SAPIENS (9606)
    >4CNI_3|Chains C, D|INTERLEUKIN-6|HOMO SAPIENS (9606)
    1 and 2 should be grouped
    '''
    protein_map = {}
    pair_keywords = ['HEAVY', 'LIGHT', 'ALPHA', 'BETA', 'VH', 'VL']
    for prot_name, _, chains, chains2, seq in proteins:
        prot_name = prot_name.upper()
        for keyword in pair_keywords:
            prot_name = prot_name.replace(keyword, '')
        if prot_name not in protein_map:
            protein_map[prot_name] = { 'chains': [], 'chains2': [], 'seq': '' }
        protein_map[prot_name]['chains'].extend(chains)
        protein_map[prot_name]['seq'] += seq
        protein_map[prot_name]['chains2'].extend(chains2)
    proteins = []
    for prot_name in protein_map:
        proteins.append(protein_map[prot_name])

    return proteins

def _group_strategy5(proteins):
    '''
    special strategy for TCR-related cases
    e.g.
    >6MSS_1|Chain A|A11B8.2 NKT TCR alpha-chain|Mus musculus (10090)
    >6MSS_2|Chain B|A11B8.2 NKT TCR beta-chain|Mus musculus (10090)
    >6MSS_3|Chain C|Antigen-presenting glycoprotein CD1d1|Mus musculus (10090)
    >6MSS_4|Chain D|Beta-2-microglobulin|Mus musculus (10090)
    1 and 2 are TCR components, 3 and 4 are MHC components (3 is the presenting peptide, 4 is a infrastructure of MHC molecule)
    '''
    mhc_keywords = ['MHC', 'HLA', 'ANTIGEN-PRESENTING', 'BETA-2-MICROGLOBULIN', 'GLYCOPROTEIN']
    protein_map = {}
    for prot_name, _, chains, chains2, seq in proteins:
        prot_name = prot_name.upper()
        if ('TCR' in prot_name) or ('RECEPTOR' in prot_name and (('T-CELL' in prot_name) or ('T CELL' in prot_name))):
            prot_name = 'TCR'
        else:
            for keyword in mhc_keywords:
                if keyword in prot_name:
                    prot_name = 'MHC'
                    break
        if prot_name not in protein_map:
            protein_map[prot_name] = { 'chains': [], 'chains2': [], 'seq': '' }
        protein_map[prot_name]['chains'].extend(chains)
        protein_map[prot_name]['seq'] += seq
        protein_map[prot_name]['chains2'].extend(chains2)
    proteins = []
    for prot_name in protein_map:
        proteins.append(protein_map[prot_name])

    return proteins


def parse_fasta(lines):
    assert len(lines) % 2 == 0, 'Number of fasta lines is not an even number!'
    proteins = []
    for i in range(0, len(lines), 2):
        details = lines[i].split('|')
        assert len(details) == 4
        prot_name = details[2]
        prot_type = details[3]
        chain_strs = details[1]
        if chain_strs.startswith('Chains '):
            chain_strs = chain_strs.replace('Chains ', '')
        elif chain_strs.startswith('Chain'):
            chain_strs = chain_strs.replace('Chain ', '')
        else:
            raise ValueError(f'Chain details has wrong format: {chain_strs}')
        chain_strs = chain_strs.split(', ')  # e.g. Chains B[auth H], F[auth K], or Chain A
        chains = [s[0] for s in chain_strs]
        chains2 = [s[-2] if len(s) > 1 else s[0] for s in chain_strs]  # multiple models
        seq = lines[i + 1]
        proteins.append((prot_name, prot_type, chains, chains2, seq))
    if len(proteins) > 2:  # receptor or ligand has been splitted into different sets of chains
        for strategy in [_group_strategy1, _group_strategy2, _group_strategy3, _group_strategy4, _group_strategy5]:
            grouped_proteins = strategy(proteins)
            if len(grouped_proteins) == 2:
                proteins = grouped_proteins
                break
    else:
        proteins = [{ 'chains': chains, 'chains2': chains2, 'seq': seq } \
                    for _, _, chains, chains2, seq in proteins]
    return proteins


def parse_kd(kd):
    if (not kd.startswith('Kd')) and (not kd.startswith('Ki')):  # IC50 is very different from Kd and Ki, therefore discarded
        return None
    if '=' not in kd: # e.g. Kd~0.5
        return None
    kd = kd.split('=')[-1].strip()
    aff, unit = float(kd[:-2]), kd[-2:]
    if unit == 'mM':
        aff *= 1e-3
    elif unit == 'nM':
        aff *= 1e-9
    elif unit == 'uM':
        aff *= 1e-6
    elif unit == 'pM':
        aff *= 1e-12
    elif unit == 'fM':
        aff *= 1e-15
    else:
        return None   # unrecognizable unit
    return aff


def process_line_PP(line, pdb_dir):

    if line.startswith('#'):  # annotation
        return None

    line_split = re.split(r'\s+', line)
    pdb, kd = line_split[0], line_split[3]
    try:
        resolution = float(line_split[1])
    except ValueError:
        resolution = None
    year = int(line_split[2])

    if (not kd.startswith('Kd')) and (not kd.startswith('Ki')):  # IC50 is very different from Kd and Ki, therefore discarded
        print_log(f'{pdb} not measured by Kd or Ki, dropped.', level='ERROR')
        return None
    
    if '=' not in kd:  # some data only provide a threshold, e.g. Kd<1nM, discarded
        print_log(f'{pdb} Kd only has threshold: {kd}', level='ERROR')
        return None

    aff = parse_kd(kd)
    if aff is None: return None

    # affinity data
    pkd = -math.log(aff, 10) # pK = -log_10 (Kd)
    
    fasta = url_get(f'http://www.pdbbind.org.cn/FASTA/{pdb}.txt')
    if fasta is None:
        print_log(f'Failed to fetch fasta for {pdb}!', level='ERROR')
        return None
    fasta = fasta.text.strip().split('\n')
    proteins = parse_fasta(fasta)
    if len(proteins) != 2:  # irregular fasta, cannot distinguish which chains composes one protein
        print_log(f'{pdb} has {len(proteins)} chain sets!', level='ERROR')
        return None
    
    pdb_file = os.path.join(pdb_dir, pdb + '.ent.pdb')
    return Entry(
        id=pdb,
        pdb_path=pdb_file,
        rec_chains=[],
        lig_chains=[],
        resolution=resolution,
        year=year,
        kd=kd,
        pkd=pkd,
        other_info=proteins
    )

def parse_index_PP(fpath, pdb_dir):
    with open(fpath, 'r') as fin:
        lines = fin.readlines()
    entries = []
    for i, line in enumerate(lines):
        item = process_line_PP(line, pdb_dir)
        if item is None:
            continue
        entries.append(item)
    return entries


def parse_index_PL(fpath, pdb_dir):
    with open(fpath, 'r') as fin:
        lines = fin.readlines()
    
    entries = []
    for line in lines:
        if line.startswith('#'):
            continue
        line = re.split(r'\s+', line)
        pdb_id, resolution, year, kd = line[:4]
        try: resolution = float(resolution)
        except ValueError: resolution = None
        try: year = int(year)
        except ValueError: year = None
        kd = parse_kd(kd)
        if kd is None: continue
        prot_path = os.path.join(pdb_dir, pdb_id, pdb_id + '_protein.pdb')
        lig_path = os.path.join(pdb_dir, pdb_id, pdb_id + '_ligand.sdf')
        if not os.path.exists(prot_path) or not os.path.exists(lig_path):
            continue
        entries.append(Entry(
            id=pdb_id,
            pdb_path=(prot_path, lig_path),
            rec_chains=[],
            lig_chains=[],
            resolution=resolution,
            year=year,
            kd=kd,
            pkd=-math.log(kd, 10)
        ))
    return entries


def parse_index_PN(fpath, pdb_dir):
    with open(fpath, 'r') as fin:
        lines = fin.readlines()
    
    entries = []
    for line in lines:
        if line.startswith('#'):
            continue
        line = re.split(r'\s+', line)
        pdb_id, resolution, year, kd = line[:4]
        try: resolution = float(resolution)
        except ValueError: resolution = None
        try: year = int(year)
        except ValueError: year = None
        kd = parse_kd(kd)
        if kd is None: continue
        pdb_path = os.path.join(pdb_dir, pdb_id + '.ent.pdb')
        if not os.path.exists(pdb_path):
            continue
        entries.append(Entry(
            id=pdb_id,
            pdb_path=pdb_path,
            rec_chains=[],
            lig_chains=[],
            resolution=resolution,
            year=year,
            kd=kd,
            pkd=-math.log(kd, 10)
        ))
    return entries

def process_iterator_PP(entries: List[Entry], interface_dist: float):
    for i, entry in enumerate(entries):
        chain2blocks = pdb_to_list_blocks(entry.pdb_path, dict_form=True)

        # identify receptor and ligand
        proteins = entry.other_info
        chain_set_id = None
        for chain_set_name in ['chains', 'chains2']:
            all_in = True
            for c in proteins[0][chain_set_name] + proteins[1][chain_set_name]:
                if c not in chain2blocks:
                    all_in = False
                    break
            if all_in:
                chain_set_id = chain_set_name
                break
        if chain_set_id is None:
            print_log(f'Chains {proteins[0]["chains"] + proteins[1]["chains"]} have at least one missing in {entry.id}: {list(chain2blocks.keys())}', level='ERROR')
            continue

        rec_chains, lig_chains = proteins[0][chain_set_id], proteins[1][chain_set_id]
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
        }
        n_atoms = 0
        for block in rec_blocks: n_atoms += len(block)
        for block in lig_blocks: n_atoms += len(block)

        yield entry.id, data, [n_atoms, properties], i + 1

def process_iterator_PL(entries: List[Entry], interface_dist: float):
    for i, entry in enumerate(entries):
        chain2blocks = pdb_to_list_blocks(entry.pdb_path[0], dict_form=True)
        list_blocks = sdf_to_list_blocks(entry.pdb_path[1], silent=True)
        if len(list_blocks) == 0:
            continue
        lig_blocks, smiles = list_blocks[0]
        rec_chains = list(chain2blocks.keys())
        rec_blocks = []
        for c in rec_chains: rec_blocks.extend(chain2blocks[c])
        (rec_blocks, _), _ = blocks_interface(
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
            'lig_chains': [],
            'rec_seqs': [''.join(VOCAB.abrv_to_symbol(block.abrv) for block in chain2blocks[c]) for c in rec_chains],
            'lig_seqs': [smiles],
        }
        n_atoms = 0
        for block in rec_blocks: n_atoms += len(block)
        for block in lig_blocks: n_atoms += len(block)

        yield entry.id, data, [n_atoms, properties], i + 1


def process_iterator_PN(entries: List[Entry], interface_dist: float):
    for i, entry in enumerate(entries):
        chain2blocks = pdb_to_list_blocks(entry.pdb_path, dict_form=True)
        rec_chains, lig_chains = [], []
        rec_blocks, lig_blocks = [], []
        rec_seqs, lig_seqs = [], []
        for c in chain2blocks:
            blocks = chain2blocks[c]
            seq = ''.join([VOCAB.abrv_to_symbol(block.abrv) for block in blocks])
            if len(blocks) * 2 == len(seq): # RNA/DNA, symbol has two chars (DA, RA, ..)
                lig_chains.append(c)
                lig_blocks.extend(blocks)
                lig_seqs.append(seq)
            else: # protein
                rec_chains.append(c)
                rec_blocks.extend(blocks)
                rec_seqs.append(seq)

        (rec_blocks, lig_blocks), _ = blocks_interface(
            rec_blocks, lig_blocks, interface_dist
        )
        if len(rec_blocks) == 0 or len(lig_blocks) == 0:
            continue

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
            'rec_seqs': rec_seqs,
            'lig_seqs': lig_seqs,
        }
        n_atoms = 0
        for block in rec_blocks: n_atoms += len(block)
        for block in lig_blocks: n_atoms += len(block)

        yield entry.id, data, [n_atoms, properties], i + 1


def process_iterator_NL(entries: List[Entry], interface_dist: float):
    for i, entry in enumerate(entries):
        #print(entry)
        chain2blocks = pdb_to_list_blocks(entry.pdb_path, dict_form=True, allow_het=True)
        rec_chains, lig_chains = [], []
        rec_blocks, lig_blocks = [], []
        rec_seqs, lig_seqs = [], []
        bases = [tup[0] for tup in const.bases]
        abr_bases = ['A', 'G', 'C', 'U']
        for c in chain2blocks:
            blocks = chain2blocks[c]
            nucleic_blocks, sm_blocks = [], []
            for j, block in enumerate(blocks):
                if block.abrv in bases:
                    nucleic_blocks.append(block)
                elif block.abrv in abr_bases:
                    block.abrv = 'R' + block.abrv
                    nucleic_blocks.append(block)
                else:
                    sm_blocks.append(block)
            rec_chains.append(c)
            rec_blocks.extend(nucleic_blocks)
            rec_seqs.append(''.join([VOCAB.abrv_to_symbol(block.abrv) for block in nucleic_blocks]))
            
            if len(sm_blocks):
                lig_chains.append(c)
                for block in sm_blocks: # each is a small molecule
                    for unit in block.units:
                        lig_blocks.append(Block(
                            abrv=VOCAB.symbol_to_abrv(unit.element),
                            units=[unit]
                        ))
                    lig_seqs.append(block.abrv)

        (rec_blocks, _), _ = blocks_interface(
            rec_blocks, lig_blocks, interface_dist
        )
        if len(rec_blocks) == 0 or len(lig_blocks) == 0:
            continue

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
            'rec_seqs': rec_seqs,
            'lig_seqs': lig_seqs,
        }
        n_atoms = 0
        for block in rec_blocks: n_atoms += len(block)
        for block in lig_blocks: n_atoms += len(block)

        yield entry.id, data, [n_atoms, properties], i + 1


def main(args):

    if not os.path.exists(os.path.join(args.out_dir, 'PP')):
        print_log('Processing PP')
        PP_index_file = os.path.join(args.data_dir, 'PP', 'index', 'INDEX_general_PP.2020')
        PP_entries = parse_index_PP(PP_index_file, os.path.join(args.data_dir, 'PP'))
        create_mmap(
            process_iterator_PP(PP_entries, args.interface_dist),
            os.path.join(args.out_dir, 'PP'), len(PP_entries)
        )
    else: print_log('PP exists, continue')
    
    pn_dir = os.path.join(args.out_dir, 'PN')
    if not os.path.exists(pn_dir):
        print_log('Processing PN')
        PN_index_file = os.path.join(args.data_dir, 'PN', 'index', 'INDEX_general_PN.2020')
        PN_entries = parse_index_PN(PN_index_file, os.path.join(args.data_dir, 'PN'))
        create_mmap(
            process_iterator_PN(PN_entries, args.interface_dist),
            pn_dir, len(PN_entries)
        )
    else: print_log('PN exists, continue')

    nl_dir = os.path.join(args.out_dir, 'NL')
    if not os.path.exists(nl_dir):
        print_log('Processing NL')
        NL_index_file = os.path.join(args.data_dir, 'NL', 'index', 'INDEX_general_NL.2020')
        NL_entries = parse_index_PN(NL_index_file, os.path.join(args.data_dir, 'NL'))
        create_mmap(
            process_iterator_NL(NL_entries, args.interface_dist),
            nl_dir, len(NL_entries)
        )
    else: print_log('NL exists, continue')
    
    pl_refined_dir = os.path.join(args.out_dir, 'PL-refined')
    if not os.path.exists(pl_refined_dir):
        print_log('Processing PL-refined')
        PL_refined_index_file = os.path.join(args.data_dir, 'refined-set', 'index', 'INDEX_refined_set.2020')
        PL_refined_entries = parse_index_PL(PL_refined_index_file, os.path.join(args.data_dir, 'refined-set'))
        create_mmap(
            process_iterator_PL(PL_refined_entries, args.interface_dist),
            pl_refined_dir, len(PL_refined_entries)
        )
    else: print_log('PL-refined exists, continue')
    
    pl_general_dir = os.path.join(args.out_dir, 'PL-general')
    if not os.path.exists(pl_general_dir):
        print_log('Processing PL-general')
        PL_general_index_file = os.path.join(args.data_dir, 'v2020-other-PL', 'index', 'INDEX_general_PL.2020')
        PL_general_entries = parse_index_PL(PL_general_index_file, os.path.join(args.data_dir, 'v2020-other-PL'))
        create_mmap(
            process_iterator_PL(PL_general_entries, args.interface_dist),
            pl_general_dir, len(PL_general_entries)
        )
    else: print_log('PL-general exists, continue')

    print_log('Finished!')


if __name__ == '__main__':
    main(parse())