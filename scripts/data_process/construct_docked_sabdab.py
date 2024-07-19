#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import math
import json
import argparse
from copy import deepcopy
from typing import List, Optional
from dataclasses import dataclass
from multiprocessing import Process, Queue

import numpy as np
from p_tqdm import p_map

from data.format import VOCAB
from utils.logger import print_log
from data.funcs.esmfold import parallel_structure_prediction
from data.funcs.rmsd import compute_rmsd
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
    pdb_id: str
    pdb_path: str
    heavy_chain: Optional[str]
    light_chain: Optional[str]
    antigen_chains: List[str]
    antibody_chains: List[str]
    year: Optional[int]
    resolution: Optional[float]
    kd: float
    pkd: float
    antigen_seqs: List[str] = None
    antibody_seqs: List[str] = None # heavy chain goes first
    plddt: Optional[float] = -1
    rmsd: Optional[float] = -1
    negatives: dict = None
    save_path: str = None

    def set_ag_seqs(self, seqs):
        self.antigen_seqs = deepcopy(seqs)

    def set_ab_seqs(self, seqs):
        self.antibody_seqs = deepcopy(seqs)

    def add_negative(self, name, seqs, plddt):
        if self.negatives is None:
            self.negatives = {}
        self.negatives[name] = {
            'plddt': plddt,
            'seqs': deepcopy(seqs)
        }

    def save(self):
        data = {
            'id': self.id,
            'resolution': self.resolution,
            'year': self.year,
            'pKd': self.pkd,
            'antigen_chains': self.antigen_chains,
            'heavy_chain': self.heavy_chain,
            'light_chain': self.light_chain,
            'antigen_seqs': self.antigen_seqs,
            'rmsd': self.rmsd,
            'dock': {}
        }
        docked = [('pos', {'plddt': self.plddt, 'antibody_seqs': self.antibody_seqs})]
        if self.negatives is not None:
            for neg_name in self.negatives:
                docked.append((neg_name, self.negatives[neg_name]))
        docked = sorted(docked, key=lambda tup: tup[1]['plddt'], reverse=True)
        for name, info in docked:
            data['dock'][name] = info
        with open(self.save_path, 'w') as fout:
            json.dump(data, fout, indent=2)


def format_chain_name(chain_name):
    if chain_name == '' or chain_name == 'NA':
        return ''
    return chain_name


def parse_index(summary, struct_dir, out_dir):
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
        pdb_id = line[name2idx['pdb']]

        kd = line[name2idx['affinity']]
        try:
            kd = float(kd)
            pkd = -math.log(kd, 10)
        except ValueError:
            kd, pkd = None, None
    
        antigen_chains = line[name2idx['antigen_chain']].replace(' ', '')
        if antigen_chains == 'NA' or antigen_chains == '':
            continue
        antigen_chains = antigen_chains.split('|')
        antigen_type = line[name2idx['antigen_type']]
        if antigen_type != 'protein':
            continue
        heavy_chain = format_chain_name(line[name2idx['Hchain']])
        light_chain = format_chain_name(line[name2idx['Lchain']])
        if heavy_chain in antigen_chains or light_chain in antigen_chains:
            continue
        if light_chain.lower() == heavy_chain.lower():
            print_log(f'{pdb_id}: heavy chain {heavy_chain}, light chain {light_chain}', level='WARN')
            heavy_chain, light_chain = heavy_chain.upper(), ''
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
        antibody_chains = []
        if heavy_chain != '': antibody_chains.append(heavy_chain)
        if light_chain != '': antibody_chains.append(light_chain)

        _id = f'{pdb_id}_{"".join(antigen_chains)}_{heavy_chain}_{light_chain}'
        entries.append(Entry(
            id=_id,
            pdb_id=pdb_id,
            pdb_path=os.path.join(struct_dir, pdb_id + '.pdb'),
            heavy_chain=None if heavy_chain == '' else heavy_chain,
            light_chain=None if light_chain == '' else light_chain,
            antigen_chains=antigen_chains,
            antibody_chains=antibody_chains,
            year=year,
            resolution=resolution,
            kd=kd,
            pkd=pkd,
            save_path=os.path.join(out_dir, _id, 'metadata.json')
        ))
    return entries


def get_plddt(path):
    gen_chain2blocks = pdb_to_list_blocks(path, dict_form=True)
    plddt = []
    for c in gen_chain2blocks:
        blocks = gen_chain2blocks[c]
        for block in blocks:
            for unit in block.units:
                plddt.append(unit.get_property('bfactor'))
    plddt = sum(plddt) / len(plddt)
    return plddt


def judge_positive_worker(task_queue, result_queue):
    while True:
        task = task_queue.get()
        if task is None:
            break
        gt_pdb, gen_pdb, chains = task
        gt_chain2blocks = pdb_to_list_blocks(gt_pdb, dict_form=True)
        gen_chain2blocks = pdb_to_list_blocks(gen_pdb, dict_form=True)
        # align RMSD
        gt_ca_x, gen_ca_x = [], []
        for i, ref_c in enumerate(chains):
            gen_c = chr(ord('A') + i)
            if gen_c not in gen_chain2blocks:
                continue
            ref_blocks, gen_blocks = gt_chain2blocks[ref_c], gen_chain2blocks[gen_c]
            for j, block in enumerate(ref_blocks):
                if not block.has_unit('CA') or j >= len(gen_blocks):
                    continue
                gt_ca_x.append(block.get_unit_by_name('CA').get_coord())
                gen_ca_x.append(gen_blocks[j].get_unit_by_name('CA').get_coord())
        gt_ca_x, gen_ca_x = np.array(gt_ca_x), np.array(gen_ca_x)
        rmsd = compute_rmsd(gen_ca_x, gt_ca_x)
        # get plddt
        plddt = get_plddt(gen_pdb)
        result_queue.put((gen_pdb, rmsd, plddt))


def set_seqs(entry: Entry):
    try:
        chain2blocks = pdb_to_list_blocks(entry.pdb_path, dict_form=True)
    except Exception:
        return None
    ag_seqs, ab_seqs = [], []
    for c in entry.antigen_chains:
        blocks = chain2blocks[c]
        ag_seqs.append(''.join([VOCAB.abrv_to_symbol(block.abrv) for block in blocks]))
    for c in entry.antibody_chains:
        blocks = chain2blocks[c]
        ab_seqs.append(''.join([VOCAB.abrv_to_symbol(block.abrv) for block in blocks]))
    entry.set_ag_seqs(ag_seqs)
    entry.set_ab_seqs(ab_seqs)
    return entry


def main(args):
    entries = parse_index(args.summary, args.struct_dir, args.out_dir)
    entries = p_map(set_seqs, entries, num_cpus=32)
    entries = [entry for entry in entries if entry is not None]
    print_log(f'Number of entries: {len(entries)}')

    # get sequences
    pos_entries = [entry for entry in entries if entry.pkd is not None]
    print_log(f'Number of entries after pkd filter: {len(entries)}')

    # judger process
    task_queue = Queue()
    result_queue = Queue()
    p = Process(target=judge_positive_worker, args=(task_queue, result_queue))
    p.start()

    # construct positive complexes
    path_to_entry = {}
    seqs, out_paths = [], []
    entry: Entry = None
    for entry in pos_entries:
        seq = ':'.join(entry.antigen_seqs + entry.antibody_seqs)
        out_path = os.path.join(args.out_dir, entry.id, 'pos.pdb')
        seqs.append(seq)
        out_paths.append(out_path)
        os.makedirs(os.path.join(args.out_dir, entry.id), exist_ok=True)
        path_to_entry[out_path] = entry

    # structure prediction
    threshold = 20.0
    success_cnt, total_cnt, task_cnt = 0, 0, 0
    for path in parallel_structure_prediction(seqs, out_paths, [0, 1, 2, 3, 4, 5, 6, 7], silent=True):
        if path is None:
            continue
        entry = path_to_entry[path]
        task_queue.put((entry.pdb_path, path, entry.antigen_chains + entry.antibody_chains))
        task_cnt += 1
        if result_queue.qsize() > 0:
            finish_path, rmsd, plddt = result_queue.get()
            total_cnt += 1
            path_to_entry[finish_path].rmsd = rmsd
            path_to_entry[finish_path].plddt = plddt 
            path_to_entry[finish_path].save()
            if rmsd < threshold:
                success_cnt += 1
                print_log(f'{finish_path} success, rmsd {rmsd}, plddt {plddt}, {success_cnt}/{total_cnt}')
            else:
                print_log(f'{finish_path} failed, rmsd {rmsd}, plddt {plddt}, {success_cnt}/{total_cnt}')
            task_cnt -= 1
    
    while task_cnt > 0:
        finish_path, rmsd, plddt = result_queue.get()
        total_cnt += 1
        path_to_entry[finish_path].rmsd = rmsd
        path_to_entry[finish_path].plddt = plddt 
        path_to_entry[finish_path].save()
        if rmsd < threshold:
            success_cnt += 1
            print_log(f'{finish_path} success, rmsd {rmsd}, plddt {plddt}, {success_cnt}/{total_cnt}')
        else:
            print_log(f'{finish_path} failed, rmsd {rmsd}, plddt {plddt}, {success_cnt}/{total_cnt}')
        task_cnt -= 1
    
    # construct negative complexes
    print_log('Constructing negative samples')
    seqs, out_paths = [], []
    n_neg = 10
    path2entry, path2seqs = {}, {}
    for entry in pos_entries:
        if entry.plddt < 0:
            continue # failed
        # sample 10 negatives
        valid_idx = [j for j in range(len(entries)) if entries[j].pdb_id != entry.pdb_id]
        neg_idx = np.random.choice(valid_idx, size=n_neg, replace=False)
        for n, i in enumerate(neg_idx):
            seq = ':'.join(entry.antigen_seqs + entries[i].antibody_seqs)
            out_path = os.path.join(args.out_dir, entry.id, f'neg{n}.pdb')
            seqs.append(seq)
            out_paths.append(out_path)
            path2entry[out_path] = entry
            path2seqs[out_path] = entries[i].antibody_seqs
    
    cnt = 0
    for path in parallel_structure_prediction(seqs, out_paths, [0, 1, 2, 3, 4, 5, 6, 7], silent=True):
        cnt += 1
        if path is None:
            continue
        entry = path2entry[path]
        entry.add_negative(
            os.path.basename(path).strip('.pdb'),
            path2seqs[path],
            get_plddt(path)
        )
        entry.save()
        print_log(f'{path} finished, {cnt}/{len(seqs)}')

    print_log('All finished')
            
    task_queue.put(None)
    p.join()


if __name__ == '__main__':
    np.random.seed(0)
    main(parse())