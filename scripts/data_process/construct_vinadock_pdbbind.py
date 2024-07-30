#!/usr/bin/python
# -*- coding:utf-8 -*-
import re
import os
import json
import time
import shutil
import argparse

from tqdm import tqdm
import numpy as np

from data.format import VOCAB
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.converter.sdf_to_list_blocks import sdf_to_list_blocks
from utils.logger import print_log

from .process_PDBbind import Entry, parse_index_PL
from .vina.api import VinaDock, Task


def parse():
    parser = argparse.ArgumentParser(description='construct negative samples with vina dock')
    parser.add_argument('--chembl_tsv', type=str, required=True, help='TSV of chembl database')
    parser.add_argument('--pdbbind_dir', type=str, required=True, help='Directory of pdbbind database')
    parser.add_argument('--out_dir', type=str, required=True, help='output directory')
    parser.add_argument('--n_cpus', type=int, default=8)
    return parser.parse_args()


def get_seqs(pdb):
    list_blocks = pdb_to_list_blocks(pdb)
    seqs = []
    for blocks in list_blocks:
        seqs.append(''.join([VOCAB.abrv_to_symbol(block.abrv) for block in blocks]))
    return seqs


def merge_sdf(paths, out_path):
    fout = open(out_path, 'w')
    for path in paths:
        with open(path, 'r') as fin:
            fout.write(fin.read())
    fout.close()


def dock(vina_dock: VinaDock, entries, smis, out_dir, neg_cnt=10):
    os.makedirs(out_dir, exist_ok=True)
    smi_idxs = [i for i in range(len(smis))]
    logfile = open(os.path.join(out_dir, 'done.log'), 'w')
    entry: Entry = None
    waiting_tasks = {}

    def write_results():
        res = vina_dock.safe_get()
        if res is None: return False
        task, ligfiles, energies = res
        entry: Entry = None
        entry, current_out_dir = waiting_tasks.pop(task.id)
        os.system(f'cp {task.receptor_pdb} {current_out_dir}')
        print_log(f'{task.id} finished, {vina_dock.process_cnt()} processes alive')
        if ligfiles is None:
            logfile.write(f'{task.id}\tfailed')
            logfile.flush()
            shutil.rmtree(os.path.join(current_out_dir, 'vina_out'))
            return True

        order = sorted([i for i in range(len(ligfiles))], key=lambda i: energies[0][i])

        merge_sdf([ligfiles[i] for i in order], os.path.join(current_out_dir, 'ligands.sdf'))

        metadata = {
            'id': entry.id,
            'resolution': entry.resolution,
            'year': entry.year,
            'pK': entry.pkd,
            'pK_type': None,
            'receptor_seqs': get_seqs(os.path.join(current_out_dir, 'receptor.pdb')),
            'dock': {}
        }
        for i in order:
            metadata['dock'][task.lig_name_list[i]] = {
                'VinaScore': energies[0][i],
                'smiles': task.ligand_smi_list[i]
            }
        with open(os.path.join(current_out_dir, 'metadata.json'), 'w') as fout:
            json.dump(metadata, fout, indent=2)
        logfile.write(f'{task.id}\tsuccess\n')
        logfile.flush()
        shutil.rmtree(os.path.join(current_out_dir, 'vina_out'))
        return True

    print_log('Adding tasks')

    for entry in tqdm(entries):
        print_log(f'processing {entry}')
        current_out_dir = os.path.join(out_dir, entry.id)
        if os.path.exists(os.path.join(current_out_dir, 'metadata.json')): # already finished
            print_log(f'{entry.id} already finished, skip')
            logfile.write(f'{entry.id}\tsuccess\n')
            logfile.flush()
            continue
        tmp_dir = os.path.join(current_out_dir, 'vina_out')
        os.makedirs(tmp_dir, exist_ok=True)
        prot_path = os.path.join(tmp_dir, 'receptor.pdb')
        os.system(f'cp {entry.pdb_path[0]} {prot_path}')

        parsed_molecule = sdf_to_list_blocks(entry.pdb_path[1], silent=True)
        if len(parsed_molecule) == 0:
            print_log(f'{entry.id} ligand parsing failed, skip')
            continue
        positive_blocks, positive_smi = parsed_molecule[0]
        neg_smiles = [smis[i] for i in np.random.choice(smi_idxs, size=neg_cnt, replace=False)]
        coordinates = []
        for block in positive_blocks:
            for atom in block:
                coordinates.append(atom.get_coord())
        center = np.mean(coordinates, axis=0).tolist()
        vina_dock.put(Task(
            id = entry.id,
            ligand_smi_list = [positive_smi] + neg_smiles,
            lig_name_list = ['pos'] + [f'neg{i}' for i in range(len(neg_smiles))],
            receptor_pdb = prot_path,
            center = center,
            output_dir = tmp_dir,
            n_rigid = 1
        ))
        waiting_tasks[entry.id] = (entry, current_out_dir)
        print_log(f'{entry.id} appended, {len(waiting_tasks)} proceeding, {vina_dock.finish_cnt()} waiting to be written, {vina_dock.process_cnt()} processes alive.')
        vina_dock.repair_process()
        while write_results(): pass

    print_log(f'Submitted {len(waiting_tasks)} tasks')

    while len(waiting_tasks):
        if vina_dock.finish_cnt() == 0:
            time.sleep(10)
            continue
        vina_dock.repair_process()
        write_results()
    
    logfile.close()


def main(args):

    # read chembl
    with open(args.chembl_tsv, 'r') as fin:
        lines = fin.readlines()
    idx = lines[0].replace('"', '').split('\t').index('Smiles')
    chembl_smis = [line.split('\t')[idx].strip('"') for line in lines[1:]]

    # initialize vina dock
    vina_dock = VinaDock(num_workers=args.n_cpus)

    # dock samples
    # refined set
    PL_refined_index_file = os.path.join(args.pdbbind_dir, 'refined-set', 'index', 'INDEX_refined_set.2020')
    PL_refined_entries = parse_index_PL(PL_refined_index_file, os.path.join(args.pdbbind_dir, 'refined-set'))
    dock(vina_dock, PL_refined_entries, chembl_smis, os.path.join(args.out_dir, 'processed_refined_set'))

    # general set
    PL_general_index_file = os.path.join(args.pdbbind_dir, 'v2020-other-PL', 'index', 'INDEX_general_PL.2020')
    PL_general_entries = parse_index_PL(PL_general_index_file, os.path.join(args.pdbbind_dir, 'v2020-other-PL'))
    dock(vina_dock, PL_general_entries, chembl_smis, os.path.join(args.out_dir, 'processed_general_set'))


if __name__ == '__main__':
    np.random.seed(0)
    main(parse())