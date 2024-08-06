#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import argparse
from typing import List

import torch

from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.converter.blocks_interface import blocks_interface
from data.converter.blocks_to_data import blocks_to_data_simple
from utils.logger import print_log


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

CHECKPOINTS = [
    os.path.join(ROOT_DIR, 'checkpoints', 'model0.ckpt'),
    os.path.join(ROOT_DIR, 'checkpoints', 'model1.ckpt'),
    os.path.join(ROOT_DIR, 'checkpoints', 'model2.ckpt'),
]


def parse():
    parser = argparse.ArgumentParser(description='Inferencing pKd of arbitrary complexes')
    parser.add_argument('--ckpts', type=str, nargs='+', default=CHECKPOINTS, help='Specify checkpoints to use')
    parser.add_argument('--complex_pdb', type=str, help='PDB of complexes')
    parser.add_argument('--split_chains', type=str, help='e.g. AB_CD, small molecules should be split into a separate chain and within one single residue')
    parser.add_argument('--data_list', type=str, default=None, help='If this is specified, complex_pdb and split_chains is not needed. Content example: \naaaa.pdb\tA_D\nbbbb.pdb\tAD_EJF\n')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
    return parser.parse_args()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, pdbs: List[str], receptor_chains: List[str], ligand_chains: List[str], if_th: float = 8.0):
        self.pdbs = pdbs
        # receptor and ligand can be exchanged
        self.receptor_chains = list(receptor_chains)
        self.ligand_chains = list(ligand_chains)
        self.if_th = if_th

    def __len__(self):
        return len(self.pdbs)

    def __getitem__(self, idx):
        pdb, rec_chains, lig_chains = self.pdbs[idx], self.receptor_chains[idx], self.ligand_chains[idx]
        chain2blocks = pdb_to_list_blocks(pdb, rec_chains + lig_chains, dict_form=True, allow_het=True) # might be small molecules
        rec_blocks, lig_blocks = [], []
        for c in rec_chains: rec_blocks.extend(chain2blocks.get(c, []))
        for c in lig_chains: lig_blocks.extend(chain2blocks.get(c, []))
        (blocks1, blocks2), _ = blocks_interface(rec_blocks, lig_blocks, self.if_th)
        item = blocks_to_data_simple(blocks1, blocks2)
        return item

    def collate_fn(self, batch):
        results = {}
        for key in batch[0]:
            values = [item[key] for item in batch]
            if values[0] is None:
                results[key] = None
                continue
            if key == 'lengths':
                results[key] = torch.tensor(values, dtype=torch.long)
            else:
                try:
                    results[key] = torch.cat(values, dim=0)
                except RuntimeError:
                    print(key, [v.shape for v in values])
        return results


def abs_path(index_file, path):
    if os.path.exists(path):
        return path # already absolute path
    root_dir = os.path.dirname(index_file)
    return os.path.abspath(os.path.join(root_dir, path))


def main(args):

    # preprocess data
    ids = []
    if args.data_list is None:
        print_log(f'Infering data from {args.complex_pdb}')
        r, l = args.split_chains.split('_')
        dataset = Dataset([args.complex_pdb], [list(r)], [list(l)])
        ids.append(args.complex_pdb)
    else:
        print_log(f'Infering data from {args.data_list}')
        with open(args.data_list, 'r') as fin: lines = fin.readlines()
        pdbs, rec_chains, lig_chains = [], [], []
        for line in lines:
            path, split_chains = re.split(r'\s+', line.strip())
            pdbs.append(abs_path(args.data_list, path))
            r, l = split_chains.split('_')
            rec_chains.append(list(r))
            lig_chains.append(list(l))
        dataset = Dataset(pdbs, rec_chains, lig_chains)
        ids = pdbs
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn)
    
    # load models
    print_log(f'Using checkpoints: {args.ckpts}')
    models = [
        torch.load(ckpt, map_location='cpu') for ckpt in args.ckpts
    ]
    device = torch.device('cpu' if args.gpu < 0 else f'cuda:{args.gpu}')
    for model in models: model.to(device)

    # infer
    pkds, confs = [], []
    for batch in dataloader:
        for key in batch:
            batch[key] = batch[key].to(device)
        batch_pkd, batch_conf = [], []
        for model in models:
            pkd, conf = model.infer(batch)
            batch_pkd.append(pkd)
            batch_conf.append(conf)
        pkds.extend((sum(batch_pkd) / len(batch_pkd)).cpu().tolist())
        confs.extend((sum(batch_conf) / len(batch_pkd)).cpu().tolist())

    # print results
    print()
    print('=' * 10 + 'Results' + '=' * 10)
    print('pdb\tpKd\tKd(nM)\tbinding_confidence')
    for i, pkd, conf in zip(ids, pkds, confs):
        kd = 10**(-pkd) * (10**9) # unit: nm
        print(f'{i}\t{round(pkd, 3)}\t{round(kd, 3)}\t{round(conf, 3)}')


if __name__ == '__main__':
    main(parse())