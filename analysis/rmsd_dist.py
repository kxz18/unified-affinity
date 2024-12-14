#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import ast
import argparse

import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from .visualize.histplot import histplot


def parse():
    parser = argparse.ArgumentParser(description='RMSD curve')
    parser.add_argument('--index', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default=os.path.join(os.path.dirname(__file__), 'images', 'rmsd'))
    return parser.parse_args()


def main(args):
    with open(args.index, 'r') as fin:
        lines = fin.readlines()

    data = {
        'rmsd': [],
        'rmsd_norm': [],
        'seq_len': []
    }

    for line in lines:
        line = line.strip().split('\t')
        props = ast.literal_eval(line[-1])
        rmsd = props['rmsd']
        if isinstance(rmsd, tuple):
            rmsd = rmsd[1]
            assert len(rmsd) == len(props['sequence'])
            data['rmsd'].extend(rmsd)
            data['rmsd_norm'].extend(rmsd)
            data['seq_len'].extend([len(props['sequence']) for _ in rmsd])
        else:
            data['rmsd'].append(props['rmsd'])
            data['rmsd_norm'].append(props['rmsd'] / len(props['sequence']))
            data['seq_len'].append(len(props['sequence']))

    os.makedirs(args.out_dir, exist_ok=True)

    histplot(data, x='rmsd', save_path=os.path.join(args.out_dir, 'rmsd.png'))
    plt.clf()
    histplot(data, x='rmsd_norm', save_path=os.path.join(args.out_dir, 'rmsd_norm.png'))

    print(f'pearson between rmsd and seq len: {pearsonr(data["rmsd"], data["seq_len"])}')
    print(f'pearson between rmsd_norm and seq len: {pearsonr(data["rmsd_norm"], data["seq_len"])}')


if __name__ == '__main__':
    main(parse())
