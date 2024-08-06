import os
import sys

import numpy as np
np.random.seed(0)

valid_size = 200

pl_test = os.path.join('test_sets', 'protein_ligand.txt')
ab_test = os.path.join('test_sets', 'antibody.txt')
pp_test = os.path.join('test_sets', 'protein_protein.txt')
nl_test = os.path.join('test_sets', 'RNA_ligand.txt')

datasets = [
    # ('docked_PDBbind', True, pl_test), # split test
    # ('docked_SAbDab', True, ab_test), # split test
    # (os.path.join('PDBbind', 'PL-refined'), True, pl_test),
    # (os.path.join('PDBbind', 'PP'), False, pp_test), # no need to split test
    # ('SAbDab', True, ab_test),
    # ('PPAB', False, ab_test), # training is pp, test is ab, manually change the name 
    # (os.path.join('PDBbind', 'NL'), True, nl_test)
    ('vinadocked_PDBbind', True, pl_test)
]

root_dir = sys.argv[1]


for dataset, need_test, test_id_path in datasets:
    test_ids = {}
    with open(os.path.join(root_dir, test_id_path), 'r') as fin:
        lines = fin.readlines()
    for line in lines:
        _id = line.split('\t')[0][:4]
        assert _id not in test_ids
        test_ids[_id.lower()] = True
    print(test_ids)

    data_dir = os.path.join(root_dir, dataset)
    with open(os.path.join(data_dir, 'index.txt'), 'r') as fin:
        lines = fin.readlines()
    id2lines, test_id2lines = {}, {}
    for line in lines:
        _id = line.split('\t')[0]
        if _id[:4].lower() in test_ids:
            test_id2lines[_id] = line
        else:
            id2lines[_id] = line

    # split train/validation
    keys = list(id2lines.keys())
    if dataset == 'PPAB':
        train_ids, valid_ids = keys, []
    else:
        np.random.shuffle(keys)
        train_ids, valid_ids = keys[valid_size:], keys[:valid_size]
    
    if len(train_ids):
        with open(os.path.join(data_dir, 'train.txt'), 'w') as fout:
            for i in train_ids: fout.write(id2lines[i])
    
    if len(valid_ids):
        with open(os.path.join(data_dir, 'valid.txt'), 'w') as fout:
            for i in valid_ids: fout.write(id2lines[i])
    
    if need_test:
        with open(os.path.join(data_dir, 'test.txt'), 'w') as fout:
            for i in test_id2lines: fout.write(test_id2lines[i])
