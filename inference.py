#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse
import json
import yaml
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from data import create_dataset, create_dataloader
from utils.logger import print_log
from utils.random_seed import setup_seed, SEED
from utils.config_utils import overwrite_values


def parse():
    parser = argparse.ArgumentParser(description='inference dG')
    parser.add_argument('--config', type=str, required=True, help='Path to the test configure')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the checkpoint')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the results')

    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers to use')

    parser.add_argument('--gpu', type=int, default=0, help='GPU to use, -1 for cpu')
    return parser.parse_known_args()


def main(args, opt_args):
    # load config
    config = yaml.safe_load(open(args.config, 'r'))
    config = overwrite_values(config, opt_args)

    # load model
    model = torch.load(args.ckpt, map_location='cpu')
    print_log(f'Load model from {args.ckpt}')
    # if not isinstance(model, torch.nn.Module):
    #     weights = model
    #     ckpt_dir = os.path.dirname(args.ckpt)
    #     namespace = json.load(open(os.path.join(ckpt_dir, 'namespace.json'), 'r'))
    #     model = models.create_model(argparse.Namespace(**namespace))
    #     model.load_state_dict(weights)
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    model.to(device)
    model.eval()

    # load data
    _, _, test_set = create_dataset(config['dataset'])
    test_loader = create_dataloader(test_set, config['dataloader'].get('test', config['dataloader']))
    
    # save path
    if args.save_path is None:
        root_dir = os.path.join(os.path.dirname(args.ckpt), '..', 'results')
        os.makedirs(root_dir, exist_ok=True)
        save_path = os.path.join(root_dir, os.path.basename(args.ckpt).split('.')[0] + '_results.jsonl')
    else:
        save_path = args.save_path

    fout = open(save_path, 'w')

    idx = 0
    # batch_id = 0
    for batch in tqdm(test_loader):
        with torch.no_grad():
            # move data
            for k in batch:
                if hasattr(batch[k], 'to'):
                    batch[k] = batch[k].to(device)
            label = batch.pop('label').tolist()
            names = batch.pop('name')
            
            pred_pk, confidence = model.infer(batch)
            pred_pk, confidence = pred_pk.tolist(), confidence.tolist()

            for pk, c, gt, n in zip(pred_pk, confidence, label, names):
                item_id = test_set.get_id(idx)
                out_dict = {
                        'id': item_id,
                        'pred': pk,
                        'confidence': c,
                        'label': gt,
                        'type': n
                    }
            
                fout.write(json.dumps(out_dict) + '\n')
                idx += 1
    
    fout.close()

if __name__ == '__main__':
    args, opt_args = parse()
    print_log(f'Overwritting args: {opt_args}')
    setup_seed(SEED)
    main(args, opt_args)