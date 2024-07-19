from typing import Callable
from tqdm import tqdm

import numpy as np
import torch
import sympy

from utils import register as R


class MixDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, *datasets, collate_fn: Callable=None, weights=None, names=None) -> None:
        super().__init__()
        self.datasets = datasets
        self.cum_len = []
        self.total_len = 0
        for dataset in datasets:
            self.total_len += len(dataset)
            self.cum_len.append(self.total_len)
        self.collate_fn_specify = collate_fn
        #self.collate_fn = self.datasets[0].collate_fn if collate_fn is None else collate_fn
        if weights is not None: assert len(weights) == len(datasets)
        if names is not None: assert len(names) == len(datasets)
        self.weights = weights
        self.names = names
        self.dynamic_idx = []
        self.update_epoch()
    
    def update_epoch(self):
        for dataset in self.datasets:
            if hasattr(dataset, 'update_epoch'):
                dataset.update_epoch()
        if self.weights is None:
            self.dynamic_idx = [i for i in range(self.total_len)]
        else:
            self.dynamic_idx = []
            start_idx = 0
            for i, (w, dataset) in enumerate(zip(self.weights, self.datasets)):
                add_len, end_idx = int(len(dataset) * w), self.cum_len[i]
                self.dynamic_idx.extend(np.random.choice(
                    list(range(start_idx, end_idx)),
                    size=add_len, replace=True
                ))
                start_idx = end_idx

    def get_id(self, idx):
        idx = self.dynamic_idx[idx]
        last_cum_len = 0
        for i, cum_len in enumerate(self.cum_len):
            if idx < cum_len:
                return self.datasets[i].get_id(idx - last_cum_len)
            last_cum_len = cum_len
        return None # this is not possible

    def get_len(self, idx):
        idx = self.dynamic_idx[idx]
        last_cum_len = 0
        for i, cum_len in enumerate(self.cum_len):
            if idx < cum_len:
                return self.datasets[i].get_len(idx - last_cum_len)
            last_cum_len = cum_len
        return None # this is not possible

    def __len__(self):
        return len(self.dynamic_idx)
    
    def __getitem__(self, idx):
        idx = self.dynamic_idx[idx]
        last_cum_len = 0
        for i, cum_len in enumerate(self.cum_len):
            if idx < cum_len:
                item = self.datasets[i].__getitem__(idx - last_cum_len)
                if self.names is not None: item['name'] = self.names[i]
                return item
            last_cum_len = cum_len
        return None # this is not possible
    
    def collate_fn(self, batch):
        if self.collate_fn_specify is not None:
            return self.collate_fn_specify(batch)
        if 'name' in batch[0]:
            names = [item['name'] for item in batch]
        else:
            names = []
        # delete name
        new_batch = []
        for item in batch:
            if 'name' in item: del item['name']
            new_batch.append(item)
        results = self.datasets[0].collate_fn(new_batch)
        results['name'] = names
        if 'X' not in results:
            print(names)
        return results
    

@R.register('DynamicBatchWrapper')
class DynamicBatchWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, complexity, ubound_per_batch) -> None:
        super().__init__()
        self.dataset = dataset
        self.indexes = [i for i in range(len(dataset))]
        self.complexity = complexity
        self.eval_func = sympy.lambdify('n', sympy.simplify(complexity))
        self.ubound_per_batch = ubound_per_batch
        self.total_size = None
        self.batch_indexes = []
        self._form_batch()

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        elif hasattr(self.dataset, attr):
            return getattr(self.dataset, attr)
        else:
            raise AttributeError(f"'DynamicBatchWrapper'(or '{type(self.dataset)}') object has no attribute '{attr}'")

    def update_epoch(self):
        if hasattr(self.dataset, 'update_epoch'):
            self.dataset.update_epoch()
        self._form_batch()

    ########## overload with your criterion ##########
    def _form_batch(self):

        np.random.shuffle(self.indexes)
        last_batch_indexes = self.batch_indexes
        self.batch_indexes = []

        cur_complexity = 0
        batch = []

        for i in tqdm(self.indexes):
            item_len = self.eval_func(self.dataset.get_len(i))
            if item_len > self.ubound_per_batch:
                continue
            cur_complexity += item_len
            if cur_complexity > self.ubound_per_batch:
                self.batch_indexes.append(batch)
                batch = []
                cur_complexity = item_len
            batch.append(i)
        self.batch_indexes.append(batch)

        if self.total_size is None:
            self.total_size = len(self.batch_indexes)
        else:
            # control the lengths of the dataset, otherwise the dataloader will raise error
            if len(self.batch_indexes) < self.total_size:
                num_add = self.total_size - len(self.batch_indexes)
                self.batch_indexes = self.batch_indexes + last_batch_indexes[:num_add]
            else:
                self.batch_indexes = self.batch_indexes[:self.total_size]

    def __len__(self):
        return len(self.batch_indexes)
    
    def __getitem__(self, idx):
        return [self.dataset[i] for i in self.batch_indexes[idx]]
    
    def collate_fn(self, batched_batch):
        batch = []
        for minibatch in batched_batch:
            batch.extend(minibatch)
        return self.dataset.collate_fn(batch)