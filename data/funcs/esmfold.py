import os
from typing import List

import torch
import esm

from utils.singleton import singleton
from multiprocessing import Process, Queue

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ESMFOLD_CKPT = os.path.join(FILE_DIR, 'esm_ckpt')


@singleton
class ESMFold:
    def __init__(self) -> None:
        self.models = {}

    def load_model(self, gpu=0):
        print(f'Loading ESMFold V1 to GPU:{gpu}')
        torch.hub.set_dir(ESMFOLD_CKPT)
        model = esm.pretrained.esmfold_v1()
        model = model.eval().to(torch.device(f'cuda:{gpu}'))
        model.set_chunk_size(128)
        print(f'ESMFold V1 loaded to GPU:{gpu}')
        return model
    
    def get_model_on_gpu(self, gpu):
        assert gpu >= 0 and gpu < torch.cuda.device_count()
        if gpu not in self.models:
            self.models[gpu] = self.load_model(gpu)
        return self.models[gpu]


@torch.no_grad()
def structure_prediction(seqs: List[str], out_files: List[str], gpu=0):
    assert len(seqs) == len(out_files)
    model = ESMFold().get_model_on_gpu(gpu)
    for seq, out_file in zip(seqs, out_files):
        output = model.infer_pdb(seq)
        with open(out_file, 'w') as fout:
            fout.write(output)


@torch.no_grad()
def worker(task_queue, result_queue, gpu):
    """Worker process to perform structure prediction on a given GPU."""
    #model = shared_var_dict[gpu]
    model = ESMFold().get_model_on_gpu(gpu)
    while True:
        task = task_queue.get()
        if task is None:
            break
        seq, out_file = task
        if os.path.exists(out_file):
            result_queue.put((seq, out_file, gpu))
            continue
        try:
            output = model.infer_pdb(seq)
        except RuntimeError:
            result_queue.put((seq, None, gpu))
            continue
        with open(out_file, 'w') as fout:
            fout.write(output)
        result_queue.put((seq, out_file, gpu))


@torch.no_grad()
def parallel_structure_prediction(seqs: List[str], out_files: List[str], gpus: List[int], silent: bool=False):
    """Distribute sequences evenly across GPUs and run predictions in parallel."""
    assert len(seqs) == len(out_files), "The number of sequences and output files must match."
    
    task_queue = Queue()
    result_queue = Queue()

    # Create worker processes
    processes = []
    for gpu in gpus:
        p = Process(target=worker, args=(task_queue, result_queue, gpu))
        processes.append(p)
        p.start()

    # Distribute tasks to workers
    for seq, out_file in zip(seqs, out_files):
        task_queue.put((seq, out_file))

    # Add a sentinel (None) to signal workers to exit
    for _ in gpus:
        task_queue.put(None)

    # Collect results from workers
    for _ in range(len(seqs)):
        seq, out_file, gpu = result_queue.get()
        if not silent: print(f'GPU {gpu} processed {out_file}')
        if out_file is None:
            print(f'{seq} on {gpu} failed')
            continue
        yield out_file

    # Ensure all processes have finished
    for p in processes:
        p.join()


if __name__ == '__main__':
    import sys
    from tqdm import tqdm

    queue = parallel_structure_prediction(
        [sys.argv[1] for _ in range(20)],
        [f'tmp{i}.pdb' for i in range(20)],
        gpus=[0, 1, 2, 3, 4]
    )
    for res in queue:
        continue
    print('finished')