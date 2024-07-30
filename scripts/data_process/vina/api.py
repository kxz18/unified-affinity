import os
from typing import Optional, Tuple, List
from dataclasses import dataclass
from multiprocessing import Process, Queue

from rdkit import Chem

from .vina_dock import docking_pipeline


@dataclass
class Task:
    id: str # id of this task 
    receptor_pdbqt: Optional[str] = None #receptor pdbqt file
    ligand_sdf_list: Optional[List[str]] = None # list of ligand sdf string with only one molecule in each string
    ligand_smi_list: Optional[List[str]] = None # list of smiles if sdf strings are not provided
    center: Tuple[float, float, float] = None #center of docking box, [x, y, z]
    output_dir: str = None #output directory
    lig_name_list: Optional[list] = None #ligand name list, if None, use LIG1 LIG2 ...
    n_cpu: int = 1 #number of cpu to use in each worker
    exhaustiveness: Optional[int] = None #exhaustiveness of docking, higher value will cost more time, default 8 for routine docking, 1 for pose initial of flexible docking, 32 is recommended for high accuracy
    n_rigid: Optional[int] = None #number of rigid docking pose to keep, default 5 for routine docking, 1 for pose initial of flexible docking
    flexible: bool = False #whether sample receptor flexibility after rigid docking
    receptor_pdb: Optional[str] = None #receptor pdb file, optional, but necessary if flexible=True
    n_flexible: int = 5 #number of flexible docking pose to keep, only useful when flexible=True, in total, n_rigid*n_flexible poses will be kept for each ligand
    verbose: bool = False

    def prepare_sdf(self):
        if self.ligand_sdf_list is not None:
            return
        assert self.ligand_smi_list is not None
        os.makedirs(self.output_dir, exist_ok=True)
        out_sdf = os.path.join(self.output_dir, 'mols.sdf')
        writer = Chem.SDWriter(out_sdf)
        for smi in self.ligand_smi_list:
            mol = Chem.MolFromSmiles(smi)
            writer.write(mol)
        writer.close()
        with open(out_sdf,'r') as f:
            ligand_sdf_list = f.read().split('$$$$\n')[:-1]
        self.ligand_sdf_list = [i+'$$$$\n' for i in ligand_sdf_list]


class VinaDock:
    def __init__(self, num_workers: int=8) -> None:
        '''
            num_workers: number of workers. The number of CPUs used by each
                worker can be set in each task by n_cpu
        '''
        self.task_queue = Queue()
        self.result_queue = Queue()

        # Create worker processes
        self.processes = []
        for _ in range(num_workers):
            p = Process(target=self.worker)
            self.processes.append(p)
            p.start()

        self.closed = False

    def worker(self):
        """Worker process to perform structure prediction on a given GPU."""
        while True:
            task = self.task_queue.get()
            if task is None:
                break
            try:
                output = self.dock(task)
                self.result_queue.put(output)
            except Exception as e:
                self.result_queue.put((task, None, e))

    def put(self, task: Task):
        assert not self.closed, f'This instance has been closed, please create a new one'
        self.task_queue.put(task)

    def get(self):
        return self.result_queue.get()
    
    def finish_cnt(self):
        return self.result_queue.qsize()

    def dock(self, task: Task):
        task.prepare_sdf()
        output = docking_pipeline(
            receptor_pdbqt = task.receptor_pdbqt,
            ligand_sdf_list = task.ligand_sdf_list,
            center=task.center,
            output_dir=task.output_dir,
            lig_name_list=task.lig_name_list,
            n_cpu=task.n_cpu,
            exhaustiveness=task.exhaustiveness,
            n_rigid=task.n_rigid,
            flexible=task.flexible,
            receptor_pdb=task.receptor_pdb,
            n_flexible=task.n_flexible,
            verbose=task.verbose
        )
        liganmes, energies = output # energies have two lists: one for rigid results, one for flexible results
        if liganmes is None:
            return (task, None, energies)
        ligfiles = []
        receptor_file = task.receptor_pdb if task.receptor_pdbqt is None else task.receptor_pdbqt
        basename = os.path.splitext(os.path.basename(receptor_file))[0]
        for name in liganmes:
            ligfiles.append(os.path.join(task.output_dir, f'{basename}_{name}.sdf'))
        ligfiles = ligfiles[:len(energies[0])]
        return (task, ligfiles, energies) # sorted by energy

    def close(self):
        for p in self.processes: self.task_queue.put(None)
        for p in self.processes: p.join()
        self.closed = True

    def __del__(self):
        if not self.closed: self.close()


if __name__ == '__main__':
    vina_dock = VinaDock(num_workers=8)

    vina_dock.put(Task(
        id='example',
        ligand_smi_list=['CC[C@@]1(C(=O)N[C@H](C)c2ccc(Cl)cc2)[C@@H](C)C1(Cl)Cl', 'Cc1cccc(Cl)c1NC(=O)c1cnc(Nc2cccc(C(=O)N[C@@H]3CC[NH2+]CC3(F)F)c2)s1'],
        receptor_pdb='./example_data/7std.pdb',
        center=[28.5, 12.5, 33.2], # You need to decide the center for local docking
        output_dir='./example_out',
        n_rigid=3
    ))
    print(vina_dock.get())

    vina_dock.close()