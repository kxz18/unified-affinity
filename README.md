# Affinity Prediction with Unified Modeling

## Environment

```bash
conda env create -f env.yaml

pip install "fair-esm[esmfold]"

pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'
```

the einops version should be specified as 0.6.1, otherwise 0.7.0 will be installed and cause error

You need to `unset LD_LIBRARY_PATH` before running, otherwise ESMFold will throw CUDA error.

## Model

Core codes for the model are located at `./models/GET/get.py`

## Usage

**1. checkpoints**

First download the checkpoints of [3 models released](https://github.com/kxz18/unified-affinity/releases) and put them under the directory `./checkpoints` of this project.

**2. prepare data**

The structure of the complex should be prepared in the format of pdb, with two binding partners split into different chains. We provide two examples located at `./demo_data`. *5c7x_antibody_B_MN.pdb* contains an antibody with chain B as the antigen and chains M&N as the antibody. *6ueg_pro_lig_C_L.pdb* contains a protein-ligand complex where chain C is the protein and chain L is the small molecule.

**3. run the model**

For inference on single complex (e.g.):

```bash
python api.py --complex_pdb demo_data/5c7x_antibody_B_MN.pdb --split_chains B_MN --gpu 0
```

Example output:

```bash
2024-08-06 11:58:38::INFO::Infering data from demo_data/5c7x_antibody_B_MN.pdb
2024-08-06 11:58:38::INFO::Using checkpoints: ['./checkpoints/model0.ckpt', './checkpoints/model1.ckpt', './checkpoints/model2.ckpt']

==========Results==========
pdb     pKd     Kd(nM)  binding_confidence
demo_data/5c7x_antibody_B_MN.pdb        11.052  0.009   0.901
```

For inference on batch complexes (e.g.):

```bash
python api.py --data_list demo_data/list.txt --gpu 0
```

where the contents of the `demo_data/list.txt` contains lines of pdbs and splits of chains:

```bash
# demo_data/list.txt
5c7x_antibody_B_MN.pdb  B_MN
6ueg_pro_lig_C_L.pdb    C_L
```

Example output:

```bash
2024-08-06 11:59:31::INFO::Infering data from demo_data/list.txt
2024-08-06 11:59:31::INFO::Using checkpoints: ['./checkpoints/model0.ckpt', './checkpoints/model1.ckpt', './checkpoints/model2.ckpt']

==========Results==========
pdb     pKd     Kd(nM)  binding_confidence
/disks/disk5/private/kxz/unified-affinity/demo_data/5c7x_antibody_B_MN.pdb      11.052  0.009   0.901
/disks/disk5/private/kxz/unified-affinity/demo_data/6ueg_pro_lig_C_L.pdb        3.585   259952.278      0.597
```

**WARNING1**: Note that for complexes incorporating small molecules, the molecule should be separated into a different chain from its receptor (which is not the convention for most pdbs), and included in one single residue.

**WARNING2**: Only when the confidence is high (e.g. above 0.7), the predicted Kd is meaningful. Otherwise there might be no binding between these two molecules.