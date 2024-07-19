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

## Data Processing

### PDBbind

protein-protein
protein-ligand: refined & general
protein-RNA/DNA
RNA/DNA-ligand

```bash
python -m scripts.data_process.process_PDBbind_PP --data_dir /share/medicalData/downstream/PDBBind_All/raw/ --out_dir datasets/PDBbind
```

### SAbDab

antibody

```bash
python -m scripts.data_process.process_SAbDab --summary /share/medicalData/downstream/Antibody/SAbDab/sabdab_summary_all.tsv --struct_dir /share/medicalData/downstream/Antibody/SAbDab/all_structures/imgt/ --out_dir datasets/SAbDab
```


### Docked PDBbind PL

PL complexes with local docking from Glide

```bash
python -m scripts.data_process.process_docked_PDBbind --data_dir /share/medicalData/downstream/screen/docked_PDBbind/raw/ --out_dir datasets/docked_PDBbind
```

### Docked SAbDab

antibody docked with ESMFold

first construct dataset

```bash
python -m scripts.data_process.construct_docked_sabdab --summary /share/medicalData/downstream/Antibody/SAbDab/sabdab_summary_all.tsv --struct_dir /share/medicalData/downstream/Antibody/SAbDab/all_structures/imgt/ --out_dir /share/medicalData/downstream/Antibody/docked_SAbDab_affinity
```

then process data


## Test sets

protein-protein: benchmark5.5

antigen-antibody: An expanded benchmark for antibody-antigen docking and affinity prediction reveals insights into antibody recognition determinants

protein-small molecule: hard test set

RNA/DNA-small molecule: zero-shot