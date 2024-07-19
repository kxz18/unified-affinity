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