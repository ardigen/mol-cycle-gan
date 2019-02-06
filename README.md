# Mol-CycleGAN - a generative model for molecular optimization
Official implementation of Mol-CycleGAN for molecular optimization.

Keras CycleGan implementation is based on <a href="https://github.com/tjwei/GANotebooks">[tjwei/GANotebooks]</a>.


## Requirements
We highly recommend to use conda for package management -- the `environment.yml` file is provided.

The environment can be created by running:
```
conda env create -f environment.yml
```

We use <a href="https://github.com/wengong-jin/icml18-jtnn">Junction Tree Variational Autoencoder implementation</a> as a submodule in Mol-CycleGAN code.
After cloning this repo, the following script should be executed before running the code
```
./scripts/init_repo.sh 
```

## Datasets
We provide the user with all datasets needed to reproduce the aromatic rings experiments.

Downloading all the input data (ZINC 250k dataset and related JT-VAE encodings) can be performed by running:
```
./scripts/download_input_data.sh
```

Downloading all the data from aromatic rings experiments (train / test splits of datasets, molecules returned by Mol-CycleGAN and related SMILES) can be performed by running:
```
./scripts/download_ar_data.sh
```


## Basic use
This code is an implementation of CycleGan for molecular optimization.

Training of the model can be performed by running:
```
python train.py
```
with specified training parameters.

After the model is trained and the test set translation is generated, for decoding the molecules the JT-VAE code should be used. This can be performed by running:
```
python decode.py
```
with specified decoding parameters.


## Experiments
We provide all the data and code needed to reproduce the `Aromatic rings` experiment.

1. In `data/input_data/aromatic_rings/datasets_generator_aromatic_rings.ipynb` one can find the data factory - the code that is needed to create train and test sets used in the experiment.

2. Training of the model can be performed by running `./scripts/run_aromatic_rings_training.sh`. It calls the `train.py` function with base parameters, which are set to process the aromatic rings data.

3. Decoding the molecules can be performed by running `./scripts/run_aromatic_rings_decoding.sh`. It calls the `decode.py` function with base parameters, which are set to process the aromatic rings data.

4. The analysis of the output is provided in the notebook `experiments/aromatic_rings.ipynb`.

## Disclaimer
The code for Mol-Cycle-Gan was natively written in Python3, however, the JT-VAE package is written in Python2. To ensure the ease of use, we used downgraded versions of packages, so that the entire experiment can be run in a single environment.
Since many of those packages are outdated, we strongly recommend using the ```environment.yml``` file provided to construct the working environment.