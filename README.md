# Chemlactica / Chemma: Large Language Models for Small Molecules

TL;DR
* A family of models that understand small organic molecules written in SMILES, their basic properties, and similarities between molecules.
* [**Chemlactica-125M** 🤗](https://huggingface.co/yerevann/chemlactica-125m) and [**-1.3B** 🤗](https://huggingface.co/yerevann/chemlactica-1.3b) trained on top of Meta's [Galactica models](https://huggingface.co/facebook/galactica-1.3b).
* [**Chemma-2B** 🤗](https://huggingface.co/yerevann/chemma-2b) is built on top of Google's [Gemma-2B](https://huggingface.co/google/gemma-2b).
* All models are trained on **40B** tokens covering 100M+ molecules from PubChem. [The dataset is also available at 🤗](https://huggingface.co/datasets/yerevann/PubChemForLM).
* A prompt like `</s>[SAS]2.25[/SAS][SIMILAR]0.62 CC(=O)OC1=CC=CC=C1C(=O)O[/SIMILAR][START_SMILES]` will generate a molecule that has ~2.25 SAS score and has ~0.62 similarity score to the given molecule.
* The models can be easily tuned to perform property prediction (~0.3 RMSE on FreeSolv from MoleculeNet).
* The models wrapped into a **genetic-like optimization algorithm** beat all **molecular optimization** benchmarks we tried.
  * [**Practical Molecular Optimization**](https://arxiv.org/abs/2206.12411): **17.5** vs 16.2 (previous SOTA: [Genetic-guided GFlowNets](https://arxiv.org/abs/2402.05961)).
  * Optimization for **docking** with AutoDock Vina: 3-4x less oracle calls for generating 100 _good_ molecules than previous SOTA.
  * QED optimization from the [RetMol paper](https://arxiv.org/abs/2208.11126): **99%** success rate with 10K oracle calls with Chemlactica-125M (vs. 96% with 50K calls).
* All details in the paper [Small Molecule Optimization with Large Language Models](https://yerevann.com/papers/small-molecule-optimization-with-large-language-models).
 

## Table of contents
- [Description](#Description)
- [Prerequisites](#Prerequisites)
- [Installation](#Installation)
- [Usage](#Usage)
- [Tests](#Tests)

## Description
Fine tuning the galactica models on chemistry data from PubChem.
## Prerequisites
- Python 3.11
- conda
## Installation
```bash
conda create -n ChemLactica python=3.11 -y -f environment.yml
conda activate chemlactica
```

## Usage
### Pretraining
Instructions coming soon...

### Fine-tuning
Instructions coming soon...

### Molecular optimization
Instructions coming soon...

## Tests
The test for running the a small sized model with the same
architecture as galactica on a small set of data is located at /tests/precommit_test.py and can be called as follows:
``` bash
python -m unittest precommit_test.py
```
This test is also run as part of the CI pipeline on the main branch on a public github runner.
