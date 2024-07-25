# Chemlactica / Chemma: Large Language Models for Small Molecules

TL;DR
* A family of models that "understand" small organic molecules (SMILES), their basic properties (molecular weight, QED, SAS, TPSA, CLogP, ...), and similarities between molecules (Tanimoto over ECFC4).
* [**Chemlactica-125M** 🤗](https://huggingface.co/yerevann/chemlactica-125m) and [**-1.3B** 🤗](https://huggingface.co/yerevann/chemlactica-1.3b) are trained on top of Meta's [Galactica models](https://huggingface.co/facebook/galactica-1.3b).
* [**Chemma-2B** 🤗](https://huggingface.co/yerevann/chemma-2b) is built on top of Google's [Gemma-2B](https://huggingface.co/google/gemma-2b).
* All models are trained on **40B** tokens covering 100M+ molecules from PubChem. [Check the corpus at 🤗](https://huggingface.co/datasets/yerevann/PubChemForLM).
* A prompt like `</s>[SAS]2.25[/SAS][SIMILAR]0.62 CC(=O)OC1=CC=CC=C1C(=O)O[/SIMILAR][START_SMILES]` will generate a molecule that has ~2.25 SAS score and has ~0.62 similarity score to the given molecule.
* The models can be easily tuned to perform property prediction (~0.3 RMSE on [FreeSolv](https://paperswithcode.com/sota/molecular-property-prediction-on-freesolv) from MoleculeNet).
* The models wrapped into a **genetic-like optimization algorithm** beat all **molecular optimization** benchmarks we tried.
  * [**Practical Molecular Optimization**](https://arxiv.org/abs/2206.12411)
    * **17.5** vs 16.2 (previous SOTA: [Genetic-guided GFlowNets](https://arxiv.org/abs/2402.05961)).
  * Optimization for **docking** with AutoDock Vina
    * 3-4x fewer oracle calls for generating 100 _good_ molecules than previous SOTA ([Beam Enumeration](https://arxiv.org/abs/2309.13957)).
  * QED optimization from the [RetMol paper](https://arxiv.org/abs/2208.11126)
    * **99%** success rate with 10K oracle calls with Chemlactica-125M (vs. 96% with 50K calls of the original paper).
* Read the details in the paper [_Small Molecule Optimization with Large Language Models_](https://yerevann.com/papers/small-molecule-optimization-with-large-language-models.pdf).

We are looking forward to the community utilizing these models for solving various problems in molecular design.

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
