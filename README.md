# ChemLactica

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
### Training
The script for training the model is ```train.py```
which can be run from the command line using the following syntax:
``` bash
python train.py --model_type galactica/125m --training_data_dir .small_data/train --valid_data_dir .small_data/valid --max_steps 128 --eval_steps 64 --track --eval_accumulation_steps 8
```
Here's what these arguments do
- `--model_type <model_name>` - type of model to train, one of galactica/125m, galactica/1.3B , galactica/20B
- `--training_data_dir` - directory containing training data
- `--valid_data_dir` - directory containing validation data
- `--max_steps` - maximum number of steps to run training for
- `--eval_steps` - the interval at which to run evaluation
- `--track` - whether to track model checkpoint or not
- `--eval_accumulation_steps` - the number of steps after which to move the prediction tensor from GPU
                        to CPU during the evaluation (specified to avoid OOM errors)

## Tests
The test for running the a small sized model with the same
architecture as galactica on a small set of data is located at /tests/precommit_test.py and can be called as follows:
``` bash
python -m unittest precommit_test.py
```
This test is also run as part of the CI pipeline on the main branch on a public github runner.
