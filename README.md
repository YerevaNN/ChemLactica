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
In order to run the optimization algorithm, define the Oracle as a class, that is responsible for calculating the Oracle fuction for given molecules.

```python
# Oracle implementation scheme

class ExampleOracle:
    def __init__(self, ...):
        # maximum number of oracle calls to make
        self.max_oracle_calls: int = ...

        # the frequence with which to log
        self.freq_log: int = ...

        # the buffer to keep track of all unique molecules generated
        self.mol_buffer: Dict = ...

        # the maximum possible oracle score or an upper bound
        self.max_possible_oracle_score: float = ... 

    def __call__(self, molecules):
        """
            Evaluate and return the oracle scores for molecules. Log the intermediate results if necessary.
        """
        ...
        return oracle_scores

    @property
    def finish(self):
        """ 
            Specify the stopping condition for the optimization process.
        """
        return stopping_condition
```

Define configuration and hyperparameters used for the optimization process in a yaml file.

```yaml
# yaml config scheme

checkpoint_path: /path/to/model_dir
tokenizer_path: /path/to/tokenizer_dir

... optimization algorithm hyperparameter (pool size, number of similar molecules to use, etc.) ...

generation_config:
  ... molecule generation hyperparameters ...

strategy: [rej-sample-v2] # or use [default] for not performing the fine-tuning step during the optimization.

rej_sample_config:
    ... fine tuning hyperparameters ...
```

Putting everything toghether and running the optimization process.

```python
from chemlactica.mol_opt.optimization import optimize

# Load config
config = yaml.safe_load(open(path_to_yaml_config))

# Load the model and the tokenizer
model = AutoModelForCausalLM.from_pretrained(...)
tokenizer = AutoTokenizer.from_pretrained(...)

# Create Oracle
oracle = ExampleOracle(...)

# Call the optimize function to optimize against the defined oracle
optimize(
    model, tokenizer,
    oracle, config
)
```

[example_run.py]() illustrates a full working example of an optimization run. For more complex examples refer to the [ChemlacticaTestSuit]() repository [mol_opt/run.py]() and [retmol/run_qed.py]() files.

## Tests
The test for running the a small sized model with the same
architecture as galactica on a small set of data is located at /tests/precommit_test.py and can be called as follows:
``` bash
python -m unittest precommit_test.py
```
This test is also run as part of the CI pipeline on the main branch on a public github runner.
