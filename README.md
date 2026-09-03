# Chemlactica and Chemma

Chemlactica and Chemma are causal language models for small-molecule generation,
property conditioning, and molecular optimization. They use structured text tags
for SMILES, molecular properties, and molecular similarity.

## Models and data

| Model | Base model | Model card |
| --- | --- | --- |
| Chemlactica-125M | Galactica-125M | [Hugging Face](https://huggingface.co/yerevann/chemlactica-125m) |
| Chemlactica-1.3B | Galactica-1.3B | [Hugging Face](https://huggingface.co/yerevann/chemlactica-1.3b) |
| Chemma-2B | Gemma-2B | [Hugging Face](https://huggingface.co/yerevann/chemma-2b) |

The models were trained for 40 billion tokens over more than 100 million PubChem
molecules. The public training corpus is
[PubChemForLM](https://huggingface.co/datasets/yerevann/PubChemForLM).

A structured prefix such as

```text
[SAS]2.25[/SAS][SIMILAR]CC(=O)OC1=CC=CC=C1C(=O)O 0.62[/SIMILAR][START_SMILES]
```

asks the model to generate a molecule with the requested synthetic-accessibility
score and similarity. The tags used for optimization must come from the training
format; arbitrary oracle names are not natural-language instructions to these
models.

## Installation

The pinned Conda environment includes the training and optimization dependencies:

```bash
conda env create -f environment.yml
conda activate gemma_env_new
pip install -e .
```

Model inference requires hardware appropriate for the selected checkpoint. The
optimization code also requires RDKit.

## Molecular optimization

The optimization loop is in
[`chemlactica/mol_opt/optimization.py`](chemlactica/mol_opt/optimization.py), and
[`example_run.py`](chemlactica/mol_opt/example_run.py) shows the oracle interface
and configuration flow.

PMO experiments support two explicit information regimes:

- **Task-agnostic:** the objective is a black-box oracle. No task-derived
  structured prefix is added.
- **Task-informed:** public structure of the objective is encoded with the
  model's trained tags, such as `[QED]`, `[FORMULA]`, `[TPSA]`, `[CLOGP]`, and
  `[SIMILAR]`. Unsupported classifier names (`DRD2`, `GSK3B`, and `JNK3`) are not
  inserted as text.

Use the canonical registry rather than maintaining per-run prompt variants:

```python
from chemlactica.mol_opt.optimization import optimize
from chemlactica.mol_opt.pmo_prompts import get_pmo_additional_properties

task_name = "qed"
prompt_mode = "task-informed"  # or "task-agnostic"

optimize(
    model,
    tokenizer,
    oracle,
    config,
    additional_properties=get_pmo_additional_properties(task_name, prompt_mode),
)
```

The Valsartan SMARTS task has no trained SMARTS tag. Its exact SMARTS literal is
therefore kept unchanged as the reference inside `[SIMILAR]`; it is not expanded
into a molecule. The full registry and its rationale are implemented in
[`pmo_prompts.py`](chemlactica/mol_opt/pmo_prompts.py).

### Reproduced PMO results

The current canonical reproduction values below are sums of Top-10 AUC over all
23 Practical Molecular Optimization tasks. Each entry is the mean of five seeds
evaluated over a 10,000-call horizon. Some saturated QED trajectories were stopped
early and flat-filled to the horizon, following the benchmark's stopping
convention.

| Model | Task-agnostic | Task-informed |
| --- | ---: | ---: |
| Chemlactica-125M | 16.944598 | 20.429421 |
| Chemlactica-1.3B | 17.168555 | 20.192229 |
| Chemma-2B | 17.500147 | 20.550791 |

The two columns expose different amounts of task information and should be treated
as separate comparison regimes. The reproductions use the corrected Sitagliptin
oracle and a fixed prompt registry; online oracle calls are counted in the PMO
budget. The public benchmark implementations are maintained in
[ChemLacticaTestSuite](https://github.com/YerevaNN/ChemLacticaTestSuite), while
the canonical task-informed prompt definitions are maintained in this repository.

The current manuscript is available on
[arXiv](https://arxiv.org/abs/2407.18897).

## Training and fine-tuning

Training entry points and model-specific configurations are under
[`chemlactica/`](chemlactica/). These files reflect the original research
workflows and may require adapting paths, distributed settings, and data locations
for a new cluster. The Hugging Face model cards are the recommended starting point
for inference.

## Tests

Run the lightweight unit suite with:

```bash
python confirm_tests.py --run unit
```

The task-informed prompt tests verify coverage of all 23 PMO tasks, exact trained
tag syntax, an empty task-agnostic path, and the Valsartan SMARTS handling.

## Citation and license

Read [_Small Molecule Optimization with Large Language Models_](https://arxiv.org/abs/2407.18897)
for the method and experimental context.
The software in this repository is released under the [MIT License](LICENSE).
