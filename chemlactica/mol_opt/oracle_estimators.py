from typing import List
import time
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoModelForCausalLM, PreTrainedModel, AutoModel, AutoConfig, AutoTokenizer
import torch
import torch.nn as nn
import numpy as np
from chemlactica.mol_opt.utils import MoleculeEntry
from sklearn.linear_model import Ridge


def find_second_eos_token_indices(sequences, eos_token_id):
    return torch.where(sequences[:, 1:] == eos_token_id)


def init_linear_layer(layer, emb_length):
    torch.nn.init.normal_(
        layer.weight,
        mean=0.0, std=1 / np.sqrt(emb_length + 1)
    )
    torch.nn.init.constant_(layer.bias, val=0.0)
    return layer


class ScalarHeadLM(PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.lm_backbone = AutoModel.from_pretrained(
            config._name_or_path,
            config=config
        )
        self.scalar_head = nn.Linear(config.hidden_size, 1)
        init_linear_layer(self.scalar_head)

    def forward(self, **kwargs):
        output = self.lm_backbone(**kwargs)
        return self.scalar_head(output.last_hidden_state)


class LinearFingerprintModel:

    def __init__(self):
        self.emb_length = 2048
        self.linear = Ridge()
        self.all_entries = []
        self.is_fit = False

    def __call__(self, mol_entries: List[MoleculeEntry]):
        mol_embs = np.array([entry.fingerprint for entry in mol_entries])
        return self.linear.predict(mol_embs)
    
    def fit(self, mol_entries: List[MoleculeEntry]):
        self.is_fit = True
        start_time = time.time()
        self.all_entries.extend(mol_entries)
        mol_embs = np.array([entry.fingerprint for entry in self.all_entries])
        scores = np.array([entry.score for entry in self.all_entries])
        self.linear.fit(mol_embs, scores)
        print(f"Fit time {time.time() - start_time:.4f}s")


class ScalarOracleApproximator:

    def __init__(self, config, tokenizer):
        self.scalar_head_lm = ScalarHeadLM(config)
        self.tokenizer = tokenizer

    def __call__(self, mol_entries):
        prompts = [f"</s>[START_SMILES]{e.smiles}[END_SMILES]</s>" for e in mol_entries]
        data = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.scalar_head_lm.device)
        del data["token_type_ids"]
        outputs = self.scalar_head_lm(
            **data
        )
        print(outputs)


class SFTOracleApproximator:
    
    def __init__(self, config, tokenizer, device):
        self.ml = AutoModelForCausalLM.from_pretrained(
            config._name_or_path,
            config=config
        ).to(device)
        self.tokenizer = tokenizer


if __name__ == "__main__":
    config = AutoConfig.from_pretrained("/nfs/dgx/raid/chem/checkpoints/facebook/galactica-125m/26d322857a184fcbafda5d4a/checkpoint-118784")
    tokenizer = AutoTokenizer.from_pretrained("chemlactica/tokenizer/ChemLacticaTokenizer66", padding_side="left")
    scalar_oracle_approx = ScalarOracleApproximator(config, tokenizer)

    mol_entries = [MoleculeEntry("CCC" + i * "C") for i in range(10)]
    scalar_oracle_approx(mol_entries)