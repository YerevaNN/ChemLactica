import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
import transformers
import torch.nn as nn
import os


class LinearFloat32(nn.Linear):
    def forward(self, _input) -> torch.Tensor:
        return super().forward(_input).to(torch.float32)


def custom_opt_init(func):
    def inner_func(self, config, *args, **kwargs):
        func(self, config, *args, **kwargs)
        self.lm_head = LinearFloat32(
            config.word_embed_proj_dim, config.vocab_size, bias=False
        )

    return inner_func


OPTForCausalLM.__init__ = custom_opt_init(OPTForCausalLM.__init__)


def load_model(from_pretrained: str, train_config):
    if from_pretrained == "small_opt":
        return transformers.OPTForCausalLM(
            transformers.OPTConfig(
                vocab_size=train_config["vocab_size"],
                hidden_size=train_config["hidden_size"],
                num_hidden_layers=train_config["num_hidden_layers"],
                ffn_dim=train_config["ffn_dim"],
                max_position_embeddings=train_config["max_position_embeddings"],
                num_attention_heads=train_config["num_attention_heads"],
                word_embed_proj_dim=train_config["word_embed_proj_dim"],
            )
        )
    return AutoModelForCausalLM.from_pretrained(from_pretrained)


# class CustomTokenizer:
#     __instance = None
#     precomuted_ids = {}
#     bos_token = "<s>"
#     bos_token_id = 0
#     pad_token = "<pad>"
#     pad_token_id = 1
#     eos_token = "</s>"
#     eos_token_id = 2
#     qed_token = "[QEDĠ"
chemlactica_special_tokens = [
    "[SYNONYM ",
    "[RELATED ",
    "[SIMILAR ",
    "[EXPERIMENTAL ",
    "[SAS ",
    "[WEIGHT ",
    "[TPSA ",
    "[CLOGP ",
    "[QED ",
    "[NUMHDONORS ",
    "[NUMHACCEPTORS ",
    "[NUMHETEROATOMS ",
    "[NUMROTATABLEBONDS ",
    "[NOCOUNT ",
    "[NHOHCOUNT ",
    "[RINGCOUNT ",
    "[HEAVYATOMCOUNT ",
    "[FRACTIONCSP3 ",
    "[NUMAROMATICRINGS ",
    "[NUMSATURATEDRINGS ",
    "[NUMAROMATICHETEROCYCLES ",
    "[NUMAROMATICCARBOCYCLES ",
    "[NUMSATURATEDHETEROCYCLES ",
    "[NUMSATURATEDCARBOCYCLES ",
    "[NUMALIPHATICRINGS ",
    "[NUMALIPHATICHETEROCYCLES ",
    "[NUMALIPHATICCARBOCYCLES ",
    "[IUPAC ",
]
#     model_size = None

#     @staticmethod
#     def set_model_size(model_size):
#         CustomTokenizer.model_size = model_size

#     @staticmethod
#     def get_instance():
#         if CustomTokenizer.__instance is None:
#             CustomTokenizer.__instance = CustomTokenizer.new_instance()
#             CustomTokenizer.precomuted_ids = {
#                 v: torch.tensor(CustomTokenizer.__instance.encode(v), dtype=torch.int32)
#                 for v in ["[START_SMILES]", "[END_SMILES]", "[", "]", "<pad>"]
#             }
#         return CustomTokenizer.__instance

#     @staticmethod
#     def new_instance():
#         tok = AutoTokenizer.from_pretrained(    
#             f"facebook/galactica-{CustomTokenizer.model_size}"
#             # 'src/tokenizer/ChemLacticaTokenizer'
#             # 'src/tokenizer/GalacticaTokenizer'
#         )
#         tok.bos_token = CustomTokenizer.bos_token
#         tok.bos_token_id = CustomTokenizer.bos_token_id
#         tok.pad_token = CustomTokenizer.pad_token
#         tok.pad_token_id = CustomTokenizer.pad_token_id
#         tok.eos_token = CustomTokenizer.eos_token
#         tok.eos_token_id = CustomTokenizer.eos_token_id
#         # tok.add_tokens(
#         #     CustomTokenizer.chemlactica_special_tokens
#         # )
#         return tok


def get_tokenizer():
    if getattr(get_tokenizer, "first_call", True):
        setattr(get_tokenizer, "tokenizer", create_tokenizer())
        setattr(get_tokenizer, "first_call", False)
        print(f"Process {os.getpid()} created a tokenizer")
        def on_delete():
            print(f"Process {os.getpid()} terminated")
        get_tokenizer.__del__ = on_delete

    return get_tokenizer.tokenizer


def create_tokenizer():
    tok = AutoTokenizer.from_pretrained(    
        # f"facebook/galactica-125m"
        'src/tokenizer/ChemLacticaTokenizer'
        # 'src/tokenizer/GalacticaTokenizer'
        # "src/tokenizer/galactica-125m"
    )
    bos_token = "<s>"
    bos_token_id = 0
    pad_token = "<pad>"
    pad_token_id = 1
    eos_token = "</s>"
    eos_token_id = 2
    tok.bos_token = bos_token
    tok.bos_token_id = bos_token_id
    tok.pad_token = pad_token
    tok.pad_token_id = pad_token_id
    tok.eos_token = eos_token
    tok.eos_token_id = eos_token_id
    # tok.add_tokens(
    #     CustomTokenizer.chemlactica_special_tokens
    # )
    return tok


class ProgressBar(tqdm.tqdm):
    __instance = None
    __total = None

    def __init__(self, total=None, *args, **kwargs):
        new_total = ProgressBar.__total if total is None else total
        super().__init__(total=new_total, *args, **kwargs)
        if ProgressBar.__instance is not None:
            raise Exception(f"There can only be one instance of {__class__.__name__}")

        ProgressBar.__instance = self
        ProgressBar.__total = new_total

    @staticmethod
    def set_total(total):
        ProgressBar.__total = total

    @staticmethod
    def delete_instance():
        ProgressBar.__instance = None

    @staticmethod
    def get_instance():
        return ProgressBar.__instance


if __name__ == "__main__":
    import glob

    # from utils import CustomTokenizer
    from config.create_train_config import model_train_configs
    from datasets import load_dataset
    from dataset_utils import process_dataset

    train_config = model_train_configs["125m"]
    train_config["block_size"] = 2048

    # CustomTokenizer.set_model_size("125m")
    # tokenizer = CustomTokenizer.get_instance()
    # tokenizer.save_pretrained("ChemLacticaTokenizer")
    training_data_dir = ".small_data/valid"

    # training_data_files = glob.glob(training_data_dir + "/xae_shuf.jsonl")
    training_data_files = glob.glob(training_data_dir + "/*.jsonl")

    dataset = load_dataset(
        "text",
        data_files={"train": training_data_files, "validation": training_data_files},
        streaming=True,
    )

    processed_dataset = process_dataset(
        dataset=dataset, train_config=train_config, process_batch_sizes=(50, 50)
    )

    sample = next(iter(processed_dataset["train"]))
    prompt = "[START_SMILES] CCCCN [END_SMILES][CLOGP 0.00][SAS 123][QED]"
    # print(tokenizer.decode(sample["input_ids"]))
    # print(*sample["input_ids"].numpy())
    # print(len(tokenizer.decode(sample["input_ids"])), len(sample["input_ids"].numpy()))
    # print('*'*20)
    # print(prompt)
    # print(tokenizer.encode(prompt))
