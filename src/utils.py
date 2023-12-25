from transformers import AutoTokenizer
import os


chemlactica_special_start_tokens = [
    "[SYNONYM]",
    "[RELATED]",
    "[SIMILAR]",
    "[PROPERTY]",
    "[SAS]",
    "[WEIGHT]",
    "[TPSA]",
    "[CLOGP]",
    "[QED]",
    "[NUMHDONORS]",
    "[NUMHACCEPTORS]",
    "[NUMHETEROATOMS]",
    "[NUMROTATABLEBONDS]",
    "[NOCOUNT]",
    "[NHOHCOUNT]",
    "[RINGCOUNT]",
    "[HEAVYATOMCOUNT]",
    "[FRACTIONCSP3]",
    "[NUMAROMATICRINGS]",
    "[NUMSATURATEDRINGS]",
    "[NUMAROMATICHETEROCYCLES]",
    "[NUMAROMATICCARBOCYCLES]",
    "[NUMSATURATEDHETEROCYCLES]",
    "[NUMSATURATEDCARBOCYCLES]",
    "[NUMALIPHATICRINGS]",
    "[NUMALIPHATICHETEROCYCLES]",
    "[NUMALIPHATICCARBOCYCLES]",
    "[IUPAC]",
    "[VAR_NAME]",
    "[VAR_DESC]",
    "[VAR_UNIT]",
    "[VAR_VAL]",
    "[ASSAY_NAME]",
    "[ASSAY_DESC]",
]

chemlactica_special_end_tokens = [
    s.replace("[", "[/") for s in chemlactica_special_start_tokens
]


def get_tokenizer_path():
    return "src/tokenizer/ChemLacticaTokenizer66"


def get_tokenizer_special_tokens():
    import json

    with open(os.path.join(get_tokenizer_path(), "special_tokens_map.json"), "r") as _f:
        special_tokens_json = json.load(_f)
    return special_tokens_json["additional_special_tokens"]


chemlactica_special_tokens = (
    chemlactica_special_start_tokens + chemlactica_special_end_tokens
)
chemlactica_special_tokens = {"additional_special_tokens": chemlactica_special_tokens}
chemlactica_special_tokens["pad_token"] = "<pad>"


# chemlactica_special_tokens =dict(zip(chemlactica_special_tokens, chemlactica_special_tokens))
# chemlactica_special_tokens["pad_token"] = "<pad>"


def get_tokenizer():
    if getattr(get_tokenizer, "first_call", True):
        setattr(get_tokenizer, "tokenizer", create_tokenizer())
        setattr(get_tokenizer, "first_call", False)
        print(f"Process {os.getpid()} created a tokenizer")

    return get_tokenizer.tokenizer


def create_tokenizer():
    # auth_token = os.environ["HF_TOKEN"]
    tok = AutoTokenizer.from_pretrained(
        # f"facebook/galactica-125m"
        # "src/tokenizer/ChemLacticaTokenizer"
        # "src/tokenizer/galactica-125m"
        get_tokenizer_path()
    )
    bos_token = "<s>"
    bos_token_id = 1
    # pad_token = "<pad>"
    # pad_token_id = 1
    eos_token = "</s>"
    eos_token_id = 2
    tok.bos_token = bos_token
    tok.bos_token_id = bos_token_id
    # tok.pad_token = pad_token
    # tok.pad_token_id = pad_token_id
    tok.eos_token = eos_token
    tok.eos_token_id = eos_token_id
    # tok.add_special_tokens(chemlactica_special_tokens)
    return tok


class ForcedStop(RuntimeError):
    def __init__(self, message="Forced stop occurred"):
        super().__init__(message)


def signal_handler(sig, frame):
    raise ForcedStop


if __name__ == "__main__":
    import glob

    from config.create_train_config import model_train_configs
    from datasets import load_dataset
    from dataset_utils import process_dataset

    train_config = model_train_configs["llama2"]
    tokenizer = get_tokenizer("meta-llama/Llama-2-7b-hf")

    # CustomTokenizer.set_model_size("125m")
    # tokenizer = CustomTokenizer.get_instance()
    tokenizer.save_pretrained("tokenizer/chemllama2-tokenizer")
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

    # sample = next(iter(processed_dataset["train"]))
    prompt = "[START_SMILES] CCCCN [END_SMILES][CLOGP 0.00][SAS 123][QED]"
    # print(tokenizer.decode(sample["input_ids"]))
    # print(*sample["input_ids"].numpy())
    # print(len(tokenizer.decode(sample["input_ids"])), len(sample["input_ids"].numpy()))
    # print('*'*20)
    # print(prompt)
    # print(tokenizer.encode(prompt))
