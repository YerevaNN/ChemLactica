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

chemlactica_special_tokens = (
    chemlactica_special_start_tokens + chemlactica_special_end_tokens
)


def get_tokenizer(tokenizer_id):
    if getattr(get_tokenizer, "first_call", True):
        setattr(get_tokenizer, "tokenizer", create_tokenizer(tokenizer_id))
        setattr(get_tokenizer, "first_call", False)
        print(f"Process {os.getpid()} created a tokenizer")

    return get_tokenizer.tokenizer


def create_tokenizer(tokenizer_id):
    tok = AutoTokenizer.from_pretrained(
        # f"facebook/galactica-125m"
        tokenizer_id
        # "src/tokenizer/ChemLacticaTokenizer"
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
    #     chemlactica_special_tokens
    # )
    return tok


if __name__ == "__main__":
    import glob

    from config.create_train_config import model_train_configs
    from datasets import load_dataset
    from dataset_utils import process_dataset

    train_config = model_train_configs["mistral7b"]
    tokenizer = get_tokenizer("mistralai/Mistral-7B-v0.1")

    # CustomTokenizer.set_model_size("125m")
    # tokenizer = CustomTokenizer.get_instance()
    tokenizer.save_pretrained("Mistral-7B-v0.1Tokenizer")
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
