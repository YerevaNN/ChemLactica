import torch
import multiprocessing
from datasets.iterable_dataset import IterableDataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from contextlib import nullcontext
from transformers import Trainer, TrainingArguments
from chemlactica.jsonl_dataset import samples_generator
from chemlactica.utils.dataset_utils import process_dataset
import glob
from accelerate import PartialState

distributed_state = PartialState()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def find_all_linear_names(model):
    cls = (
        torch.nn.Linear
    )  # if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    # if 'embed_tokens' in lora_module_names:
    #     lora_module_names.remove('embed_tokens')
    return list(lora_module_names)


if __name__ == "__main__":
    # accelerator = Accelerator()
    model_id = "google/gemma-2b"

    config = AutoConfig.from_pretrained(model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_config(
        config, attn_implementation="flash_attention_2"
    )
    linear_names = find_all_linear_names(model)

    args = TrainingArguments(
        lr_scheduler_type="linear",
        output_dir="./test-galore",
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,
        max_steps=100,
        save_steps=10,
        bf16=True,
        # gradient_accumulation_steps = 43,
        per_device_train_batch_size=2,
        optim="adamw_torch",
        tf32=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=1,
        # ddp_find_unused_parameters = False,
        # gradient_checkpointing = True,
        # optim_args="rank=64, update_proj_gap=100, scale=0.10",
        # optim_target_modules=linear_names,
        # torch_compile = True,
    )

    training_data_dir = (
        "/mnt/sxtn/rdkit_computed_rel+form/train_rdkit_computed_rel+form"
    )
    training_data_files = glob.glob(training_data_dir + "/*.jsonl")
    print(training_data_files)
    with (
        multiprocessing.Manager()
        if distributed_state.is_main_process
        else nullcontext()
    ) as manager:
        shared_jsonl_files = None
        if distributed_state.is_main_process:
            shared_jsonl_files = manager.dict()
        train_dataset = IterableDataset.from_generator(
            samples_generator,
            gen_kwargs={
                "files": training_data_files,
                "shared_jsonl_files": shared_jsonl_files,
            },
        )
        train_config = {
            "tokenizer_path": "/mnt/sxtn/gemma_tok/",
            "block_size": 2048,
        }

        train_dataset = process_dataset(
            dataset=train_dataset,
            train_config=train_config,
            tokenizer=tokenizer,
            process_batch_sizes=(200, 200),
            is_eval=False,
            assay=False,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            # tokenizer = tokenizer,
            # dataset_text_field='text',
            # max_seq_length=512,
        )

        trainer.train()
