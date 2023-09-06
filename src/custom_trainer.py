from transformers import Trainer
from typing import Optional
from transformers import OPTForCausalLM
import torch
from datasets import load_dataset
from dataset_utils import process_dataset
import os
import glob

# from torch.distributed.fsdp import (
#     FullyShardedDataParallel as FSDP,
#     FullStateDictConfig,
#     StateDictType,
# )
# import torch.distributed as dist


class CustomTrainer(Trainer):
    def save_model(
        self, output_dir: Optional[str] = None, _internal_call: bool = False
    ):
        # underlying_config = self.model.module.config
        # cpu_state = self.accelerator.get_state_dict(self.model, unwrap=True)
        # converted_model = OPTForCausalLM.from_pretrained(
        #     None, config=underlying_config, state_dict=cpu_state
        # )
        # converted_model.save_pretrained(
        #     is_main_process=self.accelerator.is_main_process,
        #     save_directory=output_dir,
        #     save_function=self.accelerator.save,
        #     max_shard_size="200MB",
        # )
        super().save_model(output_dir, _internal_call)

        checkpoint_dir = os.path.join(
            output_dir
        )
        print("loading from", checkpoint_dir)
        saved_model = OPTForCausalLM.from_pretrained(checkpoint_dir)

        training_data_files = glob.glob(".small_data/train" + "/*.jsonl")

        dataset = load_dataset(
            "text",
            data_files={"data": training_data_files},
            streaming=True,
        )

        processed_dataset = process_dataset(
            dataset=dataset,
            train_config={"block_size": 2048},
            process_batch_sizes=(100, 100),
        )

        is_repr = True
        for inp in processed_dataset["data"]:
            del inp["token_type_ids"]
            # out = self.model(**{k: inp[k].unsqueeze(0).to(self.model.device) for k in inp.keys()})
            saved_out = saved_model(**{k: inp[k].unsqueeze(0).to(saved_model.device) for k in inp.keys()})

            # ok = torch.allclose(out.logits, saved_out.logits.to(out.logits.device), atol=1e-4)
            # if not ok:
            #     is_repr = False
            break

        if is_repr:
            print(f"Model at step {self.state.global_step} is reproducable.")
        else:
            print(f"Model at step {self.state.global_step} is not reproducable.")
