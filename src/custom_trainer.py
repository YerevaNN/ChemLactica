import shutil
from transformers import Trainer
from typing import Optional
from transformers import OPTForCausalLM
import accelerate
import os
import json


# set this function to empty function to not skip any steps manually
accelerate.skip_first_batches = lambda x: None


class CustomTrainer(Trainer):

    def __init__(self, train_jsonl_datasets, *args, **kwargs):
        self._train_jsonl_datasets = train_jsonl_datasets
        super().__init__(*args, **kwargs)

    def _save_checkpoint(self, model, trial, metrics=None):
        if shutil.disk_usage('/').free > 3 * 1024 ** 3: 
            super()._save_checkpoint(model, trial, metrics=None)
        else:
            print("**disk is full didn't save**")

    def save_model(
        self, output_dir: Optional[str] = None, _internal_call: bool = False
    ):
        underlying_config = self.model.module.config
        cpu_state = self.accelerator.get_state_dict(self.model, unwrap=True)
        converted_model = OPTForCausalLM.from_pretrained(
            None, config=underlying_config, state_dict=cpu_state
        )
        converted_model.save_pretrained(
            is_main_process=self.accelerator.is_main_process,
            save_directory=output_dir,
            save_function=self.accelerator.save,
            max_shard_size="200MB",
        )

        if self.accelerator.is_main_process:
            # save file offsets
            jsonl_generators_dict = {}
            for file_name, ds in self._train_jsonl_datasets.items():
                jsonl_generators_dict[file_name] = ds.get_read_position()
            with open(os.path.join(output_dir, "jsonl_generators.json"), "w") as _f:
                json.dump(jsonl_generators_dict, _f)