import shutil
from transformers import Trainer
from typing import Optional
from transformers import OPTForCausalLM


class CustomTrainer(Trainer):
    # def __init__(*args, **kwargs):
    #     super().__init__(*args, **kwargs)

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
            max_shard_size="500MB",
        )