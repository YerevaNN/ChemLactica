from transformers import Trainer
from typing import Optional
from transformers import OPTForCausalLM

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)


class CustomTrainer(Trainer):
    def save_model(
        self, output_dir: Optional[str] = None, _internal_call: bool = False
    ):
        underlying_config = self.model.module.config
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

        with FSDP.state_dict_type(
            self.model, StateDictType.FULL_STATE_DICT, save_policy
        ):
            cpu_state = self.model.state_dict()
            if self.accelerator.is_main_process:
                converted_model = OPTForCausalLM(underlying_config)
                converted_model.save_pretrained(
                    save_directory=output_dir,
                    max_shard_size="1GB",
                    save_function=self.accelerator.save,
                    state_dict=cpu_state,
                )
        self.accelerator.wait_for_everyone()
