import shutil
from transformers import Trainer
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP


class CustomTrainer(Trainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        if shutil.disk_usage('/').free > 3 * 1024 ** 3: 
            super()._save_checkpoint(model, trial, metrics=None)
        else:
            print("**disk is full didn't save**")

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """
            This code is added because we had a failure when resuming training.
            Basically, we load the model with fsdp when the model is not fsdp wrapped.
            In the future versions transformers this issue is handled, by adding an extra check,
            but not in 4.31.0 version. So this is our manual check addition to solve the problem.
        """
        if type(self.model) != FSDP: return
        return super()._load_from_checkpoint(resume_from_checkpoint, model)

    # def save_model(
    #     self, output_dir: Optional[str] = None, _internal_call: bool = False
    # ):
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
        # self.model = BetterTransformer.reverse(self.model)
        # super().save_model(output_dir, _internal_call)
        # self.model = BetterTransformer.transform(self.model)
        # self.model = BetterTransformer.reverse(self.model)
        # self.accelerator.save_state(output_dir)
        # self.model = BetterTransformer.transform(self.model)
