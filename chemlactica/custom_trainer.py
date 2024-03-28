import submitit
from typing import Any, Dict
import os
from torch._tensor import Tensor
from torch.nn.modules import Module

# from torch.distributed.fsdp.fully_sharded_data_parallel import (
#     FullyShardedDataParallel as FSDP,
# )
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

# from transformers.utils import is_torch_tpuc _available
from trl import IterativeSFTTrainer
from chemlactica.utils.utils import get_tokenizer
from dataclasses import dataclass, field


# if is_torch_tpu_available(check_device=False):
#     import torch_xla.core.xla_model as xm


@dataclass
class CustomArguments(TrainingArguments):
    slurm_eval: bool = field(
        default=False, metadata={"help": "Whether to run eval via slurm job."}
    )
    command: str = field(default=None)
    experiment_name: str = field(default=None)
    tokenizer_path: str = field(
        default="/auto/home/menuab/code/ChemLactica/chemlactica/tokenizer/ChemLacticaTokenizer66"
    )
    # train_config: dict = field(default=None)


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        # the number of samples to print when the training begins, for debugging purposes
        self.num_samples_to_print = 5
        self.tokenizer_path = kwargs["args"].tokenizer_path
        super().__init__(*args, **kwargs)

    def training_step(self, model: Module, inputs: Dict[str, Tensor | Any]) -> Tensor:
        if self.num_samples_to_print:
            tokenizer = get_tokenizer(self.tokenizer_path)
            for i in range(min(inputs["input_ids"].size(0), self.num_samples_to_print)):
                print(f"Sample {i + 1}:", tokenizer.decode(inputs["input_ids"][i]))
            self.num_samples_to_print = None
        return super().training_step(model, inputs)

    def _build_slurm_eval_command(self, train_command, trial):
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        search_string = "--from_pretrained"
        train_command[train_command(search_string) + 1] = output_dir
        train_command.append("--evaluate_only")  # Drastically changes script behaviour
        eval_command = train_command
        return eval_command

    def _submitit_slurm_eval_launch(self, eval_function):
        slurm_nice_setting = 1000
        executor = submitit.AutoExecutor(folder="your_experiment_folder")
        cpus_per_task = self.args.dataloader_num_workers
        mem_gb = 96
        executor.update_parameters(
            name=self.args.experiment_name + "eval",
            timeout_min=120,
            cpus_per_task=cpus_per_task,
            gpus_per_node=1,
            mem_gb=mem_gb,
            slurm_srun_args=[f"--nice={str(slurm_nice_setting)}"],
        )
        exit()
        job = executor.submit(eval_function)  # noqa

    # def _maybe_log_save_evaluate(
    #     self, tr_loss, model, trial, epoch, ignore_keys_for_eval
    # ):
    #     # this method is being overwritten because it currently
    #     # runs evaluation prior to saving.
    #     # For offling evaluation this doesn't work.
    #     if (
    #         self.control.should_log
    #         and self.state.global_step > self._globalstep_last_logged
    #     ):
    #         if is_torch_tpu_available():
    #             xm.mark_step()

    #         logs: Dict[str, float] = {}

    #         # all_gather + mean() to get average loss over all processes
    #         tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

    #         # reset tr_loss to zero
    #         tr_loss -= tr_loss

    #         logs["loss"] = round(
    #             tr_loss_scalar
    #             / (self.state.global_step - self._globalstep_last_logged),
    #             4,
    #         )
    #         logs["learning_rate"] = self._get_learning_rate()

    #         self._total_loss_scalar += tr_loss_scalar
    #         self._globalstep_last_logged = self.state.global_step
    #         self.store_flos()

    #         self.log(logs)

    #     metrics = None

    #     if self.control.should_save:
    #         self._save_checkpoint(model, trial, metrics=metrics)
    #         self.control = self.callback_handler.on_save(
    #             self.args, self.state, self.control
    #         )

    #     if self.control.should_evaluate:
    #         if self.args.slurm_eval:
    #             # BUILD SLURM EVALUATE COMMAND
    #             eval_command = self._build_slurm_eval_command(self.args.command, trial)
    #             print("-----------------------------------------------")
    #             print("starting slurm eval with command:", eval_command)
    #             eval_function = submitit.helpers.CommandFunction(
    #                 eval_command, verbose=True, cwd=os.getcwd()
    #             )
    #             self._submitit_slurm_eval_launch(eval_function)

    #         else:
    #             metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
    #             self._report_to_hp_search(trial, self.state.global_step, metrics)
    #         # Run delayed LR scheduler now that metrics are populated
    #         if isinstance(
    #             self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
    #         ):
    #             metric_to_check = self.args.metric_for_best_model
    #             if not metric_to_check.startswith("eval_"):
    #                 metric_to_check = f"eval_{metric_to_check}"
    #             self.lr_scheduler.step(metrics[metric_to_check])


class CustomIterativeSFTTrainer(IterativeSFTTrainer):
    def __init__(self, *args, **kwargs):
        # the number of samples to print when the training begins, for debugging purposes
        self.num_samples_to_print = 5
        super().__init__(*args, **kwargs)

    def training_step(self, model: Module, inputs: Dict[str, Tensor | Any]) -> Tensor:
        if self.num_samples_to_print:
            # tokeinzer = get_tokenizer()
            for i in range(min(inputs["input_ids"].size(0), self.num_samples_to_print)):
                print(f"Sample {i + 1}:", self.tokeinzer.decode(inputs["input_ids"][i]))
            self.num_samples_to_print = None
        return super().training_step(model, inputs)
