from transformers.trainer_callback import TrainerControl, TrainerState, TrainerCallback
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments, get_polynomial_decay_schedule_with_warmup, EarlyStoppingCallback
from torch.optim.lr_scheduler import ConstantLR
import torch
import math
import time
from chemlactica.mol_opt.utils import generate_random_number


class CustomEarlyStopCallback(TrainerCallback):

    def __init__(self, early_stopping_patience: int, early_stopping_threshold: float) -> None:
        super().__init__()
        self.best_valid_loss = math.inf
        self.early_stopping_patience = early_stopping_patience
        self.current_patiance = 0
        self.early_stopping_threshold = early_stopping_threshold

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.best_valid_loss = math.inf
        self.current_patiance = 0
        return super().on_train_begin(args, state, control, **kwargs)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        if metrics["eval_loss"] >= self.best_valid_loss - self.early_stopping_threshold:
            self.current_patiance += 1
        else:
            self.current_patiance = 0
            self.best_valid_loss = metrics["eval_loss"]
        print(f"Early Stopping patiance: {self.current_patiance}/{self.early_stopping_patience}")
        if self.current_patiance >= self.early_stopping_patience:
            control.should_training_stop = True
        return super().on_evaluate(args, state, control, **kwargs)


# class CustomSFTTrainer(SFTTrainer):
# 
    # def __init__(self, *args, patience, toll, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.patience = patience
    #     self.initial_pat = patience
    #     self.toll = toll
    #     self.best_loss = math.inf

    # def log(self, logs) -> None:
    #     if logs.get("loss"):
    #         curr_loss = logs["loss"]
    #         if curr_loss > self.best_loss - self.toll:
    #             self.patience -= 1
    #             print(f"loss did not improve, patience {self.patience}")
    #         else:
    #             print("loss improved")
    #             self.best_loss = curr_loss
    #             self.patience = self.initial_pat
    #         if self.patience == 0:
    #             print("The loss does not improve, stop training.")
    #             self.control.should_training_stop = True
    #     return super().log(logs)


def get_training_arguments(config):
    checkpoints_dir = config["checkpoints_dir"] + "_" + str(time.time())
    return TrainingArguments(
        output_dir=checkpoints_dir,
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["train_batch_size"],
        max_grad_norm=config["global_gradient_norm"],
        num_train_epochs=config["num_train_epochs"],
        evaluation_strategy="epoch",
        # save_strategy="epoch",
        dataloader_drop_last=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=config["dataloader_num_workers"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        logging_steps=1,
        metric_for_best_model="loss",
        # load_best_model_at_end=True,
        # save_total_limit=1
    )


def get_optimizer_and_lr_scheduler(model, config, max_train_steps):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["max_learning_rate"],
        betas=[config["adam_beta1"], config["adam_beta2"]],
        weight_decay=config["weight_decay"],
    )
    lr_scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=max_train_steps,
        lr_end=0,
        power=1.0,
    )
    return optimizer, lr_scheduler
