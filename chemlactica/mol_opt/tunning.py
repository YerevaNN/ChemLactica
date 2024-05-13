from transformers.trainer_callback import TrainerControl, TrainerState
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments, get_polynomial_decay_schedule_with_warmup, TrainerCallback
from torch.optim.lr_scheduler import ConstantLR
import torch
import math


class CustomSFTTrainer(SFTTrainer):

    def __init__(self, *args, patience, toll, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = patience
        self.initial_pat = patience
        self.toll = toll
        self.best_loss = math.inf

    def log(self, logs) -> None:
        if logs.get("loss"):
            curr_loss = logs["loss"]
            if curr_loss > self.best_loss - self.toll:
                self.patience -= 1
                print(f"loss did not improve, patience {self.patience}")
            else:
                print("loss improved")
                self.best_loss = curr_loss
                self.patience = self.initial_pat
            if self.patience == 0:
                print("The loss does not improve, stop training.")
                self.control.should_training_stop = True
        return super().log(logs)


def supervised_fine_tune(
        model, tokenizer,
        train_dataset, config
    ):
    model.train()
    training_args = TrainingArguments(
        output_dir=config["checkpoints_dir"],
        per_device_train_batch_size=config["train_batch_size"],
        max_grad_norm=config["global_gradient_norm"],
        num_train_epochs=config["num_train_epochs"],
        evaluation_strategy="no",
        dataloader_drop_last=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=config["dataloader_num_workers"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        logging_steps=1,
        metric_for_best_model="loss",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["max_learning_rate"],
        betas=[config["adam_beta1"], config["adam_beta2"]],
        weight_decay=config["weight_decay"],
    )
    lr_scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=config["num_train_epochs"] * (len(train_dataset) // config["train_batch_size"] + 1),
        lr_end=0.999 * config["max_learning_rate"],
        power=1.0,
    )
    collator = DataCollatorForCompletionOnlyLM(
        config["response_template"], tokenizer=tokenizer
    )
    trainer = CustomSFTTrainer(
        model=model,
        train_dataset=train_dataset,
        formatting_func=config["formatting_func"],
        args=training_args,
        packing=config["packing"],
        tokenizer=tokenizer,
        max_seq_length=config["max_seq_length"],
        # data_collator=collator,
        optimizers=[optimizer, lr_scheduler],
        patience=2,
        toll=0.0001
    )
    trainer.train()
