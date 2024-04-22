from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments, get_polynomial_decay_schedule_with_warmup
from torch.optim.lr_scheduler import ConstantLR
import torch


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
        logging_steps=1
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
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        formatting_func=config["formatting_func"],
        args=training_args,
        packing=config["packing"],
        tokenizer=tokenizer,
        max_seq_length=config["max_seq_length"],
        data_collator=collator,
        optimizers=[optimizer, lr_scheduler]
    )
    trainer.train()
