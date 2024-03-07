from custom_trainer import CustomTrainer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from chemlactica.eval_metrics import compute_metrics, preprocess_logits_for_metrics
from utils.dataset_utils import sft_formatting_prompts_func
from utils.utils import get_tokenizer


def get_trainer(train_type, model, dataset, training_args, evaluate_only, slurm_eval):
    if train_type == "pretrain":
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=dataset["train"] if not evaluate_only else None,
            eval_dataset=dataset["validation"]["validation"]
            if not evaluate_only or slurm_eval
            else None,
            # optimizers=[optimizer, lr_scheduler],
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

    elif train_type == "sft":
        tokenizer = get_tokenizer()
        response_template = "[PROPERTY]activity "
        collator = DataCollatorForCompletionOnlyLM(
            response_template, tokenizer=tokenizer
        )
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            formatting_func=sft_formatting_prompts_func,
            args=training_args,
            packing=False,
            tokenizer=tokenizer,
            max_seq_length=512,
            data_collator=collator,
            neftune_noise_alpha=5,
        )
    return trainer
