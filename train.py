import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from eval_metrics import compute_metrics
from aim.hugging_face import AimCallback


def process_str(str):
    str["text"] = str["text"].replace("\\", "")
    return str


def tokenize_function(examples):
    return tokenizer(examples["text"])


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def tokenize_function(examples):
    return tokenizer(examples["text"])


if __name__ == "__main__":
    model_checkpoint = "facebook/galactica-125m"
    block_size = 128

    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    dataset = load_dataset(
        "text",
        data_files={"train": "./train.jsonl", "validation": "./evaluation.jsonl"},
        streaming=True,
    )
    dataset = dataset.map(process_str)
    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    lm_datasets = tokenized_datasets.map(group_texts, batched=True, batch_size=1000)

    aim_callback = AimCallback(repo=".", experiment="your_experiment_name")

    training_args = TrainingArguments(
        output_dir=f"{model_checkpoint.split('/')[-1]}-finetuned-pubchem",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        max_steps=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        callbacks=[aim_callback],
    )

    trainer.train()
