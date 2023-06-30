from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
import datasets
from eval_metrics import compute_metrics
from aim.hugging_face import AimCallback


model_checkpoint = "facebook/galactica-125m"
block_size = 2000
batch_size = 1


def process_str(example):
    example["text"] = example["text"].replace(r"\"", " ")
    return example


def tokenize_function(examples):
    result = tokenizer(examples["text"])
    for k, v in result.items():
        if len(v) <= block_size:
            result[k] = v + ((block_size - len(v)) * tokenizer.encode("<pad>"))
        else:
            result[k] = v[:block_size]

    result["labels"] = result["input_ids"].copy()  # TODO: make sure this is correct
    return result


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    dataset = datasets.load_dataset(
        "text",
        data_files={"train": "./train.jsonl", "validation": "./evaluation.jsonl"},
        streaming=True,
    )

    dataset = dataset.map(process_str)
    tokenized_datasets = dataset.map(tokenize_function, remove_columns=["text"])

    lm_datasets = tokenized_datasets

    # lm_datasets = tokenized_datasets.map(group_texts, batched=True, batch_size=batch_size)

    # for sample in lm_datasets["train"]:
    #     pass
    #     print(len(sample["input_ids"]))
    #     print(len(sample["token_type_ids"]))
    #     print(len(sample["attention_mask"]))

    # print(tokenizer(" hello _ lijasdklhj lijasldkj"))

    aim_callback = AimCallback(
        repo="/mnt/sxtn/chem/ChemLactica/metadata", experiment="your_experiment_name"
    )

    # training_args = TrainingArguments(
    #     output_dir=f"{model_checkpoint.split('/')[-1]}-finetuned-pubchem",
    #     evaluation_strategy="steps",
    #     eval_steps=1,
    #     learning_rate=2e-5,
    #     weight_decay=0.01,
    #     max_steps=10,
    #     per_device_train_batch_size=batch_size,
    #     per_device_eval_batch_size=batch_size
    # )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     compute_metrics=compute_metrics,
    #     train_dataset=lm_datasets["train"],
    #     eval_dataset=lm_datasets["validation"],
    #     # callbacks=[aim_callback]
    # )

    # trainer.train()
