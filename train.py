import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import aim_utils

model_checkpoint = "facebook/galactica-125m"

model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def process_str(str):
    str["text"] = str["text"].replace("\\", "")
    return str


dataset = load_dataset(
    "text", data_files="./523129_start.jsonl", split="train", streaming=True
)
dataset = dataset.map(process_str)


def tokenize_function(examples):
    return tokenizer(examples["text"])


tokenized_datasets = dataset.map(tokenize_function, batched=True)

model_name = model_checkpoint.split("/")[-1]

training_args = TrainingArguments(
    output_dir=f"{model_name}-finetuned-pubchem",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    max_steps=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    # eval_dataset=lm_datasets["validation"],
    # callbacks=[aim_utils.AimTrackerCallback]
)

trainer.train()
