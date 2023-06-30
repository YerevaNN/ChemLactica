import evaluate
import transformers
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
import numpy as np

# perplexity_metric = evaluate.load("perplexity", module_type="metric")

galactica_model_checkpoint = "facebook/galactica-125m"
galactica_tokenizer = AutoTokenizer.from_pretrained(galactica_model_checkpoint)


@torch.no_grad()
def perplexity(logits: torch.Tensor, labels: torch.Tensor):
    comp_perp = torch.exp(
        F.cross_entropy(
            logits.view(logits.size(0) * logits.size(1), logits.size(2)),
            labels.view(labels.size(0) * labels.size(1)),
            reduction="mean",
        )
    )  # TODO: Check the correctness of perplexity
    return comp_perp.item()


def prefix_function():
    pass


# TODO: this is super slow, optimize this function
@torch.no_grad()
def property_wise_perplexity(
    logits: torch.Tensor, labels: torch.Tensor, property_names
):
    # print(logits.shape, labels.shape)
    properties_perploxity_freq = {}
    properties_perploxity = {}
    for pred, target in zip(logits, labels):
        print(target[:10])
        for name in property_names:
            prop_ids = torch.tensor(galactica_tokenizer.encode(name))
            start_index = -1
            end_index = -1
            for i in range(len(target) - len(prop_ids)):
                if torch.all(target[i : i + len(prop_ids)] == prop_ids):
                    start_index = i + len(prop_ids) + 1

            if start_index != -1:
                for i in range(start_index + 1, len(target)):
                    if galactica_tokenizer.encode("}") == target[i]:
                        end_index = i
                        break

            if start_index != -1 and end_index != -1:
                if properties_perploxity.get(name) == None:
                    properties_perploxity[name] = 0
                    properties_perploxity_freq[name] = 0
                properties_perploxity[name] += perplexity(
                    pred[start_index:end_index], target[start_index:end_index]
                )

    for k, v in properties_perploxity:
        properties_perploxity[k] = v / properties_perploxity_freq[k]

    return properties_perploxity


def compute_metrics(eval_pred: transformers.EvalPrediction):
    logits, labels = eval_pred.predictions, eval_pred.label_ids

    logits, labels = torch.tensor(logits), torch.tensor(labels)
    prop_wise_perp = property_wise_perplexity(
        logits, labels, ["similarity", "SAS", "WEIGHT"]
    )
    return prop_wise_perp
    # return {"perplexity" : perplexity(logits, labels)}


"""
perplexity = evaluate.load("perplexity", module_type="metric")
input_texts = ["lorem ipsum", "Happy Birthday!", "Bienvenue"]

results = perplexity.compute(model_id='gpt2',
                             add_start_token=False,
                             predictions=input_texts)
print(list(results.keys()))

[out]:

>>>['perplexities', 'mean_perplexity']
print(round(results["mean_perplexity"], 2))
>>>646.75
print(round(results["perplexities"][0], 2))
>>>32.25
"""
