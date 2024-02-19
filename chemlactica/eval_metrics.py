import transformers
import torch
import torch.nn.functional as F
import math
from collections import namedtuple
from .utils.utils import get_start2end_tags_map, get_tokenizer
from functools import cache


def get_prop2index_map(start2end_tags: dict):
    prop2index_map = {start: i for i, start in enumerate(start2end_tags.keys())}
    def inner_func(prop: str):
        return prop2index_map[prop]
    return inner_func


def get_index2prop_map(start2end_tags: dict):
    index2prop_map = [start for start in start2end_tags.keys()]
    def inner_func(index: int):
        return index2prop_map[index]
    return inner_func


# TODO: add overflow error handling here
@torch.no_grad()
def perplexity(logits: torch.Tensor, labels: torch.Tensor, base=2):
    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)

    loss = F.cross_entropy(logits, labels, reduction="none")
    pad_mask = labels != 1  # CustomTokenizer.precomuted_ids["<pad>"][0]
    loss = loss * pad_mask  # ignore pad tokens
    comp_perp = base ** (loss.sum() / pad_mask.sum() / math.log(2))
    return comp_perp.item()


@torch.no_grad()
def preprocess_logits_for_metrics(logits: torch.Tensor, labels: torch.Tensor):
    batch_size = labels.size(0)

    logits = logits[..., :-1, :].contiguous().view(-1, logits.size(2))
    labels = labels[..., 1:].contiguous().view(-1)

    tokenizer = get_tokenizer()
    # print(tokenizer.decode(labels))

    # metrics_tensor is matrix containing perplexities related to properties
    # metrics_tensor[0][i] shows the perplexity of the ith property
    # metrics_tensor[1][i] shows the number of times the ith property occured
    # metrics_tensor[...][-1] is for the perplexity of the whole sequence
    start2end_tags = get_start2end_tags_map()
    metrics_tensor = torch.zeros(2, len(start2end_tags) + 1, device=labels.device)
    metrics_tensor[0][-1] = perplexity(logits, labels)
    metrics_tensor[1][-1] = 1

    start_tags_mask = torch.zeros(labels.size(0), dtype=torch.bool, device=labels.device)
    end_tags_mask = torch.zeros(labels.size(0), dtype=torch.bool, device=labels.device)
    for start, end in start2end_tags.items():
        start_mask = (labels == tokenizer.encode(start)[0])
        start_tags_mask = torch.bitwise_or(start_tags_mask, start_mask)

        end_mask = (labels == tokenizer.encode(end)[0])
        end_tags_mask = torch.bitwise_or(end_tags_mask, end_mask)

    start_tags_indices = torch.where(start_tags_mask)[0]
    end_tags_indices = torch.where(end_tags_mask)[0]

    prop2index = get_prop2index_map(start2end_tags)
    # two pointers
    first_ptr = 0
    second_ptr = 0
    while first_ptr < start_tags_indices.size(0) and second_ptr < end_tags_indices.size(0):
        while second_ptr < end_tags_indices.size(0) and start_tags_indices[first_ptr] >= end_tags_indices[second_ptr]:
            second_ptr += 1
        if second_ptr < end_tags_indices.size(0):
            # [PROP_NAME]...value...[/PROP_NAME]
            #      ^               ^
            # start_index       end_index (one before the closing tag)
            start_index = start_tags_indices[first_ptr]
            end_index = end_tags_indices[second_ptr]
            index = prop2index(tokenizer.decode(labels[start_index]))
            # print(tokenizer.decode(labels[start_index]), ":", index, "perp", perplexity(logits[start_index+1:end_index], labels[start_index+1:end_index]))
            metrics_tensor[0][index] += perplexity(logits[start_index+1:end_index], labels[start_index+1:end_index])
            metrics_tensor[1][index] += 1
        first_ptr += 1

    return metrics_tensor


def compute_metrics(eval_pred: transformers.EvalPrediction):
    logits, _ = torch.tensor(eval_pred.predictions), torch.tensor(eval_pred.label_ids)

    properties_perp = logits[::2].sum(axis=0)
    properties_count = logits[1::2].sum(axis=0)
    properties_count = torch.max(torch.ones_like(properties_count), properties_count)

    index2prop = get_index2prop_map(get_start2end_tags_map())
    propwise_perp = {
        index2prop(i)[1:-1]: loss
        for i, loss in enumerate(properties_perp[:-1] / properties_count[:-1])
    }
    propwise_perp["perplexity"] = properties_perp[-1] / properties_count[-1]
    propwise_perp["SMILES"] = propwise_perp.pop("START_SMILES")
    return propwise_perp
