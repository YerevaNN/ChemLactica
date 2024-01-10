import transformers
import torch
import torch.nn.functional as F
import math
from collections import namedtuple
from utils import get_start2end_tags_map, get_tokenizer
from functools import cache


PropertyEntry = namedtuple(
    "PropertyEntry", ["name", "start_idx", "start_token_len", "prop_idx"]
)

property_names = (
    # "name",
    # "experimental",
    "perplexity",
    "SMILES",
    "CID",
    "SAS",
    "WEIGHT",
    "TPSA",
    "CLOGP",
    "QED",
    "NUMHDONORS",
    "NUMHACCEPTORS",
    "NUMHETEROATOMS",
    "NUMROTATABLEBONDS",
    "NOCOUNT",
    "NHOHCOUNT",
    "RINGCOUNT",
    "HEAVYATOMCOUNT",
    "FRACTIONCSP3",
    "NUMAROMATICRINGS",
    "NUMSATURATEDRINGS",
    "NUMAROMATICHETEROCYCLES",
    "NUMAROMATICCARBOCYCLES",
    "NUMSATURATEDHETEROCYCLES",
    "NUMSATURATEDCARBOCYCLES",
    "NUMALIPHATICRINGS",
    "NUMALIPHATICHETEROCYCLES",
    "NUMALIPHATICCARBOCYCLES",
    "IUPAC",
)


# def construct_prop_entries():
#     property_entries: List[Tuple] = [
#         (
#             n,
#             torch.tensor(
#                 CustomTokenizer.get_instance().encode(f"[{n}"), dtype=torch.int32
#             ),
#             CustomTokenizer.precomuted_ids["]"],
#         )
#         for n in property_names
#     ]

#     property_entries[1] = (
#         "SMILES",
#         CustomTokenizer.precomuted_ids["[START_SMILES]"],
#         CustomTokenizer.precomuted_ids["[END_SMILES]"],
#     )
#     return property_entries


# """
#     the point of this function is to construct the property_entries list specified in the
#     construct_prop_entries function once and return the already created instance when requested.
# """

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

    # start_brackets = torch.where(
    #     torch.bitwise_or(
    #         labels == CustomTokenizer.precomuted_ids["["][0],
    #         labels == CustomTokenizer.precomuted_ids["[START_SMILES]"][0],
    #     )
    # )[0].to(labels.device)
    # start_brackets = torch.where(
    #     torch.bitwise_or(
    #         labels == CustomTokenizer.precomuted_ids["["][0],
    #         labels == CustomTokenizer.precomuted_ids["[START_SMILES]"][0],
    #     )
    # )[0].to(labels.device)
    # start_brackets = torch.cat(
    #     [start_brackets, torch.tensor([len(labels)], device=labels.device)]
    # )

    # starting_indices = []
    # for i, st_brack in enumerate(start_brackets):
    #     st_brack = st_brack.item()
    #     for prop_idx, (name, st_ids, end_ids) in enumerate(
    #         property_entries[1:], start=1
    #     ):
    #         st_ids = st_ids.to(labels.device)
    #         if st_brack + len(st_ids) < len(labels) and torch.all(
    #             labels[st_brack : st_brack + len(st_ids)] == st_ids  # noqa
    #         ):
    #             starting_indices.append(
    #                 PropertyEntry(name, st_brack, len(st_ids), prop_idx)
    #             )

    # starting_indices.append(PropertyEntry("none", len(labels), 0, -1))

    # for i in range(len(starting_indices) - 1):
    #     name, start_idx, start_token_len, prop_idx = starting_indices[i]
    #     _, end_idx, end_token_len, _ = starting_indices[i + 1]
    #     start_idx += start_token_len
    #     end_idx -= 1
    #     if start_idx < end_idx:
    #         metrics_tensor[0][prop_idx] += perplexity(
    #             logits[start_idx:end_idx], labels[start_idx:end_idx]
    #         )
    #         metrics_tensor[1][prop_idx] += 1

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
