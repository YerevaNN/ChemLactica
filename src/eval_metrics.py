import transformers
import torch
import torch.nn.functional as F
from typing import List, Tuple
import math
from utils import ProgressBar
from collections import namedtuple


ProgressBar.set_total(45637)


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


# def get_property_entries():
#     return getattr(get_property_entries, "property_entries", construct_prop_entries())


# TODO: add overflow error handling here
@torch.no_grad()
def perplexity(logits: torch.Tensor, labels: torch.Tensor, base=2):
    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)

    loss = F.cross_entropy(logits, labels, reduction="none")
    pad_mask = labels != 1 # CustomTokenizer.precomuted_ids["<pad>"][0]
    loss = loss * pad_mask  # ignore pad tokens
    comp_perp = base ** (loss.sum() / pad_mask.sum() / math.log(2))
    return comp_perp.item()


@torch.no_grad()
def preprocess_logits_for_metrics(logits: torch.Tensor, labels: torch.Tensor):
    batch_size = labels.size(0)

    logits = logits[..., :-1, :].contiguous().view(-1, logits.size(2))
    labels = labels[..., 1:].contiguous().view(-1)

    # print("input:", CustomTokenizer.get_instance().decode(labels))
    # print(batch_size)

    # property_entries = get_property_entries()
    property_entries = []

    if ProgressBar.get_instance() is None:
        ProgressBar()

    metrics_tensor = torch.zeros(2, len(property_names), device=labels.device)
    metrics_tensor[0][0] = perplexity(logits, labels)
    metrics_tensor[1][0] = 1

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

    ProgressBar.get_instance().update(batch_size)

    return metrics_tensor


def compute_metrics(eval_pred: transformers.EvalPrediction):
    logits, _ = torch.tensor(eval_pred.predictions), torch.tensor(eval_pred.label_ids)

    ProgressBar.delete_instance()
    properties_perp = logits[::2].sum(axis=0)
    properties_count = logits[1::2].sum(axis=0)
    properties_count = torch.max(torch.ones_like(properties_count), properties_count)

    prep = {
        property_names[i]: loss
        for i, loss in enumerate(properties_perp / properties_count)
    }
    return prep
