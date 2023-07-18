import transformers
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
from typing import List, Tuple
import math
from utils import ProgressBar
from collections import namedtuple


ProgressBar.set_total(22384)

galactica_model_checkpoint = "facebook/galactica-125m"
galactica_tokenizer = AutoTokenizer.from_pretrained(galactica_model_checkpoint)

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
    "RELATED",
    "EXPERIMENTAL",
    "SYNONYMS",
)

text_to_ids = {
    v: torch.tensor(galactica_tokenizer.encode(v), dtype=torch.int32)
    for v in ["[START_SMILES]", "[END_SMILES]", "[", "]", "<pad>"]
}

property_entries: List[Tuple] = [
    (
        n,
        torch.tensor(galactica_tokenizer.encode(f"[{n}"), dtype=torch.int32),
        text_to_ids["]"],
    )
    for n in property_names
]
property_entries[1] = (
    "SMILES",
    text_to_ids["[START_SMILES]"],
    text_to_ids["[END_SMILES]"],
)

PropertyEntry = namedtuple(
    "PropertyEntry", ["name", "start_idx", "start_token_len", "prop_idx"]
)


@torch.no_grad()
def perplexity(logits: torch.Tensor, labels: torch.Tensor, base=2):
    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)

    loss = F.cross_entropy(logits, labels, reduction="none")
    pad_mask = labels != text_to_ids["<pad>"][0]
    loss = loss * pad_mask  # ignore pad tokens
    comp_perp = base ** (loss.sum() / pad_mask.sum() / math.log(2))
    return comp_perp.item()


def process_property(start_ids, end_ids, start_brackets, end_brackets, logits, labels):
    # be cautious, if the property is not detected in the sequence it will get 0 perplexity
    property_perp = 0.0
    property_count = 0

    start_index = end_index = None

    for i, s in enumerate(start_brackets):
        if i + len(start_ids) < len(start_brackets) and torch.all(
            labels[s : s + len(start_ids)] == start_ids  # noqa
        ):
            start_index = s + len(start_ids)
            j = i
            while j < len(end_brackets) and end_brackets[j] < start_index:
                j += 1

            if j < len(end_brackets):
                end_index = end_brackets[j]
                assert start_index < end_index
                property_perp += perplexity(
                    logits[start_index:end_index], labels[start_index:end_index]
                )
                print(
                    galactica_tokenizer.decode(start_ids),
                    galactica_tokenizer.decode(labels[start_index:end_index]),
                )
                property_count += 1
                start_index = end_index = None

    return property_perp, property_count


pbar = None


@torch.no_grad()
def preprocess_logits_for_metrics(logits: torch.Tensor, labels: torch.Tensor):
    global pbar
    batch_size = labels.size(0)

    logits = logits[..., :-1, :].contiguous().view(-1, logits.size(2))
    labels = labels[..., 1:].contiguous().view(-1)

    if ProgressBar.get_instance() is None:
        ProgressBar()

    metrics_tensor = torch.zeros(2, len(property_names), device=labels.device)
    metrics_tensor[0][0] = perplexity(logits, labels)
    metrics_tensor[1][0] = 1

    start_brackets = torch.where(
        torch.bitwise_or(
            labels == text_to_ids["["][0], labels == text_to_ids["[START_SMILES]"][0]
        )
    )[0].to(labels.device)
    start_brackets = torch.cat(
        [start_brackets, torch.tensor([len(labels)], device=labels.device)]
    )

    starting_indices = []
    for i, st_brack in enumerate(start_brackets):
        st_brack = st_brack.item()
        for prop_idx, (name, st_ids, end_ids) in enumerate(
            property_entries[1:], start=1
        ):
            st_ids = st_ids.to(labels.device)
            if st_brack + len(st_ids) < len(labels) and torch.all(
                labels[st_brack : st_brack + len(st_ids)] == st_ids  # noqa
            ):
                starting_indices.append(
                    PropertyEntry(name, st_brack, len(st_ids), prop_idx)
                )

    starting_indices.append(PropertyEntry("none", len(labels), 0, -1))

    for i in range(len(starting_indices) - 1):
        name, start_idx, start_token_len, prop_idx = starting_indices[i]
        _, end_idx, end_token_len, _ = starting_indices[i + 1]
        start_idx += start_token_len
        end_idx -= 1
        assert start_idx <= end_idx
        if start_idx < end_idx:
            # print(name, galactica_tokenizer.decode(labels[start_idx:end_idx]))
            metrics_tensor[0][prop_idx] += perplexity(
                logits[start_idx:end_idx], labels[start_idx:end_idx]
            )
            metrics_tensor[1][prop_idx] += 1

    # start_brackets = torch.where(labels == text_to_ids["["][0])[0].to(
    #     labels.device
    # )
    # end_brackets = torch.where(labels == text_to_ids["]"][0])[0].to(
    #     labels.device
    # )
    # smiles_start_brackets = torch.where(
    #     labels == text_to_ids["[START_SMILES]"][0]
    # )[0].to(labels.device)
    # smiles_end_brackets = torch.where(
    #     labels == text_to_ids["[END_SMILES]"][0]
    # )[0].to(labels.device)

    # for i, (name, start_ids, end_ids) in enumerate(property_entries[1:], start=1):
    #     property_perp, property_count = process_property(
    #         start_ids.to(labels.device),
    #         end_ids.to(labels.device),
    #         # start_brackets if name != "SMILES" else smiles_start_brackets,
    #         # end_brackets if name != "SMILES" else smiles_end_brackets,
    #         start_brackets,
    #         end_brackets,
    #         logits,
    #         labels,
    #     )

    #     metrics_tensor[0][i] = property_perp
    #     metrics_tensor[1][i] = property_count

    ProgressBar.get_instance().update(batch_size)

    return metrics_tensor


def compute_metrics(eval_pred: transformers.EvalPrediction):
    global pbar
    logits, _ = torch.tensor(eval_pred.predictions), torch.tensor(eval_pred.label_ids)

    ProgressBar.delete_instance()
    properties_perp = logits[::2].sum(axis=0)
    properties_count = logits[1::2].sum(axis=0)
    properties_count = torch.max(torch.ones_like(properties_count), properties_count)
    # # print(properties_perp, properties_count)
    prep = {
        property_names[i]: loss
        for i, loss in enumerate(properties_perp / properties_count)
    }
    return prep
