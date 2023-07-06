import transformers
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
from typing import List, Tuple

galactica_model_checkpoint = "facebook/galactica-125m"
galactica_tokenizer = AutoTokenizer.from_pretrained(galactica_model_checkpoint)


@torch.no_grad()
def perplexity(logits: torch.Tensor, labels: torch.Tensor):
    logits = logits.view(logits.size(0) * logits.size(1), logits.size(2))
    labels = labels.view(labels.size(0) * labels.size(1))

    comp_perp = torch.exp(F.cross_entropy(logits, labels, reduction="mean"))
    return comp_perp.item()


# TODO: optimize this function more
@torch.no_grad()
def property_wise_perplexity(
    logits: torch.Tensor, labels: torch.Tensor, property_entries: List[Tuple]
):
    logits = logits.view(logits.size(0) * logits.size(1), logits.size(2))
    labels = labels.view(labels.size(0) * labels.size(1))
    properties_perploxity = {}

    for name, start, end in property_entries:
        start_ids = torch.tensor(galactica_tokenizer.encode(start))
        end_ids = torch.tensor(galactica_tokenizer.encode(end))

        logits_ids, labels_ids = [], []

        start_index = end_index = None
        for i in range(len(labels)):
            if i + len(start_ids) < len(labels) and torch.all(
                labels[i : i + len(start_ids)] == start_ids  # noqa
            ):
                start_index = i + len(start_ids)
            if i + len(end_ids) < len(logits) and torch.all(
                labels[i : i + len(end_ids)] == end_ids  # noqa
            ):
                end_index = i
                if start_index is not None and end_index is not None:
                    assert start_index <= end_index
                    logits_ids.extend(logits[start_index:end_index].tolist())
                    labels_ids.extend(labels[start_index:end_index].tolist())
                    start_index = end_index = None

        properties_perploxity[name] = perplexity(
            torch.tensor(logits_ids).unsqueeze(0), torch.tensor(labels_ids).unsqueeze(0)
        )

    return properties_perploxity


def compute_metrics(eval_pred: transformers.EvalPrediction):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    logits, labels = torch.tensor(logits), torch.tensor(labels)

    property_entries: List[Tuple] = [
        (n, f"[{n}", "]")
        for n in (
            # "name",
            # "experimental",
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
    ]
    property_entries.append(("SMILES", "[START_SMILES]", "[END_SMILES]"))
    prop_wise_perp = property_wise_perplexity(logits, labels, property_entries)

    return {"perplexity": perplexity(logits, labels), **prop_wise_perp}
