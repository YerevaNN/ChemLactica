import transformers
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
from typing import List, Tuple
import tqdm

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
)

property_entries: List[Tuple] = [
    (
        n,
        torch.tensor(galactica_tokenizer.encode(f"[{n}"), dtype=torch.int32),
        torch.tensor(galactica_tokenizer.encode("]"), dtype=torch.int32),
    )
    for n in property_names
]
property_entries[1] = (
    "SMILES",
    torch.tensor(galactica_tokenizer.encode("[START_SMILES]"), dtype=torch.int32),
    torch.tensor(galactica_tokenizer.encode("[END_SMILES]"), dtype=torch.int32),
)

pbar = None


@torch.no_grad()
def perplexity(logits: torch.Tensor, labels: torch.Tensor):
    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)

    comp_perp = 2 ** (
        F.cross_entropy(logits, labels, reduction="mean")
    )  # TODO: should the power base be exp or 2?
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
            while j < len(end_brackets) and end_brackets[j] <= s:
                j += 1

            if j < len(end_brackets):
                end_index = end_brackets[j]
                # print(galactica_tokenizer.decode(start_ids),
                # galactica_tokenizer.decode(labels[start_index:end_index]))
                property_perp += perplexity(
                    logits[start_index:end_index], labels[start_index:end_index]
                )
                property_count += 1
                start_index = end_index = None

    return property_perp / max(1, property_count)


@torch.no_grad()
def preprocess_logits_for_metrics(logits: torch.Tensor, labels: torch.Tensor):
    global pbar
    # global pol
    # start_time = time.time()
    original_device = labels.device
    batch_size = labels.size(0)
    logits = logits.view(-1, logits.shape[-1]).to(original_device)
    labels = labels.view(-1).to(original_device)

    if pbar is None:
        pbar = tqdm.tqdm(total=22384)

    metrics_tensor = torch.empty(batch_size * len(property_names), device=labels.device)
    metrics_tensor[0] = perplexity(logits, labels)

    start_brackets = torch.where(labels == galactica_tokenizer.encode("[")[0])[0].to(
        labels.device
    )
    end_brackets = torch.where(labels == galactica_tokenizer.encode("]")[0])[0].to(
        labels.device
    )
    smiles_start_brackets = torch.where(
        labels == galactica_tokenizer.encode("[START_SMILES] ")[0]
    )[0].to(labels.device)
    smiles_end_brackets = torch.where(
        labels == galactica_tokenizer.encode(" [END_SMILES]")[0]
    )[0].to(labels.device)

    for i, (name, start_ids, end_ids) in enumerate(property_entries[1:], start=1):
        metrics_tensor[i] += process_property(
            start_ids.to(labels.device),
            end_ids.to(labels.device),
            start_brackets if name != "SMILES" else smiles_start_brackets,
            end_brackets if name != "SMILES" else smiles_end_brackets,
            logits,
            labels,
        )

    pbar.update(batch_size)

    ret = metrics_tensor.view(batch_size, -1)
    # print(time.time() - start_time, "seconds")
    return ret


def compute_metrics(eval_pred: transformers.EvalPrediction):
    global pbar
    logits, _ = eval_pred.predictions, eval_pred.label_ids
    # logits, labels = torch.tensor(logits), torch.tensor(labels)
    # prop_wise_perp = property_wise_perplexity(logits, labels, property_entries)

    # return {"perplexity": perplexity(logits, labels), **prop_wise_perp}
    # print(logits.shape, labels.shape)
    # logits = preprocess_logits_for_metrics(logits, labels)
    pbar = None
    logits_sum = logits.mean(axis=0)[: len(property_names)]
    prep = {property_names[i]: loss for i, loss in enumerate(logits_sum)}
    return prep
