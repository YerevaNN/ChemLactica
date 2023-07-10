import transformers
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
from typing import List, Tuple
import tqdm
from multiprocessing import get_context

galactica_model_checkpoint = "facebook/galactica-125m"
galactica_tokenizer = AutoTokenizer.from_pretrained(galactica_model_checkpoint)

pbar = None

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

property_entries: List[Tuple] = [(n, f"[{n}", "]") for n in property_names]
property_entries[1] = ("SMILES", "[START_SMILES]", "[END_SMILES]")


@torch.no_grad()
def perplexity(logits: torch.Tensor, labels: torch.Tensor):
    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)

    comp_perp = torch.exp(
        F.cross_entropy(logits, labels, reduction="mean")
    )  # TODO: should the power base be exp or 2?
    return comp_perp.item()


def prefix_function(ids: torch.tensor):
    length = ids.size(0)
    pref_func = torch.zeros(length, dtype=torch.int32)
    for i in range(1, length):
        j = pref_func[i - 1].item()
        while j > 0 and ids[i].item() != ids[j].item():
            j = pref_func[j - 1].item()
        if ids[i].item() == ids[j].item():
            j += 1
        pref_func[i] = j
    return pref_func


def process_property(entry):
    # be cautious, if the property is not detected in the sequence it will get 0 perplexity
    property_perp = 0.0
    start, end, labels, logits = entry
    start_ids = torch.tensor(
        galactica_tokenizer.encode(start), device=labels.device, dtype=torch.int32
    )
    end_ids = torch.tensor(
        galactica_tokenizer.encode(end), device=logits.device, dtype=torch.int32
    )
    logits_ids, labels_ids = [], []

    start_index = end_index = None
    # KMP algorithm
    start_pref_func = prefix_function(
        torch.cat((start_ids, torch.tensor([-1], device=labels.device), labels), dim=0)
    )
    end_pref_func = prefix_function(
        torch.cat((end_ids, torch.tensor([-1], device=labels.device), labels), dim=0)
    )

    for i in range(len(labels)):
        s_i = i + len(start_ids)
        e_i = i + len(end_ids)
        if s_i < len(start_pref_func) and start_pref_func[s_i] == len(start_ids):
            start_index = i
        if e_i < len(end_pref_func) and end_pref_func[e_i] == len(end_ids):
            end_index = i - len(end_ids)
            if start_index is not None and end_index is not None:
                assert start_index <= end_index
                # print(name, galactica_tokenizer.decode(labels[start_index:end_index]))
                logits_ids.extend(logits[start_index:end_index].tolist())
                labels_ids.extend(labels[start_index:end_index].tolist())
                start_index = end_index = None

    if len(logits_ids) > 0:
        property_perp = perplexity(
            torch.tensor(logits_ids).unsqueeze(0),
            torch.tensor(labels_ids).unsqueeze(0),
        )

    return property_perp


@torch.no_grad()
def preprocess_logits_for_metrics(logits: torch.Tensor, labels: torch.Tensor):
    global pbar
    compute_device = "cpu"
    original_device = labels.device
    batch_size = labels.size(0)
    logits = logits.view(-1, logits.shape[-1]).to(compute_device)
    labels = labels.view(-1).to(compute_device)

    if pbar is None:
        pbar = tqdm.tqdm(total=22384)

    metrics_tensor = torch.empty(
        batch_size * len(property_names), device=compute_device
    )
    metrics_tensor[0] = perplexity(logits, labels)

    with get_context("spawn").Pool(32) as pol:
        for i, value in enumerate(
            pol.map(
                process_property,
                tuple(
                    (start, end, labels, logits)
                    for name, start, end in property_entries[1:]
                ),
            )
        ):
            metrics_tensor[i + 1] = value

    pbar.update(batch_size)

    ret = metrics_tensor.view(batch_size, -1)
    # print(time.time() - start_time, "seconds")
    return ret.to(original_device)


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


# @torch.no_grad()
# def property_wise_perplexity(
#     logits: torch.Tensor, labels: torch.Tensor, property_entries: List[Tuple]
# ):
#     logits = logits.view(logits.size(0) * logits.size(1), logits.size(2))
#     labels = labels.view(labels.size(0) * labels.size(1))
#     properties_perploxity = {}

#     for name, start, end in property_entries:
#         start_ids = torch.tensor(
#             galactica_tokenizer.encode(start), device=logits.device
#         )
#         end_ids = torch.tensor(galactica_tokenizer.encode(end), device=logits.device)

#         logits_ids, labels_ids = [], []

#         start_index = end_index = None
#         for i in range(len(labels)):
#             if i + len(start_ids) < len(labels) and torch.all(
#                 labels[i : i + len(start_ids)] == start_ids  # noqa
#             ):
#                 start_index = i + len(start_ids)
#             if i + len(end_ids) < len(logits) and torch.all(
#                 labels[i : i + len(end_ids)] == end_ids  # noqa
#             ):
#                 end_index = i
#                 if start_index is not None and end_index is not None:
#                     assert start_index <= end_index
#                     logits_ids.extend(logits[start_index:end_index].tolist())
#                     labels_ids.extend(labels[start_index:end_index].tolist())
#                     start_index = end_index = None

#         if len(logits_ids) > 0:
#             properties_perploxity[name] = perplexity(
#                 torch.tensor(logits_ids).unsqueeze(0),
#                 torch.tensor(labels_ids).unsqueeze(0),
#             )
#             # print(galactica_tokenizer.decode(torch.tensor(logits_ids).argmax(-1)))
#             # print(galactica_tokenizer.decode(labels_ids))

#     return properties_perploxity
