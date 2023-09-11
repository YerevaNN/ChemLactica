from accelerate import Accelerator
import torch
from transformers import OPTForCausalLM, AdamW, get_linear_schedule_with_warmup
import glob
import os
from datasets import load_dataset
from dataset_utils import process_dataset
from utils import CustomTokenizer


if __name__ == "__main__":
    accelerator = Accelerator()
    model = OPTForCausalLM.from_pretrained("facebook/galactica-125m")
    CustomTokenizer.set_model_size("125m")

    # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
    # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
    # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
    model = model.to(accelerator.device)

    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=1e-5)

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=10000,
    )

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, lr_scheduler = accelerator.prepare(
        model, optimizer, lr_scheduler
    )

    training_data_files = glob.glob(".small_data/train/*.jsonl")
    valid_data_files = glob.glob(".small_data/valid/*.jsonl")
    dataset = load_dataset(
        "text",
        data_files={"train": training_data_files, "validation": valid_data_files},
        streaming=True,
    )

    processed_dataset = process_dataset(
        dataset=dataset, train_config={"block_size": 2048}, process_batch_sizes=(50, 50)
    )

    model.train()
    for inp in processed_dataset["train"]:
        del inp["token_type_ids"]
        outputs = model(**{k: v.unsqueeze(0).to(model.device) for k, v in inp.items()})
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        _input = inp
        break

    model.eval()
    with torch.no_grad():
        out = model(**{k: v.unsqueeze(0).to(model.device) for k, v in _input.items()})

    accelerator.save_state("sample")

    acc = Accelerator()
    saved_model = OPTForCausalLM.from_pretrained("facebook/galactica-125m")
    saved_model = acc.prepare(saved_model)
    saved_model = saved_model.to(acc.device)
    acc.load_state("sample")

    saved_model.eval()
    with torch.no_grad():
        saved_out = saved_model(**{k: v.unsqueeze(0).to(saved_model.device) for k, v in _input.items()})

    is_repr = torch.allclose(out.logits, saved_out.logits.to(out.logits.device), atol=1e-4)
    if is_repr:
        print(f"Model is reproducable.")
    else:
        print(f"Model is not reproducable.")