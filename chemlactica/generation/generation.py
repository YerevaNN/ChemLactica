from argparse import ArgumentParser
from typing import List
import torch

from model_utils import load_model
from utils import get_tokenizer


def generate(prompts: List[str], model, **gen_kwargs):
    if type(prompts) == str:
        prompts = [prompts]
    tokenizer = get_tokenizer()

    generation_dict = {}
    for prompt in prompts:
        data = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        outputs = model.generate(
            input_ids=data.input_ids,
            **gen_kwargs
        )
        if not generation_dict.get(prompt):
            generation_dict[prompt] = []
        for out in outputs:
            generation_dict[prompt].append(tokenizer.decode(out[len(data):]))
    return generation_dict


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--prompts",
        type=str,
        nargs="*",
        required=True,
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--flash_attn",
        action="store_true",
        dest="use_flash_attn",
        help="whether or not to use flash attn)",
    )
    parser.add_argument(
        "--no_flash_attn",
        action="store_false",
        dest="use_flash_attn",
        help="whether or not to use flash attn",
    )
    parser.set_defaults(use_flash_attn=False)
    parser.add_argument(
        "--device",
        type=str,
        required=True
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        required=False,
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        dest="do_sample",
    )
    parser.add_argument(
        "--no_do_sample",
        action="store_false",
        dest="do_sample",
    )
    parser.set_defaults(do_sample=False)
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        required=False,
        default=None
    )
    parser.add_argument(
        "--diversity_penalty",
        type=int,
        required=False,
        default=None
    )
    parser.add_argument(
        "--repetition_penalty",
        type=int,
        required=False,
        default=None
    )
    parser.add_argument(
        "--length_penalty",
        type=int,
        required=False,
        default=None
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--num_beams_groups",
        type=int,
        required=False,
        default=None,
    )

    args = parser.parse_args()
    args = {key: value for key, value in args.__dict__.items() if value != None}
    
    # generate(
    #     args.pop("prompts"), args.pop("checkpoint_path"),
    #     args.pop("use_flash_attn"), device=args.pop("device"),
    #     **args
    # )