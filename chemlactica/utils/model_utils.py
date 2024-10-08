from transformers import (
    OPTForCausalLM,
    OPTConfig,
    MistralForCausalLM,
    AutoModelForCausalLM,
)

# from .utils import get_tokenizer_special_tokens

import bitsandbytes as bnb

# from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import torch
from transformers import BitsAndBytesConfig
import torch.nn as nn


class LinearFloat32(nn.Linear):
    def forward(self, _input) -> torch.Tensor:
        return super().forward(_input).to(torch.float32)


def cast_lm_head_to_fp32_init(func):
    def inner_func(self, config, *args, **kwargs):
        func(self, config, *args, **kwargs)
        self.lm_head = LinearFloat32(
            config.word_embed_proj_dim, config.vocab_size, bias=False
        )

    return inner_func


def float_casting_decorator(layer_class):
    class FloatCastingLayer(layer_class):
        def __init__(self, *args, **kwargs):
            super(FloatCastingLayer, self).__init__(*args, **kwargs)

        def forward(
            self,
            x,
            *args,
            **kwargs,
        ):
            return super().forward(x, *args, **kwargs).to(torch.float32)

    return FloatCastingLayer


quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


def find_all_linear_names(model):
    cls = (
        bnb.nn.Linear4bit
    )  # if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    # if 'embed_tokens' in lora_module_names:
    #     lora_module_names.remove('embed_tokens')
    return list(lora_module_names)


def select_attention_implementation(use_flash_attn):
    # flash attention 1 not even supported in huggingface
    # also, we have no special need for it regardless
    # see: https://github.com/huggingface/transformers/blob/da3c79b245afcce88f5db79ada10bf5b7c200ab1/src/transformers/models/opt/modeling_opt.py#L496 # noqa
    if use_flash_attn:
        return "flash_attention_2"
    else:
        return "eager"


# def get_llama_token_count():
#     added_chem_token_count = len(get_tokenizer_special_tokens())
#     added_pad_token_count = 1
#     return added_chem_token_count + added_pad_token_count


def load_model(
    from_pretrained: str,
    use_flash_attn=True,
    dtype=None,
    model_config=None,
    auth_token=None,
    gradient_checkpointing=True,
):
    attn_implementation = select_attention_implementation(use_flash_attn)
    if from_pretrained == "small_opt":
        return OPTForCausalLM(
            OPTConfig(
                vocab_size=model_config["vocab_size"],
                hidden_size=model_config["hidden_size"],
                num_hidden_layers=model_config["num_hidden_layers"],
                ffn_dim=model_config["ffn_dim"],
                max_position_embeddings=model_config["max_position_embeddings"],
                num_attention_heads=model_config["num_attention_heads"],
                word_embed_proj_dim=model_config["word_embed_proj_dim"],
            )
        )
    if "galactica" in from_pretrained.lower():
        OPTForCausalLM.__init__ = cast_lm_head_to_fp32_init(OPTForCausalLM.__init__)
        model = OPTForCausalLM.from_pretrained(
            from_pretrained, torch_dtype=dtype, attn_implementation=attn_implementation
        )

        # model.lm_head = float_casting_decorator(model.lm_head.__class__)(
        #     in_features=model.lm_head.in_features,
        #     out_features=model.lm_head.out_features,
        # )
        # model.lm_head.forward = cast_to_fp32(OPTForCausalLM.lm_head.forward)

    if "mistral" in from_pretrained.lower():
        model = MistralForCausalLM.from_pretrained(
            from_pretrained,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            sliding_window=model_config["sliding_window"],
        )

    if "gemma" in from_pretrained.lower():
        model = AutoModelForCausalLM.from_pretrained(
            from_pretrained,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        )

    if gradient_checkpointing:
        model.use_cache = (
            False  # use cache true doesn't work with gradient checkpointing
        )
    return model
