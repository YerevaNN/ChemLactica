from transformers import OPTForCausalLM, OPTConfig
from utils import get_tokenizer_special_tokens
import bitsandbytes as bnb
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import torch
from custom_modeling_opt import CustomOPTForCausalLM
from transformers import BitsAndBytesConfig


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


def get_llama_token_count():
    added_chem_token_count = len(get_tokenizer_special_tokens())
    added_pad_token_count = 1
    return added_chem_token_count + added_pad_token_count


def load_model(
    from_pretrained: str,
    use_flash_attn=True,
    dtype=None,
    train_config=None,
    auth_token=None,
):
    if from_pretrained == "small_opt":
        return OPTForCausalLM(
            OPTConfig(
                vocab_size=train_config["vocab_size"],
                hidden_size=train_config["hidden_size"],
                num_hidden_layers=train_config["num_hidden_layers"],
                ffn_dim=train_config["ffn_dim"],
                max_position_embeddings=train_config["max_position_embeddings"],
                num_attention_heads=train_config["num_attention_heads"],
                word_embed_proj_dim=train_config["word_sembed_proj_dim"],
            )
        )
    if "galactica" in from_pretrained.lower():
        model = CustomOPTForCausalLM.from_pretrained(
            from_pretrained, use_flash_attn=use_flash_attn, torch_dtype=dtype
        )
    return model
