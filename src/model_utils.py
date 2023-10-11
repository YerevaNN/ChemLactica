from transformers import OPTForCausalLM, OPTConfig
from custom_modeling_opt import CustomOPTForCausalLM


def load_model(from_pretrained: str, flash_att=False, dtype=None, train_config=None):
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
    model = CustomOPTForCausalLM.from_pretrained(
        from_pretrained, use_flash_attn=flash_att, torch_dtype=dtype
    )
    return model
