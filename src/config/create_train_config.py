import yaml


model_train_configs = {
    "125m": {
        "n_layers": 12,
        "d_model": 768,
        "n_heads": 12,
        "d_heads": 64,
        "batch_size": 500000,
        "block_size": 2048,
        "vocab_size": 50000,
        "max_learning_rate": 6e-4,
        "warmup_steps": 500,  # deviation from the paper
        "global_gradient_norm": 1.0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "weight_decay": 0.1,
        "dropout_prob": 0.1,
        "eval_step": 256,
        "learning_rate_decay": 0.1,
    },
    "1.3b": {
        "n_layers": 24,
        "d_model": 2048,
        "n_heads": 32,
        "d_heads": 64,
        "batch_size": 1000000,
        "block_size": 2048,
        "vocab_size": 50000,
        "max_learning_rate": 2e-4,
        "warmup_steps": 500,  # deviation from the paper
        "global_gradient_norm": 1.0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "weight_decay": 0.1,
        "dropout_prob": 0.1,
        "learning_rate_decay": 0.1,
    },
    "6.7b": {
        "n_layers": 32,
        "d_model": 4096,
        "n_heads": 32,
        "d_heads": 128,
        "batch_size": 2000000,
        "block_size": 2048,
        "vocab_size": 50000,
        "max_learning_rate": 1.2e-4,
        "warmup_steps": 500,  # deviation from the paper
        "global_gradient_norm": 1.0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "weight_decay": 0.1,
        "dropout_prob": 0.1,
        "learning_rate_decay": 0.1,
    },
    "small_opt": {
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "ffn_dim": 16,
        "max_position_embeddings": 2048,
        "num_attention_heads": 1,
        "word_embed_proj_dim": 16,
        "batch_size": 2,
        "block_size": 2048,
        "vocab_size": 50000,
        "max_learning_rate": 6e-4,
        "warmup_steps": 500,  # deviation from the paper
        "global_gradient_norm": 1.0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "weight_decay": 0.1,
        "dropout_prob": 0.1,
        "learning_rate_decay": 0.1,
    },
}

# manual changes
model_train_configs["125m"]["batch_size"] = 4
model_train_configs["1.3b"]["batch_size"] = 1

for key in model_train_configs.keys():
    model_train_configs[key]["max_learning_rate"] *= 0.08


with open("models_train_config.yaml", "w") as f_:
    yaml.dump(model_train_configs, f_)
