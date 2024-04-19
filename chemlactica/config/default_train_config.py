from dataclasses import dataclass


@dataclass
class ModelConfig:
    block_size: int = 2048
    vocab_size: int = 50000
    separator_token: str = "</s>"
    tokenizer_path: str = "chemlactica/tokenizer/ChemLacticaTokenizer66"


@dataclass
class TrainConfig:
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    batch_size: int = 500000
    dropout_prob: float = 0.1
    eval_step: int = 256
    global_gradient_norm: float = 1.0
    learning_rate_decay: float = 0.1
    max_learning_rate: float = 6.0e-4
    warmup_steps: int = 500
    weight_decay: float = 0.1
    optimizer: str = "adamw_torch"
    lr_scheduler_type: str = "linear"  # other options [linear, constant_with_warmup]
    bf16: bool = True
    bf16_full_eval: bool = True
    fp16: bool = False
    tf32: bool = True
    evaluation_strategy: str = "steps"  # options are [no, steps, epoch]
    # set manually to total number of checkpoints anticipated to minimize device OOM errors
    save_total_limit: int = 4
    grad_accumulation_scheduler: bool = False
    dynamic_grad_accumulation: bool = False
    grad_accumulation_patience: int = 4000
    grad_accumulation_max: int = 256
    grad_accumulation_delta_steps: int = 100
    grad_accumulation_delta_percentage: float = 0.02


@dataclass
class SFTTrainConfig:
    packing: bool = False
    max_seq_length: int = 512
    neftune_noise_alpha: int = 0
