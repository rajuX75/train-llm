import torch
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

# ============== CONFIGURATION ==============
@dataclass
class ModelConfig:
    """Model configuration, loaded from YAML"""
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float
    layer_norm_eps: float
    hidden_dropout_prob: float
    attention_dropout_prob: float
    initializer_range: float
    use_flash_attn: bool
    use_rotary: bool
    tie_word_embeddings: bool
    gradient_checkpointing: bool

    def __post_init__(self):
        """Validate configuration"""
        assert self.hidden_size % self.num_attention_heads == 0, \
            "hidden_size must be divisible by num_attention_heads"
        assert self.num_attention_heads % self.num_key_value_heads == 0, \
            "num_attention_heads must be divisible by num_key_value_heads"
        if self.use_flash_attn:
            try:
                import flash_attn
            except ImportError:
                print("⚠️ Flash Attention not available, disabling.")
                self.use_flash_attn = False


@dataclass
class TrainingConfig:
    """Training configuration, loaded from YAML"""
    # Paths
    data_dir: str
    output_dir: str
    cache_dir: str

    # Training params
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    min_learning_rate: float
    weight_decay: float
    num_epochs: int
    warmup_steps: int
    max_grad_norm: float

    # Optimization
    use_mixed_precision: bool
    use_gradient_checkpointing: bool
    use_compile: bool

    # Logging
    wandb_enabled: bool
    logging_steps: int
    save_steps: int
    eval_steps: int
    save_total_limit: int

    # Resume training
    resume_from_checkpoint: Optional[str]

    # Hardware
    device: str
    num_workers: int
    seed: int

    def __post_init__(self):
        """Validate and setup directories"""
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)