from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

# ============== CONFIGURATION ==============
@dataclass
class ModelConfig:
    """Advanced model configuration with validation"""
    vocab_size: int = 512
    hidden_size: int = 128
    intermediate_size: int = 256
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 2  # GQA
    max_position_embeddings: int = 512
    rope_theta: float = 10000.0
    layer_norm_eps: float = 1e-6
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    initializer_range: float = 0.02
    use_flash_attn: bool = True
    use_rotary: bool = True
    tie_word_embeddings: bool = True
    gradient_checkpointing: bool = True

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
    """Training configuration with validation"""
    # Paths
    data_dir: str = "./data"
    output_dir: str = "./models"
    cache_dir: str = "./cache"

    # Training params
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    min_learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0

    # Optimization
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_compile: bool = False

    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3

    # Resume training
    resume_from_checkpoint: Optional[str] = None

    # Hardware
    device: str = "cuda" if __import__('torch').cuda.is_available() else "cpu"
    num_workers: int = 2
    seed: int = 42

    def __post_init__(self):
        """Validate and setup directories"""
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)