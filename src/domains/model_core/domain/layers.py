import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

from shared.config.training_config import ModelConfig

class RMSNorm(nn.Module):
    """RMS Normalization with numerical stability."""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class RotaryPositionalEmbedding(nn.Module):
    """Optimized RoPE implementation with caching."""
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._update_cache(max_seq_len)

    def _update_cache(self, seq_len: int):
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()[None, :, None, :]
            self._sin_cached = emb.sin()[None, :, None, :]

    def rotate_half(self, x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k):
        seq_len = q.shape[1]
        if seq_len > self._seq_len_cached:
            self._update_cache(seq_len)

        cos = self._cos_cached[:, :seq_len, :, :]
        sin = self._sin_cached[:, :seq_len, :, :]

        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)

        return q_embed, k_embed

class MultiHeadAttention(nn.Module):
    """Optimized multi-head attention with GQA and optional Flash Attention."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if config.use_rotary:
            self.rotary = RotaryPositionalEmbedding(
                dim=self.head_dim,
                max_seq_len=config.max_position_embeddings,
                base=config.rope_theta
            )

        self.dropout = nn.Dropout(config.attention_dropout_prob)

    def forward(self, hidden_states, attention_mask=None, use_cache=False, past_key_value=None):
        batch_size, seq_len, _ = hidden_states.shape

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.config.use_rotary:
            query, key = self.rotary(query.transpose(1, 2), key.transpose(1, 2))
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)

        if use_cache:
            if past_key_value is not None:
                key = torch.cat([past_key_value[0], key], dim=2)
                value = torch.cat([past_key_value[1], value], dim=2)
            past_key_value = (key, value)

        if self.num_key_value_groups > 1:
            key = repeat(key, 'b h s d -> b (h g) s d', g=self.num_key_value_groups)
            value = repeat(value, 'b h s d -> b (h g) s d', g=self.num_key_value_groups)

        if self.config.use_flash_attn and FLASH_ATTN_AVAILABLE and not use_cache:
            query = rearrange(query, 'b h s d -> b s h d')
            key = rearrange(key, 'b h s d -> b s h d')
            value = rearrange(value, 'b h s d -> b s h d')

            attn_output = flash_attn_func(
                query, key, value,
                dropout_p=self.config.attention_dropout_prob if self.training else 0.0,
                causal=True
            )
            attn_output = rearrange(attn_output, 'b s h d -> b s (h d)')
        else:
            attn_weights = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            if not use_cache:
                causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=query.device), diagonal=1)
                attn_weights = attn_weights + causal_mask

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
            attn_weights = self.dropout(attn_weights)

            attn_output = torch.matmul(attn_weights, value)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if use_cache:
            return attn_output, past_key_value
        return attn_output

class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))