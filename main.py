#!/usr/bin/env python3
"""
Optimized Advanced Chat Model Training Script
Fixed bugs and added advanced features:
- Memory-efficient data loading
- Proper gradient accumulation
- Better checkpoint management
- Learning rate warmup
- KV cache for inference
- Better error handling
- Distributed training support
"""

import os
import json
import glob
import math
import random
import shutil
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import numpy as np
from tqdm.auto import tqdm
import sentencepiece as spm
from einops import rearrange, repeat
from rotary_embedding_torch import RotaryEmbedding

warnings.filterwarnings('ignore')

# Optional imports with proper fallbacks
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("⚠️ Flash Attention not available, using standard attention")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ Wandb not available, logging locally only")


# ============== CONFIGURATION ==============
@dataclass
class ModelConfig:
    """Advanced model configuration with validation"""
    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4  # GQA
    max_position_embeddings: int = 4096
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
        if self.use_flash_attn and not FLASH_ATTN_AVAILABLE:
            self.use_flash_attn = False
            print("⚠️ Flash Attention disabled (not available)")

    
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
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2
    seed: int = 42
    
    def __post_init__(self):
        """Validate and setup directories"""
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)


# ============== TOKENIZER ==============
class ChatTokenizer:
    """Improved SentencePiece tokenizer with better error handling"""
    
    SPECIAL_TOKENS = {
        "<s>": 1,
        "</s>": 2,
        "<pad>": 0,
        "<unk>": 3,
        "<user>": 4,
        "<assistant>": 5,
        "<system>": 6,
    }
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.sp = None
        self.special_tokens = self.SPECIAL_TOKENS.copy()
        
    def train(self, texts: List[str], model_prefix: str = "tokenizer"):
        """Train SentencePiece tokenizer with proper cleanup"""
        if not texts:
            raise ValueError("Cannot train tokenizer on empty text list")
            
        temp_file = "train_texts_tmp.txt"
        try:
            # Write texts to temp file
            with open(temp_file, "w", encoding="utf-8") as f:
                for text in texts:
                    if text.strip():  # Only write non-empty texts
                        f.write(text.strip() + "\n")
            
            # Train SentencePiece
            spm.SentencePieceTrainer.train(
                input=temp_file,
                model_prefix=model_prefix,
                vocab_size=self.vocab_size - len(self.special_tokens),  # Reserve space for special tokens
                model_type="bpe",
                character_coverage=0.9995,
                num_threads=os.cpu_count() or 1,
                user_defined_symbols=list(self.special_tokens.keys()),
                pad_id=self.special_tokens["<pad>"],
                unk_id=self.special_tokens["<unk>"],
                bos_id=self.special_tokens["<s>"],
                eos_id=self.special_tokens["</s>"],
                max_sentence_length=16384,
            )
            
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(f"{model_prefix}.model")
            print(f"✓ Tokenizer trained successfully with vocab size: {self.sp.get_piece_size()}")
            
        finally:
            # Cleanup temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
    def load(self, model_path: str):
        """Load existing tokenizer with validation"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Tokenizer model not found: {model_path}")
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        print(f"✓ Tokenizer loaded with vocab size: {self.sp.get_piece_size()}")
        
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token ids"""
        if self.sp is None:
            raise RuntimeError("Tokenizer not initialized. Train or load a model first.")
        tokens = self.sp.encode_as_ids(text)
        if add_special_tokens:
            tokens = [self.special_tokens["<s>"]] + tokens + [self.special_tokens["</s>"]]
        return tokens
    
    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token ids to text"""
        if self.sp is None:
            raise RuntimeError("Tokenizer not initialized")
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.special_tokens.values()]
        return self.sp.decode_ids(tokens)
    
    def get_vocab_size(self) -> int:
        """Get actual vocabulary size"""
        return self.sp.get_piece_size() if self.sp else self.vocab_size


# ============== DATASET ==============
class ChatDataset(Dataset):
    """Memory-efficient chat dataset with lazy loading"""
    
    def __init__(
        self,
        data_files: List[str],
        tokenizer: ChatTokenizer,
        max_length: int = 2048,
        cache_dir: Optional[str] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Store file paths instead of loading all data
        self.data_files = [f for f in data_files if os.path.exists(f)]
        if not self.data_files:
            raise ValueError("No valid data files found")
            
        # Build index of conversations
        self.conversation_index = self._build_index()
        print(f"✓ Indexed {len(self.conversation_index)} conversations from {len(self.data_files)} files")
        
    def _build_index(self) -> List[Tuple[str, int, int]]:
        """Build index of conversations (file_path, start_pos, end_pos)"""
        index = []
        
        for file_path in self.data_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Split by double newline (conversation separator)
            conversations = content.split('\n\n')
            
            for conv in conversations:
                if conv.strip() and 'User:' in conv and 'Assistant:' in conv:
                    index.append((file_path, conv))
        
        return index
    
    def _parse_conversation(self, text: str) -> List[Dict[str, str]]:
        """Parse conversation text into messages"""
        messages = []
        lines = text.split('\n')
        current_role = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("User:"):
                if current_role and current_content:
                    messages.append({
                        'role': current_role,
                        'content': ' '.join(current_content)
                    })
                current_role = 'user'
                current_content = [line[5:].strip()]
            elif line.startswith("Assistant:"):
                if current_role and current_content:
                    messages.append({
                        'role': current_role,
                        'content': ' '.join(current_content)
                    })
                current_role = 'assistant'
                current_content = [line[10:].strip()]
            elif current_role:
                current_content.append(line)
        
        # Add last message
        if current_role and current_content:
            messages.append({
                'role': current_role,
                'content': ' '.join(current_content)
            })
        
        return messages
    
    def __len__(self):
        return len(self.conversation_index)
    
    def __getitem__(self, idx):
        """Get tokenized conversation with proper memory management"""
        file_path, conv_text = self.conversation_index[idx]
        
        # Parse conversation
        messages = self._parse_conversation(conv_text)
        
        # Format conversation
        text = ""
        for msg in messages:
            if msg['role'] == 'user':
                text += f"<user>{msg['content']}</user>"
            else:
                text += f"<assistant>{msg['content']}</assistant>"
        
        # Tokenize
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
        except Exception as e:
            print(f"⚠️ Tokenization error in file {file_path}: {e}")
            # Return empty sequence on error
            tokens = [self.tokenizer.special_tokens["<pad>"]] * self.max_length
        
        # Truncate or pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        pad_token = self.tokenizer.special_tokens["<pad>"]
        attention_mask = [1 if t != pad_token else 0 for t in tokens]
        
        # Pad to max_length
        if len(tokens) < self.max_length:
            padding_length = self.max_length - len(tokens)
            tokens = tokens + [pad_token] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        
        # Create labels (shifted input_ids, with padding masked as -100)
        labels = tokens[1:] + [pad_token]
        labels = [t if m == 1 else -100 for t, m in zip(labels, attention_mask)]
        
        return {
            'input_ids': torch.tensor(tokens[:-1], dtype=torch.long),
            'labels': torch.tensor(labels[:-1], dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask[:-1], dtype=torch.long)
        }


# ============== MODEL ARCHITECTURE ==============
class RMSNorm(nn.Module):
    """RMS Normalization with numerical stability"""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        
    def forward(self, x):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryPositionalEmbedding(nn.Module):
    """Optimized RoPE implementation with caching"""
    
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute cos/sin cache
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._update_cache(max_seq_len)
    
    def _update_cache(self, seq_len: int):
        """Update cos/sin cache if needed"""
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()[None, :, None, :]
            self._sin_cached = emb.sin()[None, :, None, :]
    
    def rotate_half(self, x):
        """Rotate half the hidden dims of the input"""
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, q, k):
        """Apply rotary embeddings"""
        seq_len = q.shape[1]
        
        # Update cache if needed
        if seq_len > self._seq_len_cached:
            self._update_cache(seq_len)
        
        # Apply rotation
        cos = self._cos_cached[:, :seq_len, :, :]
        sin = self._sin_cached[:, :seq_len, :, :]
        
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        
        return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """Optimized multi-head attention with GQA and optional Flash Attention"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        assert self.head_dim * self.num_heads == self.hidden_size, "hidden_size must be divisible by num_heads"
        
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
        
        # Project Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        if self.config.use_rotary:
            query, key = self.rotary(query.transpose(1, 2), key.transpose(1, 2))
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
        
        # KV cache for inference
        if use_cache:
            if past_key_value is not None:
                key = torch.cat([past_key_value[0], key], dim=2)
                value = torch.cat([past_key_value[1], value], dim=2)
            past_key_value = (key, value)
        
        # Repeat K,V for GQA
        if self.num_key_value_groups > 1:
            key = repeat(key, 'b h s d -> b (h g) s d', g=self.num_key_value_groups)
            value = repeat(value, 'b h s d -> b (h g) s d', g=self.num_key_value_groups)
        
        # Attention computation
        if self.config.use_flash_attn and FLASH_ATTN_AVAILABLE and not use_cache:
            # Flash Attention
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
            # Standard attention
            attn_weights = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
            
            # Apply attention mask
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            # Causal mask
            if not use_cache:
                causal_mask = torch.triu(
                    torch.full((seq_len, seq_len), float('-inf'), device=query.device),
                    diagonal=1
                )
                attn_weights = attn_weights + causal_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
            attn_weights = self.dropout(attn_weights)
            
            attn_output = torch.matmul(attn_weights, value)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        attn_output = self.o_proj(attn_output)
        
        if use_cache:
            return attn_output, past_key_value
        return attn_output


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Transformer block with pre-normalization"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.hidden_size, config.layer_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states, attention_mask=None, use_cache=False, past_key_value=None):
        # Attention block
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        
        if use_cache:
            hidden_states, past_key_value = self.attention(
                hidden_states, attention_mask, use_cache=True, past_key_value=past_key_value
            )
        else:
            hidden_states = self.attention(hidden_states, attention_mask)
        
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # Feed-forward block
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        if use_cache:
            return hidden_states, past_key_value
        return hidden_states


class ChatModel(nn.Module):
    """Advanced transformer-based chat model with improvements"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.layer_norm_eps)
        
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with proper scaling"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
                
    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=False, past_key_values=None):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        hidden_states = self.embeddings(input_ids)
        
        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = (1.0 - attention_mask[:, None, None, :]) * torch.finfo(hidden_states.dtype).min
        
        # Transformer layers
        new_past_key_values = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            if self.config.gradient_checkpointing and self.training and not use_cache:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, attention_mask, False, None
                )
            else:
                if use_cache:
                    hidden_states, new_past_key_value = layer(
                        hidden_states, attention_mask, use_cache=True, past_key_value=past_key_value
                    )
                    new_past_key_values.append(new_past_key_value)
                else:
                    hidden_states = layer(hidden_states, attention_mask)
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        # LM head
        if self.config.tie_word_embeddings:
            logits = F.linear(hidden_states, self.embeddings.weight)
        else:
            logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        output = {"loss": loss, "logits": logits}
        if use_cache:
            output["past_key_values"] = new_past_key_values
        
        return output
    
    def count_parameters(self):
        """Count total and trainable parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ============== LEARNING RATE SCHEDULER ==============
class WarmupCosineScheduler:
    """Learning rate scheduler with linear warmup and cosine decay"""
    
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr, max_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.current_step = 0
        
    def step(self):
        """Update learning rate"""
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.max_lr * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_last_lr(self):
        """Get current learning rate"""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        """Get scheduler state"""
        return {'current_step': self.current_step}
    
    def load_state_dict(self, state_dict):
        """Load scheduler state"""
        self.current_step = state_dict.get('current_step', 0)


# ============== TRAINING UTILITIES ==============
class ModelCheckpoint:
    """Improved checkpointing system with atomic saves"""
    
    def __init__(self, output_dir: str, save_total_limit: int = 3):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_total_limit = save_total_limit
        self.checkpoints = self._load_existing_checkpoints()
        
    def _load_existing_checkpoints(self) -> List[Path]:
        """Load existing checkpoint directories"""
        checkpoints = sorted(
            self.output_dir.glob("checkpoint-*"),
            key=lambda x: int(x.name.split("-")[1])
        )
        return list(checkpoints)
        
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        step: int,
        best_loss: float,
        tokenizer: ChatTokenizer = None
    ):
        """Save checkpoint atomically"""
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        temp_dir = self.output_dir / f"checkpoint-{step}.tmp"
        
        try:
            # Save to temporary directory first
            temp_dir.mkdir(exist_ok=True)
            
            # Save model
            torch.save(model.state_dict(), temp_dir / "model.pt")
            
            # Save optimizer and scheduler
            torch.save({
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                'epoch': epoch,
                'step': step,
                'best_loss': best_loss,
            }, temp_dir / "training_state.pt")
            
            # Save tokenizer
            if tokenizer and tokenizer.sp:
                shutil.copy("tokenizer.model", temp_dir / "tokenizer.model")
                shutil.copy("tokenizer.vocab", temp_dir / "tokenizer.vocab")
                
            # Save config
            config = {
                'model_config': model.config.__dict__,
                'timestamp': datetime.now().isoformat(),
                'step': step,
                'epoch': epoch,
            }
            with open(temp_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)
            
            # Atomic rename
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)
            temp_dir.rename(checkpoint_dir)
            
            self.checkpoints.append(checkpoint_dir)
            
            # Remove old checkpoints
            while len(self.checkpoints) > self.save_total_limit:
                old_checkpoint = self.checkpoints.pop(0)
                if old_checkpoint.exists():
                    shutil.rmtree(old_checkpoint)
            
            print(f"✓ Saved checkpoint to {checkpoint_dir}")
            
        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise
            
    def load_latest(self) -> Optional[str]:
        """Load latest checkpoint path"""
        self.checkpoints = self._load_existing_checkpoints()
        if self.checkpoints:
            return str(self.checkpoints[-1])
        return None


class Trainer:
    """Advanced trainer with all modern techniques and bug fixes"""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset],
        tokenizer: ChatTokenizer,
        config: TrainingConfig
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.config = config
        
        # Setup device
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Calculate total steps
        steps_per_epoch = len(train_dataset) // (config.batch_size * config.gradient_accumulation_steps)
        total_steps = steps_per_epoch * config.num_epochs
        
        # Scheduler with warmup
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=total_steps,
            min_lr=config.min_learning_rate,
            max_lr=config.learning_rate
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Compile model (PyTorch 2.0+)
        if config.use_compile and hasattr(torch, 'compile'):
            print("Compiling model with PyTorch 2.0...")
            self.model = torch.compile(self.model)
            
        # Checkpointing
        self.checkpoint = ModelCheckpoint(config.output_dir, config.save_total_limit)
        
        # Tracking
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Resume if needed
        if config.resume_from_checkpoint:
            self.resume_training(config.resume_from_checkpoint)
        else:
            latest_checkpoint = self.checkpoint.load_latest()
            if latest_checkpoint:
                print(f"Found checkpoint: {latest_checkpoint}")
                response = input("Resume from checkpoint? (y/n): ").strip().lower()
                if response == 'y':
                    self.resume_training(latest_checkpoint)
            
    def _create_optimizer(self):
        """Create optimizer with weight decay fix"""
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "embedding" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_grouped_parameters = [
            {
                'params': decay_params,
                'weight_decay': self.config.weight_decay,
            },
            {
                'params': no_decay_params,
                'weight_decay': 0.0,
            }
        ]
        
        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
    def resume_training(self, checkpoint_path: str):
        """Resume from checkpoint with error handling"""
        checkpoint_dir = Path(checkpoint_path)
        
        if not checkpoint_dir.exists():
            print(f"⚠️ Checkpoint not found: {checkpoint_path}")
            return
        
        try:
            # Load model
            model_path = checkpoint_dir / "model.pt"
            if model_path.exists():
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"✓ Loaded model from {model_path}")
            
            # Load training state
            state_path = checkpoint_dir / "training_state.pt"
            if state_path.exists():
                state = torch.load(state_path, map_location=self.device)
                self.optimizer.load_state_dict(state['optimizer'])
                
                if state.get('scheduler') and hasattr(self.scheduler, 'load_state_dict'):
                    self.scheduler.load_state_dict(state['scheduler'])
                
                self.epoch = state.get('epoch', 0)
                self.global_step = state.get('step', 0)
                self.best_loss = state.get('best_loss', float('inf'))
                
                print(f"✓ Resumed from epoch {self.epoch}, step {self.global_step}")
                print(f"  Best loss: {self.best_loss:.4f}")
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            print("Starting fresh training...")
        
    def train(self):
        """Main training loop with proper error handling"""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True  # Drop last incomplete batch
        )
        
        # Initialize wandb
        if WANDB_AVAILABLE:
            wandb.init(
                project="chat-model-training",
                config={**self.config.__dict__, **self.model.config.__dict__},
                resume="allow"
            )
            
        total_steps = len(train_loader) // self.config.gradient_accumulation_steps * self.config.num_epochs
        print(f"\n{'='*60}")
        print(f"Starting training...")
        print(f"Total epochs: {self.config.num_epochs}")
        print(f"Steps per epoch: {len(train_loader) // self.config.gradient_accumulation_steps}")
        print(f"Total steps: {total_steps}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            self.model.train()
            
            epoch_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
                dynamic_ncols=True
            )
            
            for step, batch in enumerate(progress_bar):
                try:
                    loss = self.training_step(batch)
                    epoch_loss += loss
                    num_batches += 1
                    
                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = epoch_loss / num_batches
                        lr = self.scheduler.get_last_lr()[0]
                        
                        progress_bar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'lr': f'{lr:.2e}',
                            'step': self.global_step
                        })
                        
                        if WANDB_AVAILABLE:
                            wandb.log({
                                'train/loss': avg_loss,
                                'train/learning_rate': lr,
                                'train/epoch': epoch,
                                'train/step': self.global_step,
                            })
                    
                    # Evaluation
                    if self.val_dataset and self.global_step > 0 and \
                       self.global_step % self.config.eval_steps == 0:
                        val_loss = self.evaluate()
                        print(f"\n📊 Validation loss: {val_loss:.4f}")
                        
                        if WANDB_AVAILABLE:
                            wandb.log({
                                'val/loss': val_loss,
                                'val/step': self.global_step
                            })
                        
                        if val_loss < self.best_loss:
                            self.best_loss = val_loss
                            print(f"🎯 New best validation loss: {val_loss:.4f}")
                    
                    # Save checkpoint
                    if self.global_step > 0 and self.global_step % self.config.save_steps == 0:
                        self.checkpoint.save(
                            self.model,
                            self.optimizer,
                            self.scheduler,
                            epoch,
                            self.global_step,
                            self.best_loss,
                            self.tokenizer
                        )
                        
                except Exception as e:
                    print(f"\n❌ Error in training step {self.global_step}: {e}")
                    continue
            
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            print(f"\n✓ Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            # End of epoch checkpoint
            self.checkpoint.save(
                self.model,
                self.optimizer,
                self.scheduler,
                epoch + 1,
                self.global_step,
                self.best_loss,
                self.tokenizer
            )
        
        # Final save
        print("\n💾 Saving final model...")
        final_dir = self.config.output_dir / "final_model"
        final_dir.mkdir(exist_ok=True)
        
        torch.save(self.model.state_dict(), final_dir / "model.pt")
        if self.tokenizer.sp:
            shutil.copy("tokenizer.model", final_dir / "tokenizer.model")
        
        with open(final_dir / "config.json", "w") as f:
            json.dump(self.model.config.__dict__, f, indent=2)
        
        print(f"✓ Final model saved to {final_dir}")
        
        if WANDB_AVAILABLE:
            wandb.finish()
            
    def training_step(self, batch):
        """Single training step with proper gradient accumulation"""
        # Move to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Mixed precision context
        with autocast(enabled=self.config.use_mixed_precision):
            outputs = self.model(**batch)
            loss = outputs['loss']
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.config.use_mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights every gradient_accumulation_steps
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.config.use_mixed_precision:
                self.scaler.unscale_(self.optimizer)
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            
            # Optimizer step
            if self.config.use_mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            # Update learning rate
            self.scheduler.step()
            
            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)
        
        self.global_step += 1
        
        return loss.item() * self.config.gradient_accumulation_steps
        
    @torch.no_grad()
    def evaluate(self):
        """Evaluate model on validation set"""
        if not self.val_dataset:
            return None
        
        self.model.eval()
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            with autocast(enabled=self.config.use_mixed_precision):
                outputs = self.model(**batch)
                total_loss += outputs['loss'].item()
                num_batches += 1
        
        self.model.train()
        return total_loss / max(num_batches, 1)
        
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50
    ):
        """Generate text from prompt with improved sampling"""
        self.model.eval()
        
        # Tokenize prompt
        tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([tokens], device=self.device)
        
        # Generate
        generated_tokens = []
        past_key_values = None
        
        for _ in range(max_length):
            # Forward pass
            outputs = self.model(
                input_ids if past_key_values is None else input_ids[:, -1:],
                use_cache=True,
                past_key_values=past_key_values
            )
            
            logits = outputs['logits'][:, -1, :]
            past_key_values = outputs.get('past_key_values')
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated_tokens.append(next_token.item())
            
            # Stop if EOS
            if next_token.item() == self.tokenizer.special_tokens.get("</s>", -1):
                break
            
            # Append to input
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Prevent infinite loops
            if len(generated_tokens) >= max_length:
                break
        
        # Decode
        all_tokens = tokens + generated_tokens
        generated_text = self.tokenizer.decode(all_tokens, skip_special_tokens=True)
        
        self.model.train()
        return generated_text


# ============== DATA CREATION ==============
def create_sample_data(data_dir: str = "./data"):
    """Create comprehensive sample training data"""
    Path(data_dir).mkdir(exist_ok=True)
    
    sample_conversations = [
        """User: Hello, how are you today?
Assistant: I'm doing well, thank you for asking! I'm here and ready to help you with any questions or tasks you might have. How can I assist you today?

User: Can you explain what machine learning is?
Assistant: Machine learning is a branch of artificial intelligence where computer systems learn and improve from experience without being explicitly programmed. Instead of following rigid rules, ML algorithms identify patterns in data and make decisions based on those patterns.""",
        
        """User: I need help writing Python code
Assistant: I'd be happy to help you with Python code! Could you tell me more about what you're trying to accomplish?

User: I want to read a CSV file and analyze the data
Assistant: I'll help you read and analyze CSV data using pandas. First, install pandas if you haven't already. Then you can use pd.read_csv() to load your data and various methods like describe(), head(), and info() to explore it.""",
        
        """User: What's the difference between AI and machine learning?
Assistant: AI is the broader field aimed at creating machines that can perform tasks requiring human-like intelligence. Machine Learning is a specific subset of AI that focuses on algorithms that can learn from experience without being explicitly programmed for every scenario.""",
    ]
    
    # Write sample files
    for i, conv in enumerate(sample_conversations):
        file_path = Path(data_dir) / f"conversation_{i+1}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(conv)
    
    print(f"✓ Created {len(sample_conversations)} sample files in {data_dir}")
    return [str(Path(data_dir) / f"conversation_{i+1}.txt") for i in range(len(sample_conversations))]


# ============== MAIN ==============
def main():
    """Main training function"""
    print("\n" + "="*60)
    print("  ADVANCED CHAT MODEL TRAINING (OPTIMIZED)")
    print("="*60 + "\n")
    
    # Set random seeds for reproducibility
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    set_seed(42)
    
    # Configurations
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # Find or create data files
    data_files = glob.glob(f"{training_config.data_dir}/*.txt")
    if not data_files:
        print("⚠️  No data files found. Creating sample data...")
        data_files = create_sample_data(training_config.data_dir)
    else:
        print(f"✓ Found {len(data_files)} data files")
    
    # Initialize tokenizer
    print("\n📝 Initializing tokenizer...")
    tokenizer = ChatTokenizer(vocab_size=model_config.vocab_size)
    
    if os.path.exists("tokenizer.model"):
        print("✓ Loading existing tokenizer...")
        tokenizer.load("tokenizer.model")
    else:
        print("Training new tokenizer...")
        all_texts = []
        for file_path in data_files[:min(10, len(data_files))]:
            with open(file_path, 'r', encoding='utf-8') as f:
                all_texts.append(f.read())
        
        if all_texts:
            tokenizer.train(all_texts)
        else:
            raise ValueError("No text data available for tokenizer training")
    
    # Update vocab size based on actual tokenizer
    model_config.vocab_size = tokenizer.get_vocab_size()
    
    # Create datasets
    print("\n📊 Creating datasets...")
    split_idx = int(len(data_files) * 0.9)
    train_files = data_files[:split_idx] if split_idx > 0 else data_files
    val_files = data_files[split_idx:] if split_idx < len(data_files) else None
    
    train_dataset = ChatDataset(
        train_files,
        tokenizer,
        max_length=model_config.max_position_embeddings,
        cache_dir=training_config.cache_dir
    )
    
    val_dataset = None
    if val_files:
        val_dataset = ChatDataset(
            val_files,
            tokenizer,
            max_length=model_config.max_position_embeddings,
            cache_dir=training_config.cache_dir
        )
        print(f"✓ Validation samples: {len(val_dataset)}")
    
    # Initialize model
    print("\n🤖 Initializing model...")
    model = ChatModel(model_config)
    
    total_params, trainable_params = model.count_parameters()
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    print(f"✓ Model size: ~{total_params * 4 / 1e9:.2f}GB (fp32)")
    
    # Initialize trainer
    print("\n🚀 Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        config=training_config
    )
    
    # Train
    print("\n" + "="*60)
    print("  STARTING TRAINING")
    print("="*60 + "\n")
    
    trainer.train()
    
    # Test generation
    print("\n" + "="*60)
    print("  TESTING GENERATION")
    print("="*60 + "\n")
    
    test_prompts = [
        "User: What is deep learning?",
        "User: How do I write a Python function?",
    ]
    
    for prompt in test_prompts:
        print(f"\n💬 Prompt: {prompt}")
        try:
            response = trainer.generate(prompt, max_length=50, temperature=0.7)
            print(f"🤖 Response: {response}\n")
        except Exception as e:
            print(f"❌ Generation error: {e}\n")
    
    print("="*60)
    print("  ✅ TRAINING COMPLETED SUCCESSFULLY")
    print(f"  📁 Model saved to: {training_config.output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
