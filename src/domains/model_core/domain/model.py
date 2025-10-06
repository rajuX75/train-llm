import torch
import torch.nn as nn
import torch.nn.functional as F
from shared.config.training_config import ModelConfig
from domains.model_core.domain.layers import RMSNorm, MultiHeadAttention, FeedForward

class TransformerBlock(nn.Module):
    """Transformer block with pre-normalization."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.hidden_size, config.layer_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask=None, use_cache=False, past_key_value=None):
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

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        if use_cache:
            return hidden_states, past_key_value
        return hidden_states

class ChatModel(nn.Module):
    """Advanced transformer-based chat model."""

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
        """Initialize weights with proper scaling."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=False, past_key_values=None):
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embeddings(input_ids)

        if attention_mask is not None:
            attention_mask = (1.0 - attention_mask[:, None, None, :]) * torch.finfo(hidden_states.dtype).min

        new_past_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.config.gradient_checkpointing and self.training and not use_cache:
                hidden_states = torch.utils.checkpoint.checkpoint(layer, hidden_states, attention_mask, False, None)
            else:
                if use_cache:
                    hidden_states, new_past_key_value = layer(hidden_states, attention_mask, use_cache=True, past_key_value=past_key_value)
                    new_past_key_values.append(new_past_key_value)
                else:
                    hidden_states = layer(hidden_states, attention_mask)

        hidden_states = self.norm(hidden_states)

        if self.config.tie_word_embeddings:
            logits = F.linear(hidden_states, self.embeddings.weight)
        else:
            logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1), ignore_index=-100)

        output = {"loss": loss, "logits": logits}
        if use_cache:
            output["past_key_values"] = new_past_key_values

        return output

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable