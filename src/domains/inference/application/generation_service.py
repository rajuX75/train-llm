import torch
import torch.nn as nn
import torch.nn.functional as F
from domains.tokenization.domain.tokenizer import Tokenizer
from domains.model_core.domain.model import ChatModel

class GenerationService:
    """Service for generating text from a model."""

    def __init__(self, model: ChatModel, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> str:
        """Generate text from a prompt using sampling methods."""
        self.model.eval()

        tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([tokens], device=self.device)

        generated_tokens = []
        past_key_values = None

        for _ in range(max_length):
            outputs = self.model(
                input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                use_cache=True,
                past_key_values=past_key_values
            )

            logits = outputs['logits'][:, -1, :]
            past_key_values = outputs.get('past_key_values')

            # Temperature scaling
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

            # Sample the next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Stop if EOS token is generated
            if next_token.item() == self.tokenizer.special_tokens.get("</s>", -1):
                break

            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=-1)

        # Decode the full sequence
        full_sequence = self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)

        self.model.train() # Restore model to training mode
        return full_sequence