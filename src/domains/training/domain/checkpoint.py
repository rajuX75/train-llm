import os
import json
import shutil
from pathlib import Path
from typing import List, Optional, Any
from datetime import datetime
import torch
import torch.nn as nn
from domains.tokenization.domain.tokenizer import Tokenizer

class ModelCheckpoint:
    """Improved checkpointing system with atomic saves."""

    def __init__(self, output_dir: str, save_total_limit: int = 3):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_total_limit = save_total_limit
        self.checkpoints = self._load_existing_checkpoints()

    def _load_existing_checkpoints(self) -> List[Path]:
        """Load and sort existing checkpoint directories by step number."""
        checkpoints = sorted(
            [p for p in self.output_dir.glob("checkpoint-*") if p.is_dir()],
            key=lambda x: int(x.name.split("-")[1])
        )
        return checkpoints

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        step: int,
        best_loss: float,
        tokenizer: Optional[Tokenizer] = None
    ):
        """Save a training checkpoint atomically."""
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        temp_dir = self.output_dir / f"checkpoint-{step}.tmp"

        try:
            temp_dir.mkdir(exist_ok=True)

            torch.save(model.state_dict(), temp_dir / "model.pt")

            torch.save({
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                'epoch': epoch,
                'step': step,
                'best_loss': best_loss,
            }, temp_dir / "training_state.pt")

            if tokenizer and tokenizer.sp:
                # This assumes the tokenizer model is in the root directory.
                # A better approach would be to have the tokenizer manage its own persistence.
                if os.path.exists("tokenizer.model"):
                    shutil.copy("tokenizer.model", temp_dir / "tokenizer.model")
                if os.path.exists("tokenizer.vocab"):
                    shutil.copy("tokenizer.vocab", temp_dir / "tokenizer.vocab")

            config = {
                'model_config': model.config.__dict__,
                'timestamp': datetime.now().isoformat(),
                'step': step,
                'epoch': epoch,
            }
            with open(temp_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)

            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)
            temp_dir.rename(checkpoint_dir)

            self.checkpoints.append(checkpoint_dir)
            self._manage_checkpoints()

            print(f"✓ Saved checkpoint to {checkpoint_dir}")

        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise

    def _manage_checkpoints(self):
        """Remove old checkpoints to respect save_total_limit."""
        if self.save_total_limit > 0 and len(self.checkpoints) > self.save_total_limit:
            num_to_delete = len(self.checkpoints) - self.save_total_limit
            for _ in range(num_to_delete):
                old_checkpoint = self.checkpoints.pop(0)
                if old_checkpoint.exists():
                    shutil.rmtree(old_checkpoint)
                    print(f"🗑️ Removed old checkpoint: {old_checkpoint}")

    def load_latest(self) -> Optional[str]:
        """Get the path to the latest checkpoint."""
        self.checkpoints = self._load_existing_checkpoints()
        if self.checkpoints:
            return str(self.checkpoints[-1])
        return None