import torch
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm
from typing import Optional, Dict
import warnings

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("Wandb not available, logging locally only")

from domains.model_core.domain.model import ChatModel
from domains.tokenization.domain.tokenizer import Tokenizer
from shared.config.training_config import TrainingConfig
from domains.training.domain.scheduler import WarmupCosineScheduler
from domains.training.domain.checkpoint import ModelCheckpoint

class TrainingService:
    """Service to handle the model training process."""

    def __init__(
        self,
        model: ChatModel,
        train_dataset: Dataset,
        config: TrainingConfig,
        tokenizer: Tokenizer,
        val_dataset: Optional[Dataset] = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.tokenizer = tokenizer

        self.device = torch.device(config.device)
        self.model.to(self.device)

        self.optimizer = self._create_optimizer()

        steps_per_epoch = len(train_dataset) // (config.batch_size * config.gradient_accumulation_steps)
        total_steps = steps_per_epoch * config.num_epochs

        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=total_steps,
            min_lr=config.min_learning_rate,
            max_lr=config.learning_rate
        )

        self.scaler = GradScaler() if config.use_mixed_precision else None
        self.checkpoint_handler = ModelCheckpoint(config.output_dir, config.save_total_limit)

        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')

        if config.use_compile and hasattr(torch, 'compile'):
            print("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

        self.resume_from_checkpoint()

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create an AdamW optimizer with weight decay separation."""
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
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8
        )

    def resume_from_checkpoint(self):
        """Resume training from the latest or a specified checkpoint."""
        checkpoint_path = self.config.resume_from_checkpoint or self.checkpoint_handler.load_latest()
        if not checkpoint_path:
            print("No checkpoint found, starting from scratch.")
            return

        print(f"Attempting to resume from checkpoint: {checkpoint_path}")
        try:
            state = torch.load(f"{checkpoint_path}/training_state.pt", map_location=self.device)
            self.model.load_state_dict(torch.load(f"{checkpoint_path}/model.pt", map_location=self.device))
            self.optimizer.load_state_dict(state['optimizer'])
            if state.get('scheduler'):
                self.scheduler.load_state_dict(state['scheduler'])

            self.global_step = state.get('step', 0)
            self.current_epoch = state.get('epoch', 0)
            self.best_val_loss = state.get('best_loss', float('inf'))
            print(f"✓ Resumed from epoch {self.current_epoch}, step {self.global_step}")
        except FileNotFoundError:
            print(f"⚠️ Checkpoint not found at {checkpoint_path}. Starting fresh.")
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}. Starting fresh.")

    def train(self):
        """Main training loop."""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True
        )

        if WANDB_AVAILABLE and self.config.wandb_enabled:
            wandb.init(project="chat-model-training", config={**self.config.__dict__, **self.model.config.__dict__})
            wandb.watch(self.model)

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            self.model.train()

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            for batch in progress_bar:
                loss = self._training_step(batch)

                if self.global_step % self.config.logging_steps == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({'loss': f'{loss:.4f}', 'lr': f'{lr:.2e}'})
                    if WANDB_AVAILABLE and self.config.wandb_enabled:
                        wandb.log({'train/loss': loss, 'train/learning_rate': lr, 'train/step': self.global_step})

                if self.val_dataset and self.global_step > 0 and self.global_step % self.config.eval_steps == 0:
                    self._evaluate()

                if self.global_step > 0 and self.global_step % self.config.save_steps == 0:
                    self.checkpoint_handler.save(self.model, self.optimizer, self.scheduler, epoch, self.global_step, self.best_val_loss, self.tokenizer)

            # End of epoch evaluation and checkpoint
            if self.val_dataset:
                self._evaluate()
            self.checkpoint_handler.save(self.model, self.optimizer, self.scheduler, epoch + 1, self.global_step, self.best_val_loss, self.tokenizer)

        if WANDB_AVAILABLE and self.config.wandb_enabled:
            wandb.finish()

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Performs a single training step including forward pass, backward pass, and optimizer step."""
        batch = {k: v.to(self.device) for k, v in batch.items()}

        with autocast(enabled=self.config.use_mixed_precision):
            outputs = self.model(**batch)
            loss = outputs['loss']
            loss = loss / self.config.gradient_accumulation_steps

        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        self.global_step += 1
        return loss.item() * self.config.gradient_accumulation_steps

    @torch.no_grad()
    def _evaluate(self):
        """Evaluates the model on the validation set."""
        self.model.eval()
        val_loader = DataLoader(self.val_dataset, batch_size=self.config.batch_size, num_workers=0)

        total_loss = 0.0
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with autocast(enabled=self.config.use_mixed_precision):
                outputs = self.model(**batch)
                total_loss += outputs['loss'].item()

        avg_loss = total_loss / len(val_loader)
        print(f"\n📊 Validation loss: {avg_loss:.4f}")
        if WANDB_AVAILABLE and self.config.wandb_enabled:
            wandb.log({'val/loss': avg_loss, 'val/step': self.global_step})

        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            print(f"🎯 New best validation loss: {avg_loss:.4f}. Saving best model.")
            # Consider saving a 'best_model' checkpoint here

        self.model.train()