import os
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from domains.tokenization.domain.tokenizer import Tokenizer

class ChatDataset(Dataset):
    """
    Memory-efficient chat dataset that caches tokenized data to disk.
    This avoids re-tokenizing the same data every epoch.
    """

    def __init__(
        self,
        data_files: List[str],
        tokenizer: Tokenizer,
        max_length: int = 2048,
        cache_dir: Optional[str] = ".cache/data"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.data_files = [Path(f) for f in data_files if os.path.exists(f)]
        if not self.data_files:
            raise ValueError("No valid data files found")

        self.cached_samples = []
        self._prepare_dataset()

    def _prepare_dataset(self):
        """
        Processes and caches all conversations in a memory-efficient way.
        It streams files line by line and processes conversations separated by double newlines.
        """
        print(f"🔎 Preparing dataset from {len(self.data_files)} files...")
        for file_path in tqdm(self.data_files, desc="Processing files"):
            with open(file_path, 'r', encoding='utf-8') as f:
                buffer = []
                conv_idx = 0
                for line in f:
                    if line.strip() == "": # Conversation separator
                        if buffer:
                            conv_text = "".join(buffer).strip()
                            if 'User:' in conv_text and 'Assistant:' in conv_text:
                                self._process_and_cache_conversation(conv_text, file_path, conv_idx)
                                conv_idx += 1
                            buffer = []
                    else:
                        buffer.append(line)

                # Process the last conversation in the file if it doesn't end with a newline
                if buffer:
                    conv_text = "".join(buffer).strip()
                    if 'User:' in conv_text and 'Assistant:' in conv_text:
                        self._process_and_cache_conversation(conv_text, file_path, conv_idx)

        if not self.cached_samples:
             raise ValueError("No valid conversations found after processing files.")

        print(f"✓ Dataset ready. Found {len(self.cached_samples)} cached/processed samples.")

    def _process_and_cache_conversation(self, conv_text: str, file_path: Path, conv_idx: int):
        """
        Hashes, tokenizes, and caches a single conversation.
        """
        conv_hash = hashlib.md5(f"{file_path.stem}_{conv_idx}_{conv_text}".encode()).hexdigest()
        cache_file = self.cache_dir / f"{conv_hash}.pt"

        if cache_file.exists():
            self.cached_samples.append(cache_file)
        else:
            tokenized_data = self._tokenize_conversation(conv_text)
            if tokenized_data:
                try:
                    torch.save(tokenized_data, cache_file)
                    self.cached_samples.append(cache_file)
                except Exception as e:
                    print(f"Error saving cache file for conversation {conv_idx} in {file_path}: {e}")

    def _parse_conversation(self, text: str) -> List[Dict[str, str]]:
        """
        Optimized parsing of conversation text into messages.
        """
        messages = []
        lines = text.split('\n')
        current_role = None
        current_content = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            role_marker = None
            if line.startswith("User:"):
                role_marker = "User:"
                current_role = 'user'
            elif line.startswith("Assistant:"):
                role_marker = "Assistant:"
                current_role = 'assistant'

            if role_marker:
                if current_content: # Append previous message
                    messages.append({'role': messages[-1]['role'], 'content': ' '.join(current_content)})
                    current_content = []
                messages.append({'role': current_role, 'content': ''}) # Start new message
                current_content.append(line[len(role_marker):].strip())
            elif messages:
                current_content.append(line)

        if messages and current_content:
            messages[-1]['content'] = ' '.join(current_content)

        return messages

    def _tokenize_conversation(self, conv_text: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Tokenizes a single conversation and prepares it for the model.
        """
        messages = self._parse_conversation(conv_text)
        if not messages:
            return None

        text_parts = []
        for msg in messages:
            role_tag = f"<{msg['role']}>"
            end_tag = f"</{msg['role']}>"
            text_parts.append(f"{role_tag}{msg['content']}{end_tag}")

        text = "".join(text_parts)

        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
        except Exception:
            return None # Skip faulty data

        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        pad_token = self.tokenizer.special_tokens["<pad>"]
        attention_mask = [1] * len(tokens)

        if len(tokens) < self.max_length:
            padding_length = self.max_length - len(tokens)
            tokens.extend([pad_token] * padding_length)
            attention_mask.extend([0] * padding_length)

        labels = tokens[1:] + [pad_token]
        labels = [label if mask == 1 else -100 for label, mask in zip(labels, attention_mask)]

        return {
            'input_ids': torch.tensor(tokens[:-1], dtype=torch.long),
            'labels': torch.tensor(labels[:-1], dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask[:-1], dtype=torch.long)
        }

    def __len__(self):
        return len(self.cached_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Loads a pre-tokenized sample from the cache.
        """
        cache_file = self.cached_samples[idx]
        try:
            return torch.load(cache_file)
        except (EOFError, FileNotFoundError) as e:
            print(f"⚠️ Corrupted or missing cache file: {cache_file}. Skipping. Error: {e}")
            # Return a dummy sample to avoid crashing the training loop
            return self.__getitem__((idx + 1) % len(self))