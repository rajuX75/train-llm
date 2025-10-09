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
    A memory-efficient dataset that streams data directly from text files,
    tokenizes on-the-fly, and caches the tokenized samples to disk.
    This "lazy loading" approach avoids high memory usage and long startup times
    by not pre-processing the entire dataset at once.
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
            raise ValueError("No valid data files provided.")

        # Lightweight index: (file_path, conversation_start_line)
        self.sample_index = []
        self._build_index()

    def _build_index(self):
        """
        Scans through data files to build a lightweight index of sample locations.
        This index stores the file path and the starting line number of each conversation.
        """
        print("Building sample index...")
        for file_path in tqdm(self.data_files, desc="Indexing files"):
            with open(file_path, 'r', encoding='utf-8') as f:
                line_num = 0
                in_conversation = False
                for line in f:
                    # A conversation starts with "User:" and is not just whitespace
                    if line.strip().startswith("User:") and not in_conversation:
                        self.sample_index.append((file_path, line_num))
                        in_conversation = True
                    # A blank line indicates the end of a conversation
                    elif not line.strip() and in_conversation:
                        in_conversation = False
                    line_num += 1
        if not self.sample_index:
            raise ValueError("No valid conversations found in the data files.")
        print(f"✓ Index built. Found {len(self.sample_index)} conversations.")

    def _get_conversation_text(self, file_path: Path, start_line: int) -> str:
        """
        Reads a single conversation from a file starting at a specific line.
        """
        conv_lines = []
        with open(file_path, 'r', encoding='utf-8') as f:
            # Fast-forward to the starting line
            for _ in range(start_line):
                next(f)
            # Read lines until a blank line or end of file
            for line in f:
                if not line.strip():
                    break
                conv_lines.append(line)
        return "".join(conv_lines).strip()

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
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves, processes, and caches a single training sample.
        This method is the core of the lazy-loading mechanism.
        """
        file_path, start_line = self.sample_index[idx]

        # Use a unique hash for the cache key based on file and position
        conv_id = f"{file_path.stem}_{start_line}"
        conv_hash = hashlib.md5(conv_id.encode()).hexdigest()
        cache_file = self.cache_dir / f"{conv_hash}.pt"

        # 1. Check cache first
        if cache_file.exists():
            try:
                return torch.load(cache_file)
            except (EOFError, torch.serialization.DeserializationStorageError) as e:
                print(f"⚠️ Corrupted cache file: {cache_file}. Re-processing. Error: {e}")

        # 2. If not in cache, read, process, and cache it
        conv_text = self._get_conversation_text(file_path, start_line)
        if 'User:' not in conv_text or 'Assistant:' not in conv_text:
             print(f"⚠️ Skipping invalid sample at {file_path}:{start_line}")
             return self.__getitem__((idx + 1) % len(self)) # Return next sample

        tokenized_data = self._tokenize_conversation(conv_text)

        if not tokenized_data:
            print(f"⚠️ Failed to tokenize sample at {file_path}:{start_line}. Skipping.")
            return self.__getitem__((idx + 1) % len(self)) # Return next sample

        try:
            torch.save(tokenized_data, cache_file)
        except Exception as e:
            print(f"❌ Error saving cache file {cache_file}: {e}")

        return tokenized_data