import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset
from domains.tokenization.domain.tokenizer import Tokenizer

class ChatDataset(Dataset):
    """Memory-efficient chat dataset with lazy loading."""

    def __init__(
        self,
        data_files: List[str],
        tokenizer: Tokenizer,
        max_length: int = 2048,
        cache_dir: Optional[str] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.data_files = [f for f in data_files if os.path.exists(f)]
        if not self.data_files:
            raise ValueError("No valid data files found")

        self.conversation_index = self._build_index()
        print(f"✓ Indexed {len(self.conversation_index)} conversations from {len(self.data_files)} files")

    def _build_index(self) -> List[Tuple[str, str]]:
        """Build index of conversations (file_path, conversation_text)."""
        index = []
        for file_path in self.data_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            conversations = content.split('\n\n')
            for conv in conversations:
                if conv.strip() and 'User:' in conv and 'Assistant:' in conv:
                    index.append((file_path, conv))
        return index

    def _parse_conversation(self, text: str) -> List[Dict[str, str]]:
        """Parse conversation text into messages."""
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
                    messages.append({'role': current_role, 'content': ' '.join(current_content)})
                current_role = 'user'
                current_content = [line[5:].strip()]
            elif line.startswith("Assistant:"):
                if current_role and current_content:
                    messages.append({'role': current_role, 'content': ' '.join(current_content)})
                current_role = 'assistant'
                current_content = [line[10:].strip()]
            elif current_role:
                current_content.append(line)

        if current_role and current_content:
            messages.append({'role': current_role, 'content': ' '.join(current_content)})

        return messages

    def __len__(self):
        return len(self.conversation_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single tokenized conversation."""
        file_path, conv_text = self.conversation_index[idx]
        messages = self._parse_conversation(conv_text)

        text = ""
        for msg in messages:
            if msg['role'] == 'user':
                text += f"<user>{msg['content']}</user>"
            else:
                text += f"<assistant>{msg['content']}</assistant>"

        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
        except Exception as e:
            print(f"⚠️ Tokenization error in file {file_path}: {e}")
            tokens = [self.tokenizer.special_tokens["<pad>"]] * self.max_length

        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        pad_token = self.tokenizer.special_tokens["<pad>"]
        attention_mask = [1] * len(tokens)

        if len(tokens) < self.max_length:
            padding_length = self.max_length - len(tokens)
            tokens.extend([pad_token] * padding_length)
            attention_mask.extend([0] * padding_length)

        labels = tokens[1:] + [pad_token]
        labels = [t if m == 1 else -100 for t, m in zip(labels, attention_mask)]

        return {
            'input_ids': torch.tensor(tokens[:-1], dtype=torch.long),
            'labels': torch.tensor(labels[:-1], dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask[:-1], dtype=torch.long)
        }