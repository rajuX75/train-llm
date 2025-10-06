from abc import ABC, abstractmethod
from typing import List, Dict

class Tokenizer(ABC):
    """Abstract base class for a tokenizer (Port)."""

    @abstractmethod
    def train(self, texts: List[str], model_prefix: str = "tokenizer"):
        pass

    @abstractmethod
    def load(self, model_path: str):
        pass

    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        pass

    @abstractmethod
    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        pass

    @abstractmethod
    def get_vocab_size(self) -> int:
        pass

    @property
    @abstractmethod
    def special_tokens(self) -> Dict[str, int]:
        pass