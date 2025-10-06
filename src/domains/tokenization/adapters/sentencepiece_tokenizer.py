import os
from typing import List, Dict
import sentencepiece as spm
from domains.tokenization.domain.tokenizer import Tokenizer

class SentencePieceTokenizer(Tokenizer):
    """SentencePiece tokenizer implementation (Adapter)."""

    SPECIAL_TOKENS: Dict[str, int] = {
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
        self._special_tokens = self.SPECIAL_TOKENS.copy()

    def train(self, texts: List[str], model_prefix: str = "tokenizer"):
        if not texts:
            raise ValueError("Cannot train tokenizer on empty text list")

        temp_file = f"{model_prefix}_train_texts_tmp.txt"
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                for text in texts:
                    if text.strip():
                        f.write(text.strip() + "\n")

            spm.SentencePieceTrainer.train(
                input=temp_file,
                model_prefix=model_prefix,
                vocab_size=self.vocab_size - len(self._special_tokens),
                model_type="bpe",
                character_coverage=0.9995,
                num_threads=os.cpu_count() or 1,
                user_defined_symbols=list(self._special_tokens.keys()),
                pad_id=self._special_tokens["<pad>"],
                unk_id=self._special_tokens["<unk>"],
                bos_id=self._special_tokens["<s>"],
                eos_id=self._special_tokens["</s>"],
                max_sentence_length=16384,
            )

            self.sp = spm.SentencePieceProcessor()
            self.sp.load(f"{model_prefix}.model")
            print(f"✓ Tokenizer trained successfully with vocab size: {self.sp.get_piece_size()}")

        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def load(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Tokenizer model not found: {model_path}")
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        print(f"✓ Tokenizer loaded with vocab size: {self.sp.get_piece_size()}")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        if self.sp is None:
            raise RuntimeError("Tokenizer not initialized. Train or load a model first.")
        tokens = self.sp.encode_as_ids(text)
        if add_special_tokens:
            tokens = [self._special_tokens["<s>"]] + tokens + [self._special_tokens["</s>"]]
        return tokens

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        if self.sp is None:
            raise RuntimeError("Tokenizer not initialized")
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self._special_tokens.values()]
        return self.sp.decode_ids(tokens)

    def get_vocab_size(self) -> int:
        return self.sp.get_piece_size() if self.sp else self.vocab_size

    @property
    def special_tokens(self) -> Dict[str, int]:
        return self._special_tokens