"""Token counting.

The chunk size budget must be measured with the **same tokenizer as the embedding model**
(Member 2 uses a BGE checkpoint, i.e. an XLM-R SentencePiece vocabulary). `tiktoken`'s
cl100k vocabulary drifts substantially from XLM-R on academic text, which silently
produces over-long inputs that get truncated at the 512-token limit.

Backends, in order of preference:

1. ``tokenizers.Tokenizer.from_file(".../tokenizer.json")`` — one local file, no
   HuggingFace cache, no runtime network. This is what the pipeline uses.
2. ``transformers.AutoTokenizer.from_pretrained(name)`` — convenience fallback.
3. Whitespace counting — last resort, flagged loudly via ``backend``.
"""

from __future__ import annotations

import os
from functools import lru_cache

DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_TOKENIZER_FILE = "data/raw/bge_tok/tokenizer.json"


class TokenCounter:
    def __init__(self, source: str = DEFAULT_TOKENIZER_FILE, fallback: bool = True) -> None:
        self.source = source
        self._tok = None
        self._kind = "whitespace"
        self.backend = "fallback:whitespace"
        self.error = ""
        try:
            if source.endswith(".json") and os.path.exists(source):
                from tokenizers import Tokenizer

                self._tok = Tokenizer.from_file(source)
                self._kind = "tokenizers"
                self.backend = f"tokenizers-file:{os.path.basename(source)}"
            else:
                from transformers import AutoTokenizer

                self._tok = AutoTokenizer.from_pretrained(source, use_fast=True)
                self._kind = "hf"
                self.backend = f"hf:{source}"
        except Exception as exc:  # pragma: no cover - depends on local files
            if not fallback:
                raise
            self.error = f"{type(exc).__name__}: {exc}"[:300]

    # -- counting --------------------------------------------------------
    def count(self, text: str) -> int:
        if self._tok is None:
            return len(text.split())
        if self._kind == "tokenizers":
            return len(self._tok.encode(text, add_special_tokens=False).ids)
        return len(self._tok.encode(text, add_special_tokens=False))

    def count_batch(self, texts: list[str], batch_size: int = 256) -> list[int]:
        if self._tok is None:
            return [len(t.split()) for t in texts]
        out: list[int] = []
        if self._kind == "tokenizers":
            for i in range(0, len(texts), batch_size):
                encs = self._tok.encode_batch(texts[i : i + batch_size], add_special_tokens=False)
                out.extend(len(e.ids) for e in encs)
            return out
        for i in range(0, len(texts), batch_size):
            enc = self._tok(texts[i : i + batch_size], add_special_tokens=False)["input_ids"]
            out.extend(len(ids) for ids in enc)
        return out

    @property
    def max_length(self) -> int | None:
        if self._tok is None or self._kind != "hf":
            return None
        return getattr(self._tok, "model_max_length", None)

    def char_offsets(self, text: str) -> list[tuple[int, int]]:
        """Char span of every token. Used to hard-split a sentence that alone exceeds
        the chunk budget without cutting inside a token."""
        if self._tok is None:
            return [(i, i + 1) for i in range(len(text))]
        if self._kind == "tokenizers":
            return list(self._tok.encode(text, add_special_tokens=False).offsets)
        enc = self._tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return list(enc["offset_mapping"])


@lru_cache(maxsize=4)
def get_counter(source: str = DEFAULT_TOKENIZER_FILE) -> TokenCounter:
    return TokenCounter(source)
