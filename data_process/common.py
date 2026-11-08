"""Shared helpers for the cross-format conversions (WebQSP <-> text RAG format).

Everything here is dependency-light: numpy, pyarrow and (optionally) tokenizers.
Paths are resolved relative to the repository root so the scripts can be run
from anywhere: ``python data_process/<script>.py ...``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Iterator

REPO = Path(__file__).resolve().parents[1]

# ---- canonical locations -------------------------------------------------
WEBQSP_DIR = REPO / "data" / "data" / "webqsp"            # dense NSM subgraphs (GNN-RAG's pool)
NAMES_DIR = REPO / "data" / "raw" / "webqsp_names"         # fetched name sources + mid2name.tsv
WEBQSP_TEXT_DIR = REPO / "data" / "processed" / "webqsp_text"
QASPER_CHUNKS = REPO / "data" / "processed" / "chunks" / "size512_ov102_soft" / "chunks.parquet"
QASPER_QA_GOLD = REPO / "data" / "processed" / "chunks" / "size512_ov102_soft" / "qa_with_gold.jsonl"
QASPER_GNN_DIR = REPO / "data" / "processed" / "qasper_gnn"
BGE_TOKENIZER = REPO / "data" / "raw" / "bge_tok" / "tokenizer.json"

# The dense baseline's encoder (baselines/member2_dense_rag/dense_rag/core.py: EncoderConfig).
BGE_MODEL = "BAAI/bge-small-en-v1.5"
BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BGE_MAX_LENGTH = 512

MID_RE = re.compile(r"^[mg]\.[0-9a-z_]+$")


def is_mid(s: str) -> bool:
    return bool(MID_RE.match(s))


# ---- hashing / io --------------------------------------------------------
def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def config_hash(config: dict) -> str:
    """12-hex hash of a JSON-serialisable config (same length as the QASPER config_hash)."""
    blob = json.dumps(config, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def iter_jsonl(path) -> Iterator[dict]:
    """Stream a JSON-lines file (WebQSP's *.json files are JSON lines)."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl(path) -> list[dict]:
    return list(iter_jsonl(path))


def write_jsonl(path, rows: Iterable[dict], ascii_only: bool = False) -> int:
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=ascii_only, allow_nan=False) + "\n")
            n += 1
    return n


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")


def read_lines(path) -> list[str]:
    """One item per line, only the trailing newline stripped (names may contain spaces)."""
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n").rstrip("\r") for line in f]


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---- BGE token counting (same convention as the repo: no special tokens) --
class TokenCounter:
    def __init__(self, path=BGE_TOKENIZER):
        from tokenizers import Tokenizer  # local, no network

        if not os.path.exists(path):
            raise FileNotFoundError(f"BGE tokenizer not found: {path}")
        self.tok = Tokenizer.from_file(str(path))
        self.tok.no_truncation()
        self.tok.no_padding()
        self.backend = f"tokenizers-file:{Path(path).name}"

    def count_batch(self, texts: list[str]) -> list[int]:
        if not texts:
            return []
        return [len(e.ids) for e in self.tok.encode_batch(texts, add_special_tokens=False)]

    def count(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False).ids)

    def truncate(self, text: str, max_tokens: int) -> str:
        """Cut ``text`` so that it has at most ``max_tokens`` tokens (character-offset based)."""
        enc = self.tok.encode(text, add_special_tokens=False)
        if len(enc.ids) <= max_tokens:
            return text
        end = enc.offsets[max_tokens - 1][1]
        return text[:end].rstrip()
