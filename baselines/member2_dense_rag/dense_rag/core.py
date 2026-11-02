"""Member 2: exact dense retrieval and a swappable generation boundary."""
from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import faiss
import numpy as np
from .data import load_chunks


def read_jsonl(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_chunks(chunks):
    if not chunks:
        raise ValueError("Empty chunk corpus")
    seen = set()
    for c in chunks:
        for key in ("chunk_id", "paper_id", "text"):
            if not isinstance(c.get(key), str) or not c[key].strip():
                raise ValueError(f"Chunk requires non-empty string: {key}")
        if c["chunk_id"] in seen:
            raise ValueError(f"Duplicate chunk_id: {c['chunk_id']}")
        if not isinstance(c.get("metadata", {}), dict):
            raise ValueError("metadata must be an object")
        seen.add(c["chunk_id"])


def unit_vectors(values):
    x = np.array(values, dtype="float32", order="C", copy=True)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Embeddings must be a finite matrix")
    if np.any(np.linalg.norm(x, axis=1) <= 0):
        raise ValueError("Zero embedding")
    faiss.normalize_L2(x)
    return x


@dataclass(frozen=True)
class EncoderConfig:
    model: str = "BAAI/bge-small-en-v1.5"
    revision: str = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    query_instruction: str = "Represent this sentence for searching relevant passages: "
    max_length: int = 512
    batch_size: int = 32


class BGEEncoder:
    def __init__(self, config=None, device="cpu"):
        from sentence_transformers import SentenceTransformer
        self.config = config or EncoderConfig()
        self.model = SentenceTransformer(self.config.model, revision=self.config.revision, device=device)
        self.model.max_seq_length = self.config.max_length
        self.device = device

    def encode(self, texts, query=False):
        if query:
            texts = [self.config.query_instruction + text for text in texts]
        return self.model.encode(texts, batch_size=self.config.batch_size,
                                 normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False)


class DenseRetriever:
    def __init__(self, encoder, chunks, index, manifest):
        self.encoder, self.chunks, self.index, self.manifest = encoder, chunks, index, manifest
        self.paper_rows = {}
        for i, chunk in enumerate(chunks):
            self.paper_rows.setdefault(chunk["paper_id"], []).append(i)
        self.paper_indexes = {}

    @property
    def config(self):
        return {
            "method": "dense",
            "corpus_sha256": self.manifest["source_sha256"],
            "data": self.manifest.get("data", {}),
            "parameters": {
                "encoder": dict(self.manifest["encoder"]),
                "index_type": self.manifest["index_type"],
                "metric": self.manifest["metric"],
            },
        }

    @classmethod
    def build(cls, chunks_path, directory, encoder, split=None, manifest_path=None):
        chunks, provenance = load_chunks(chunks_path, split, manifest_path)
        validate_chunks(chunks)
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=False)  # Prevent accidental overwrite.
        start = time.perf_counter()
        vectors = unit_vectors(encoder.encode([c["text"] for c in chunks]))
        if len(vectors) != len(chunks):
            raise ValueError("Embedding count mismatch")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(folder / "index.faiss"))
        write_jsonl(folder / "chunks.jsonl", chunks)
        manifest = {
            "schema_version": "1.0", "encoder": asdict(encoder.config),
            "index_type": "IndexFlatIP", "metric": "cosine", "normalized": True,
            "dimension": index.d, "count": index.ntotal,
            "source_sha256": digest(chunks_path),
            "data": provenance,
            "files": {name: digest(folder / name) for name in ("index.faiss", "chunks.jsonl")},
            "build_seconds": time.perf_counter() - start,
            "python": platform.python_version(),
            "packages": {p: version(p) for p in ("numpy", "faiss-cpu", "sentence-transformers")},
        }
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return cls(encoder, chunks, index, manifest)

    @classmethod
    def load(cls, directory, encoder=None, device="cpu"):
        folder = Path(directory)
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if manifest["schema_version"] != "1.0":
            raise ValueError("Unsupported schema version")
        for name in ("index.faiss", "chunks.jsonl"):
            if digest(folder / name) != manifest["files"][name]:
                raise ValueError(f"Index bundle checksum mismatch: {name}")
        encoder = encoder or BGEEncoder(EncoderConfig(**manifest["encoder"]), device)
        if asdict(encoder.config) != manifest["encoder"]:
            raise ValueError("Encoder configuration does not match index")
        chunks = read_jsonl(folder / "chunks.jsonl")
        validate_chunks(chunks)
        index = faiss.read_index(str(folder / "index.faiss"))
        if index.ntotal != len(chunks) or index.ntotal != manifest["count"] or index.d != manifest["dimension"]:
            raise ValueError("Index shape mismatch")
        return cls(encoder, chunks, index, manifest)

    def retrieve(self, question, top_k=5, paper_id=None):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        rows = None
        index = self.index
        if paper_id is not None:
            rows = self.paper_rows.get(paper_id, [])
            if not rows:
                return []
            if paper_id not in self.paper_indexes:
                sub = faiss.IndexFlatIP(index.d)
                sub.add(np.stack([index.reconstruct(i) for i in rows]))
                self.paper_indexes[paper_id] = sub
            index = self.paper_indexes[paper_id]
        query = unit_vectors(self.encoder.encode([question], query=True))
        if query.shape != (1, index.d):
            raise ValueError("Query embedding dimension mismatch")
        scores, ids = index.search(query, min(top_k, index.ntotal))
        return [{**self.chunks[rows[int(i)] if rows is not None else int(i)],
                 "score": float(score), "rank": rank}
                for rank, (i, score) in enumerate(zip(ids[0], scores[0]), 1)]


# Backward-compatible imports; new integrations may import from pipeline directly.
from .pipeline import Baseline, Generator, Retriever, PreviewGenerator, HTTPGenerator
