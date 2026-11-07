"""Reference dense ranker (BGE-small, cosine, within paper) writing the unified ranking format.

It reproduces member 2's dense retrieval (same encoder, same scope=paper pool, cosine on
L2-normalised vectors) without faiss / sentence-transformers, so that
  (a) the unified scorer can be checked against member 2's published QASPER numbers, and
  (b) dense rankings for WebQSP-text can be produced on a GPU box.
The official dense runs remain member 2's pipeline; its predictions convert with
``score_retrieval.py from-dense``.

  python data_process/dense_rank.py export --dataset webqsp --split test   # texts for embed_bge.py
  <torch env> python data_process/embed_bge.py --input <dir>/chunks.jsonl  --out <dir>/chunks
  <torch env> python data_process/embed_bge.py --input <dir>/queries.jsonl --out <dir>/queries --query
  python data_process/dense_rank.py rank --dataset webqsp --split test --out rankings/webqsp_test_dense.jsonl
For QASPER the embeddings written by qasper_to_gnn.py (data/processed/qasper_gnn/_embed) are reused.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import QASPER_GNN_DIR, REPO, ensure_dir, log, read_lines, write_jsonl  # noqa: E402
from score_retrieval import Dataset  # noqa: E402


def emb_dir(a) -> Path:
    if a.emb_dir:
        return Path(a.emb_dir)
    if a.dataset == "qasper":
        return QASPER_GNN_DIR / "_embed"
    return REPO / "data" / "processed" / "webqsp_text" / a.split / "_embed"


def cmd_export(a):
    ds = Dataset(a.dataset, a.split)
    d = ensure_dir(emb_dir(a))
    texts = pq.read_table(ds.chunks_path, columns=["chunk_id", "text"]).to_pylist()
    n1 = write_jsonl(d / "chunks.jsonl", ({"id": r["chunk_id"], "text": r["text"]}
                                          for r in texts if r["chunk_id"] in ds.chunk_paper))
    n2 = write_jsonl(d / "queries.jsonl", ({"id": q["question_id"], "text": q["question"]} for q in ds.qa))
    log(f"wrote {n1:,} chunk texts and {n2:,} queries to {d}")
    return 0


def load(prefix):
    x = np.load(str(prefix) + ".npy")
    ids = [i for i in read_lines(str(prefix) + ".ids.txt") if i]
    return x, {k: i for i, k in enumerate(ids)}


def cmd_rank(a):
    ds = Dataset(a.dataset, a.split)
    d = emb_dir(a)
    cx, cid = load(d / "chunks")
    qx, qid = load(d / "queries")
    rows = []
    for q in ds.qa:
        pool = ds.pool.get(q["paper_id"], [])
        m = cx[[cid[c] for c in pool]]
        s = m @ qx[qid[q["question_id"]]]
        order = np.argsort(-s, kind="stable")[: a.depth]
        rows.append({"question_id": q["question_id"], "method": a.method,
                     "ranked_chunk_ids": [pool[i] for i in order.tolist()],
                     "scores": [float(s[i]) for i in order.tolist()]})
    ensure_dir(Path(a.out).parent)
    n = write_jsonl(a.out, rows)
    log(f"wrote {n} rankings to {a.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["export", "rank"])
    ap.add_argument("--dataset", required=True, choices=["qasper", "webqsp"])
    ap.add_argument("--split", required=True)
    ap.add_argument("--emb-dir", default=None)
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--method", default="dense_bge_ref")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.command == "rank" and not a.out:
        ap.error("--out is required for rank")
    return cmd_export(a) if a.command == "export" else cmd_rank(a)


if __name__ == "__main__":
    sys.exit(main())
