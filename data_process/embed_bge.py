"""Encode texts with exactly the dense baseline's encoder (BGE-small-en-v1.5, CLS pooling, L2 norm).

Runs in any environment with torch + transformers (no sentence-transformers / pyarrow needed),
e.g. a CUDA env. Reproduces baselines/member2_dense_rag/dense_rag/core.py BGEEncoder:
  model BAAI/bge-small-en-v1.5 @ 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a, max_length 512,
  query prefix "Represent this sentence for searching relevant passages: ",
  CLS pooling + L2 normalisation (the model's sentence-transformers modules.json).

Input : JSON lines {"id": str, "text": str}   (written by qasper_to_gnn.py export / dense_rank.py export)
Output: <out>.npy (float32 [n, 384]) and <out>.ids.txt (one id per line, same order as the input)

  python data_process/embed_bge.py --input X.jsonl --out X            # passages
  python data_process/embed_bge.py --input Q.jsonl --out Q --query    # queries (adds the prefix)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

MODEL = "BAAI/bge-small-en-v1.5"
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_LENGTH = 512


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--query", action="store_true", help="prepend the BGE query instruction")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    ap.add_argument("--allow-download", action="store_true", help="allow fetching the model from the HF hub")
    args = ap.parse_args()

    if not args.allow_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from transformers import AutoModel, AutoTokenizer

    ids, texts = [], []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                ids.append(row["id"])
                texts.append((QUERY_PREFIX + row["text"]) if args.query else row["text"])
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate ids in input")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModel.from_pretrained(MODEL, revision=REVISION).to(device).eval()
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))  # length-sorted batches
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    t0 = time.time()
    with torch.inference_mode():
        for b in range(0, len(order), args.batch_size):
            idx = order[b:b + args.batch_size]
            enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=MAX_LENGTH,
                      return_tensors="pt").to(device)
            cls = model(**enc).last_hidden_state[:, 0]
            cls = torch.nn.functional.normalize(cls.float(), p=2, dim=1)
            out[idx] = cls.cpu().numpy()
            if (b // args.batch_size) % 200 == 0:
                print(f"  {b + len(idx):,}/{len(texts):,}  {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    if not np.isfinite(out).all():
        raise ValueError("non-finite embeddings")
    np.save(args.out + ".npy", out)
    with open(args.out + ".ids.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(ids) + "\n")
    print(json.dumps({"n": len(ids), "dim": int(out.shape[1]), "device": device, "query": args.query,
                      "seconds": round(time.time() - t0, 1), "model": MODEL, "revision": REVISION}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
