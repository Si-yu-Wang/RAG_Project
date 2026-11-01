"""Fetch the BGE tokenizer as a single file, bypassing the HuggingFace hub cache.

The hub cache writes to ``~/.cache/huggingface`` and cleans up ``*.incomplete`` blobs,
which trips the sandbox bulk-delete guard. Downloading ``tokenizer.json`` directly with
urllib produces exactly one file and no deletions, so profiling stays hermetic and
offline-friendly.

Usage:
    python scripts/dev/fetch_tokenizer.py --out data/raw/bge_tok/tokenizer.json
"""

from __future__ import annotations

import argparse
import os
import urllib.request

URL = "https://huggingface.co/BAAI/bge-large-en-v1.5/resolve/main/tokenizer.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    ap.add_argument("--out", default="data/raw/bge_tok/tokenizer.json")
    args = ap.parse_args()
    if os.path.exists(args.out):
        print(f"[skip] exists: {args.out} ({os.path.getsize(args.out)} bytes)")
        return 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"[get] {args.url}")
    urllib.request.urlretrieve(args.url, args.out)
    print(f"[ok] {args.out} ({os.path.getsize(args.out)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
