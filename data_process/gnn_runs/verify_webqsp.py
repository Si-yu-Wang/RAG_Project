"""Verify the WebQSP text artifacts and the matching original WebQSP splits (stdlib only)."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--text-dir", type=Path, default=REPO / "data/processed/webqsp_text")
    ap.add_argument("--raw-dir", type=Path, default=REPO / "data/data/webqsp")
    ap.add_argument("--text-only", action="store_true")
    a = ap.parse_args()
    summary = json.loads((a.text_dir / "summary.json").read_text(encoding="utf-8"))
    manifests = []
    for split, n, eligible in (("dev", 250, 235), ("test", 1639, 1551)):
        directory = a.text_dir / split
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["config_hash"] != summary["config_hash"]:
            raise ValueError(f"{split}: inconsistent config_hash")
        for name, expected in manifest["files"].items():
            if sha(directory / name) != expected:
                raise ValueError(f"File hash mismatch: {directory / name}")
        records = list(rows(directory / "qa_with_gold.jsonl"))
        counts = Counter(r["zero_gold_reason"] or "eligible" for r in records)
        if len(records) != n or counts["eligible"] != eligible:
            raise ValueError(f"{split}: unexpected question counts: {counts}")
        if len({r['question_id'] for r in records}) != n:
            raise ValueError(f"{split}: duplicate question IDs")
        print(f"{split}: {len(manifest['files'])} hashes OK; {n} questions; {dict(counts)}")
        manifests.append(manifest)
    if a.text_only:
        print("TEXT ONLY verified. Raw graph training inputs were not checked.")
        return
    if manifests[0]["config"]["source_sha256"] != manifests[1]["config"]["source_sha256"]:
        raise ValueError("dev/test manifests refer to different raw inputs")
    for split, expected in manifests[0]["config"]["source_sha256"].items():
        path = a.raw_dir / f"{split}.json"
        if sha(path) != expected:
            raise ValueError(f"Raw split differs from text conversion source: {path}")
        print(f"raw {split}: source SHA256 OK")
    for name, expected in (("entities.txt", 1441420), ("relations.txt", 6102), ("vocab.txt", 6718)):
        with (a.raw_dir / name).open(encoding="utf-8") as f:
            count = sum(1 for _ in f)
        if count != expected:
            raise ValueError(f"{name}: {count} rows != expected {expected}")
        print(f"{name}: {count} rows OK (row count is not a vocabulary hash check)")
    if not (a.raw_dir / "word_emb.npy").is_file():
        raise FileNotFoundError(a.raw_dir / "word_emb.npy")
    print("OK: raw splits match the existing text artifacts. Run the loader check and GPU smoke next.")


if __name__ == "__main__":
    main()
