"""Step 1: build one global MID -> name map for the dense WebQSP subgraphs.

    names = entities_names.json  ∪  names recovered from RoG-webqsp by position

RoG-webqsp's ``graph`` column is the local dense ``webqsp`` subgraph of the same question,
triple by triple in the same order, with node MIDs replaced by names where RoG had one.
Aligning position i of both gives (MID, name) pairs. The script checks that the relations
agree at every position and that no MID receives two different names.

The map is GLOBAL (one name per MID, independent of the question). Never add names
from answers[].text: that would name exactly the answers (label leakage).

Output: data/raw/webqsp_names/mid2name.tsv  (mid \t name \t source) + mid2name_stats.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import NAMES_DIR, WEBQSP_DIR, is_mid, iter_jsonl, log, read_lines, write_json  # noqa: E402

SPLIT_SHARDS = {  # local split -> RoG split prefix
    "train": "train-",
    "dev": "validation-",
    "test": "test-",
}
_MOJIBAKE = re.compile("[ÂÃ][\u0080-¿]")
_WS = re.compile(r"[\t\r\n]+")


def clean_name(name: str) -> str:
    name = _WS.sub(" ", name).strip()
    if _MOJIBAKE.search(name):
        for enc in ("cp1252", "latin-1"):
            try:
                fixed = name.encode(enc).decode("utf-8")
                return fixed
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
    return name


def load_local_tuples(path: Path) -> dict[str, np.ndarray]:
    out = {}
    for rec in iter_jsonl(path):
        t = rec["subgraph"]["tuples"]
        out[rec["id"]] = np.asarray(t, dtype=np.int64).reshape(-1, 3) if t else np.zeros((0, 3), np.int64)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names-dir", default=str(NAMES_DIR))
    ap.add_argument("--webqsp-dir", default=str(WEBQSP_DIR))
    args = ap.parse_args()
    names_dir, wq = Path(args.names_dir), Path(args.webqsp_dir)

    entities = read_lines(wq / "entities.txt")
    relations = read_lines(wq / "relations.txt")
    log(f"webqsp vocab: {len(entities):,} entities, {len(relations):,} relations")

    raw = json.loads((names_dir / "entities_names.json").read_text(encoding="utf-8"))
    names: dict[str, str] = {}
    source: dict[str, str] = {}
    n_mojibake_fixed = 0
    for mid, name in raw.items():
        c = clean_name(name)
        if c != _WS.sub(" ", name).strip():
            n_mojibake_fixed += 1
        if c:
            names[mid] = c
            source[mid] = "gnnrag_json"
    n_json = len(names)
    log(f"entities_names.json: {n_json:,} names ({n_mojibake_fixed} mojibake fixed)")

    stats = {"json_names": n_json, "json_mojibake_fixed": n_mojibake_fixed, "splits": {}}
    rog_pairs: dict[str, str] = {}
    conflicts, json_disagree = 0, 0
    for split, prefix in SPLIT_SHARDS.items():
        local = load_local_tuples(wq / f"{split}.json")
        shards = sorted((names_dir / "rog").glob(prefix + "*.parquet"))
        if not shards:
            raise FileNotFoundError(f"no RoG shards for {split} in {names_dir / 'rog'}")
        rows = aligned = rel_mismatch = len_mismatch = missing = 0
        for shard in shards:
            pf = pq.ParquetFile(shard)
            for batch in pf.iter_batches(batch_size=32, columns=["id", "graph"]):
                for qid, graph in zip(batch.column("id").to_pylist(), batch.column("graph").to_pylist()):
                    rows += 1
                    tup = local.get(qid)
                    if tup is None:
                        missing += 1
                        continue
                    if len(graph) != len(tup):
                        len_mismatch += 1
                        continue
                    aligned += 1
                    for (h_name, r_name, t_name), (h, r, t) in zip(graph, tup):
                        if relations[r] != r_name:
                            rel_mismatch += 1
                            continue
                        for mid, nm in ((entities[h], h_name), (entities[t], t_name)):
                            if nm == mid or not is_mid(mid):
                                continue  # unnamed in RoG, or a literal node
                            nm = clean_name(nm)
                            if not nm:
                                continue
                            prev = rog_pairs.get(mid)
                            if prev is None:
                                rog_pairs[mid] = nm
                            elif prev != nm:
                                conflicts += 1
        del local
        stats["splits"][split] = {"rog_rows": rows, "aligned": aligned, "length_mismatch": len_mismatch,
                                  "missing_locally": missing, "relation_mismatch_positions": rel_mismatch}
        log(f"{split}: RoG rows {rows:,}, aligned {aligned:,}, len mismatch {len_mismatch}, "
            f"rel mismatch positions {rel_mismatch}, missing locally {missing}")

    added = 0
    for mid, nm in rog_pairs.items():
        if mid in names:
            if names[mid] != nm:
                json_disagree += 1
            continue
        names[mid] = nm
        source[mid] = "rog_alignment"
        added += 1
    stats.update({"rog_pairs": len(rog_pairs), "rog_conflicting_positions": conflicts,
                  "rog_vs_json_disagreements": json_disagree, "rog_added": added, "total_names": len(names),
                  "dense_vocab_entities": len(entities),
                  "dense_vocab_named": sum(1 for e in entities if e in names)})
    out = names_dir / "mid2name.tsv"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("mid\tname\tsource\n")
        for mid in sorted(names):
            f.write(f"{mid}\t{names[mid]}\t{source[mid]}\n")
    write_json(names_dir / "mid2name_stats.json", stats)
    log(json.dumps(stats, indent=1))
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
