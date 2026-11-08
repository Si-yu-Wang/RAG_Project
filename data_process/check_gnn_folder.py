"""Replay GNN-RAG (gnn/dataset_load.py @28d31bc) input rules on a data folder, without torch.

Checks what the loader silently gets wrong or crashes on:
  * entities.txt keys unique (load_dict maps key -> line index; duplicates corrupt ids);
  * every *.json line is pure ASCII (the loader opens files with the platform default codec,
    cp936 on Chinese Windows);
  * topic entities: non-empty list of ints in range (string seeds are silently dropped; records
    with empty `entities` are skipped, shifting the .info line order);
  * subgraph.entities ints in range; tuples [h, r, t] ints, endpoints listed in the local graph,
    0 <= r < #relations;
  * answers[].kb_id is a key of entities.txt (otherwise KeyError) and lies in the local graph;
  * node_feat.npy (if present) has exactly #entities rows (otherwise silent random init).

  python data_process/check_gnn_folder.py data/processed/qasper_gnn
  python data_process/check_gnn_folder.py data/data/webqsp --files test
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_lines  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--files", nargs="+", default=["train", "dev", "test"])
    ap.add_argument("--feat", default="node_feat.npy")
    a = ap.parse_args()
    d = Path(a.folder)
    errors = Counter()
    ents = [line.strip() for line in read_lines(d / "entities.txt")]
    if ents and ents[-1] == "":
        ents = ents[:-1]
    e2i = {}
    for i, k in enumerate(ents):
        if k in e2i:
            errors["duplicate_entity_key"] += 1
        e2i[k] = i
    rels = [r for r in read_lines(d / "relations.txt") if r.strip()]
    n_ent, n_rel = len(ents), len(rels)
    print(f"{d}: {n_ent:,} entities, {n_rel} relations")
    if not (d / "vocab.txt").exists():
        errors["missing_vocab.txt"] += 1
    report = {}
    for name in a.files:
        path = d / f"{name}.json"
        if not path.exists():
            print(f"  {name}.json: missing")
            continue
        raw = path.read_bytes()
        if not raw.isascii():
            errors[f"{name}:non_ascii_bytes"] += 1
        st = Counter()
        max_local = max_tuples = 0
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            st["records"] += 1
            topics = r["entities"]
            if not topics:
                st["skipped_empty_entities"] += 1
                continue
            if not all(isinstance(t, int) and 0 <= t < n_ent for t in topics):
                errors[f"{name}:topic_not_int_in_range"] += 1
            local = set(topics)
            for e in r["subgraph"]["entities"]:
                if not (isinstance(e, int) and 0 <= e < n_ent):
                    errors[f"{name}:subgraph_entity_not_int_in_range"] += 1
                    break
            local.update(r["subgraph"]["entities"])
            max_local = max(max_local, len(local))
            tuples = r["subgraph"]["tuples"]
            max_tuples = max(max_tuples, len(tuples))
            for h, rel, t in tuples:
                if not (isinstance(h, int) and isinstance(t, int) and isinstance(rel, int)):
                    errors[f"{name}:tuple_not_int"] += 1
                    break
                if h not in local or t not in local:
                    errors[f"{name}:tuple_endpoint_not_local"] += 1
                    break
                if not 0 <= rel < n_rel:
                    errors[f"{name}:relation_out_of_range"] += 1
                    break
            if not r["answers"]:
                st["no_answers"] += 1
            for ans in r["answers"]:
                kb = ans["kb_id"]
                key = ans["text"] if isinstance(kb, int) else kb
                if key not in e2i:
                    errors[f"{name}:answer_key_not_in_entities"] += 1
                elif e2i[key] not in local:
                    st["answer_outside_local_graph"] += 1
        st["max_local_entities"] = max_local
        st["max_tuples"] = max_tuples
        report[name] = dict(st)
        print(f"  {name}.json: {dict(st)}")
    feat = d / a.feat
    if feat.exists():
        x = np.load(feat, mmap_mode="r")
        print(f"  {a.feat}: shape {x.shape} {x.dtype}")
        if x.shape[0] != n_ent:
            errors["node_feat_rows_ne_entities"] += 1
        if not np.isfinite(np.asarray(x[: min(len(x), 100000)])).all():
            errors["node_feat_nonfinite"] += 1
    if errors:
        print("ERRORS:", dict(errors))
        return 1
    print("OK: folder satisfies the GNN-RAG loader contract")
    return 0


if __name__ == "__main__":
    sys.exit(main())
