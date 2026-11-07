"""QASPER -> GNN-RAG (gnn/ReaRev) data folder, so the GNN can act as an in-paper evidence retriever.

Graph per question (config ``qasper_inpaper_s1qc_v1``):
  * candidate pool = all chunks of the question's paper (size512_ov102_soft), identical to the
    dense baseline's ``scope=paper`` pool;
  * nodes: one virtual root ``Q::<question_id>`` (the seed / "topic entity") + one node per
    chunk instance ``<question_id>|<chunk_id>`` (per-question copies, see features);
  * relations (relations.txt, ids in this order):
      0 query.chunk.candidate         root -> every chunk instance
      1 paper.chunk.next_same_section reading-order successor inside the same section
      2 paper.chunk.next_new_section  reading-order successor across a section boundary
      3 paper.chunk.similar           in-paper BGE kNN (k=3), undirected, not already a "next" pair
    (ReaRev adds inverse directions itself, so each edge is stored once.)
  * node features (--entity_emb_file node_feat.npy): instance row = BGE(q) * BGE(c) (element-wise;
    its sum is the dense cosine), root row = 0. ReaRev has no question x node-text term, so
    without these features it only sees relation types.
  * answers = gold_chunk_ids (chunks covering a gold paragraph), as entity keys.

Files: train.json = eligible QASPER-train questions on non-held-out train papers;
       dev.json   = eligible questions on held-out train papers (model selection);
       test.json  = all QASPER-dev questions (the split the dense baseline reports; scored by
                    data_process/score_retrieval.py on the 890 eligible ones).
Never select checkpoints on test.json.

Two stages:
  python data_process/qasper_to_gnn.py export          # writes texts for embed_bge.py
  <torch env> python data_process/embed_bge.py --input .../_embed/chunks.jsonl  --out .../_embed/chunks
  <torch env> python data_process/embed_bge.py --input .../_embed/queries.jsonl --out .../_embed/queries --query
  python data_process/qasper_to_gnn.py build
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    QASPER_CHUNKS, QASPER_GNN_DIR, QASPER_QA_GOLD, WEBQSP_DIR, config_hash, ensure_dir, log, read_jsonl,
    read_lines, sha256_file, write_json, write_jsonl,
)

CONFIG_NAME = "qasper_inpaper_s1qc_v1"
RELATIONS = ["query.chunk.candidate", "paper.chunk.next_same_section", "paper.chunk.next_new_section",
             "paper.chunk.similar"]
REL_ROOT, REL_NEXT_SAME, REL_NEXT_NEW, REL_SIM = range(4)


def load_chunks(path):
    cols = ["chunk_id", "paper_id", "split", "sec_key", "chunk_idx_in_paper", "next_chunk_id", "text"]
    rows = pq.read_table(path, columns=cols).to_pylist()
    papers = {}
    for r in rows:
        papers.setdefault(r["paper_id"], []).append(r)
    for lst in papers.values():
        lst.sort(key=lambda r: r["chunk_idx_in_paper"])
    return rows, papers


def cmd_export(args) -> int:
    out = ensure_dir(Path(args.out) / "_embed")
    rows, _ = load_chunks(args.chunks)
    n1 = write_jsonl(out / "chunks.jsonl", ({"id": r["chunk_id"], "text": r["text"]} for r in rows))
    qa = read_jsonl(args.qa)
    n2 = write_jsonl(out / "queries.jsonl", ({"id": q["question_id"], "text": q["question"]} for q in qa))
    log(f"wrote {n1:,} chunk texts and {n2:,} question texts to {out}")
    log("next: run data_process/embed_bge.py on both files (queries with --query), then `build`")
    return 0


def load_emb(prefix: Path):
    x = np.load(str(prefix) + ".npy")
    ids = read_lines(str(prefix) + ".ids.txt")
    ids = [i for i in ids if i != ""]
    if len(ids) != len(x):
        raise ValueError(f"{prefix}: {len(ids)} ids vs {len(x)} rows")
    return x.astype(np.float32), {k: i for i, k in enumerate(ids)}


def paper_edges(chunks, cemb, cidx, knn_k, use_knn):
    """Edges between chunk positions (local indices in reading order) of one paper."""
    pos = {c["chunk_id"]: i for i, c in enumerate(chunks)}
    edges = []
    nxt = set()
    for i, c in enumerate(chunks):
        j = pos.get(c["next_chunk_id"]) if c["next_chunk_id"] else None
        if j is None:
            continue
        rel = REL_NEXT_SAME if chunks[j]["sec_key"] == c["sec_key"] else REL_NEXT_NEW
        edges.append((i, rel, j))
        nxt.add((min(i, j), max(i, j)))
    n_sim = 0
    if use_knn and len(chunks) > 1:
        m = cemb[[cidx[c["chunk_id"]] for c in chunks]]
        sim = m @ m.T
        np.fill_diagonal(sim, -np.inf)
        k = min(knn_k, len(chunks) - 1)
        pairs = set()
        for i in range(len(chunks)):
            # stable: highest similarity first, ties by reading order
            nb = np.argsort(-sim[i], kind="stable")[:k]
            for j in nb.tolist():
                pairs.add((min(i, j), max(i, j)))
        for a, b in sorted(pairs - nxt):
            edges.append((a, REL_SIM, b))
            n_sim += 1
    return edges, len(nxt), n_sim


def cmd_build(args) -> int:
    out = ensure_dir(args.out)
    rows, papers = load_chunks(args.chunks)
    qa = read_jsonl(args.qa)
    use_feat = not args.no_features
    use_knn = not args.no_knn
    if use_feat or use_knn:
        cemb, cidx = load_emb(Path(args.out) / "_embed" / "chunks")
        missing = [r["chunk_id"] for r in rows if r["chunk_id"] not in cidx]
        if missing:
            raise ValueError(f"{len(missing)} chunks lack embeddings, e.g. {missing[:3]}")
    if use_feat:
        qemb, qidx = load_emb(Path(args.out) / "_embed" / "queries")

    # ---- splits ----
    train_papers = sorted({q["paper_id"] for q in qa if q["split"] == "train"})
    rng = np.random.default_rng(args.seed)
    n_hold = int(round(args.holdout_frac * len(train_papers)))
    held = set(rng.choice(train_papers, size=n_hold, replace=False).tolist())

    def eligible(q):
        return q["zero_gold_reason"] == "" and len(q["gold_chunk_ids"]) > 0

    files = {
        "train": [q for q in qa if q["split"] == "train" and q["paper_id"] not in held and eligible(q)],
        "dev": [q for q in qa if q["split"] == "train" and q["paper_id"] in held and eligible(q)],
        "test": [q for q in qa if q["split"] == args.test_split],
    }

    # ---- per-paper edges (label-free: reading order + chunk text only) ----
    pedges = {}
    n_next = n_sim = 0
    for pid, chunks in papers.items():
        e, a, b = paper_edges(chunks, cemb if use_knn else None, cidx if use_knn else None, args.knn_k, use_knn)
        pedges[pid] = e
        n_next += a
        n_sim += b

    # ---- entities, records, features ----
    entity_keys, id_rows = [], []
    feats = []
    stats = {}
    for fname in ("train", "dev", "test"):
        records = []
        nodes_pq, tuples_pq, ans_pq = [], [], []
        for q in files[fname]:
            qid, pid = q["question_id"], q["paper_id"]
            chunks = papers[pid]
            root = len(entity_keys)
            entity_keys.append(f"Q::{qid}")
            id_rows.append({"entity_id": root, "key": f"Q::{qid}", "kind": "root", "file": fname,
                            "question_id": qid, "paper_id": pid, "chunk_id": None})
            inst = []
            for c in chunks:
                eid = len(entity_keys)
                key = f"{qid}|{c['chunk_id']}"
                entity_keys.append(key)
                id_rows.append({"entity_id": eid, "key": key, "kind": "chunk", "file": fname,
                                "question_id": qid, "paper_id": pid, "chunk_id": c["chunk_id"]})
                inst.append(eid)
            if use_feat:
                qv = qemb[qidx[qid]]
                feats.append(np.zeros((1, qv.shape[0]), np.float32))
                feats.append(qv[None, :] * cemb[[cidx[c["chunk_id"]] for c in chunks]])
            tuples = [[root, REL_ROOT, e] for e in inst]
            tuples += [[inst[a], r, inst[b]] for a, r, b in pedges[pid]]
            gold = list(dict.fromkeys(q["gold_chunk_ids"])) if eligible(q) else []
            chunk_set = {c["chunk_id"] for c in chunks}
            bad = [g for g in gold if g not in chunk_set]
            if bad:
                raise ValueError(f"{qid}: gold chunk outside its paper: {bad[:3]}")
            answers = [{"kb_id": f"{qid}|{g}", "text": f"{qid}|{g}"} for g in gold]
            records.append({"id": qid, "question": q["question"], "entities": [root], "answers": answers,
                            "subgraph": {"entities": [root] + inst, "tuples": tuples}})
            nodes_pq.append(1 + len(inst))
            tuples_pq.append(len(tuples))
            if gold:
                ans_pq.append(len(gold))
        write_jsonl(out / f"{fname}.json", records, ascii_only=True)

        def summ(xs):
            if not xs:
                return None
            return {"mean": round(statistics.mean(xs), 2), "median": float(np.median(xs)),
                    "p90": float(np.percentile(xs, 90)), "max": max(xs)}

        stats[fname] = {"questions": len(records), "papers": len({q["paper_id"] for q in files[fname]}),
                        "questions_with_answers": len(ans_pq), "nodes_per_question": summ(nodes_pq),
                        "tuples_per_question": summ(tuples_pq), "answers_per_question": summ(ans_pq)}
        if fname == "test":
            stats[fname]["zero_gold_reason"] = dict(Counter(q["zero_gold_reason"] or "eligible"
                                                            for q in files[fname]))

    (out / "entities.txt").write_text("\n".join(entity_keys) + "\n", encoding="utf-8", newline="\n")
    (out / "relations.txt").write_text("\n".join(RELATIONS) + "\n", encoding="utf-8", newline="\n")
    for name in ("vocab.txt", "word_emb.npy"):
        shutil.copyfile(Path(args.webqsp_dir) / name, out / name)
    if use_feat:
        feat = np.concatenate(feats, axis=0).astype(np.float32)
        if feat.shape[0] != len(entity_keys):
            raise AssertionError("feature rows != entities")
        np.save(out / "node_feat.npy", feat)
    elif (out / "node_feat.npy").exists():
        (out / "node_feat.npy").unlink()
    pq.write_table(pa.Table.from_pylist(id_rows), out / "id_map.parquet")
    (out / "heldout_train_papers.txt").write_text("\n".join(sorted(held)) + "\n", encoding="utf-8")

    cfg = {"pool": "in-paper chunks (size512_ov102_soft)", "seed_node": "virtual root per question",
           "relations": RELATIONS, "knn_k": args.knn_k if use_knn else 0,
           "node_features": "bge(q)*bge(c)" if use_feat else None,
           "answers": "gold_chunk_ids", "holdout_frac": args.holdout_frac, "seed": args.seed,
           "test_split": args.test_split,
           "inputs_sha256": {"chunks": sha256_file(args.chunks), "qa_with_gold": sha256_file(args.qa)},
           "encoder": {"model": "BAAI/bge-small-en-v1.5", "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
                       "normalize": True, "max_length": 512}}
    manifest = {
        "config_name": CONFIG_NAME, "config": cfg, "config_hash": config_hash(cfg),
        "entities": len(entity_keys), "relations": len(RELATIONS),
        "next_pairs": n_next, "similar_pairs": n_sim, "files": stats,
        "node_feat": ({"rows": len(entity_keys), "dim": int(cemb.shape[1]), "dtype": "float32"}
                      if use_feat else None),
        "gnn_rag_flags": ("--relation_word_emb False --lm sbert"
                          + (" --entity_emb_file node_feat.npy --kge_frozen 1" if use_feat else "")),
    }
    write_json(out / "manifest.json", manifest)
    log(json.dumps(manifest, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["export", "build"])
    ap.add_argument("--chunks", default=str(QASPER_CHUNKS))
    ap.add_argument("--qa", default=str(QASPER_QA_GOLD))
    ap.add_argument("--out", default=str(QASPER_GNN_DIR))
    ap.add_argument("--webqsp-dir", default=str(WEBQSP_DIR), help="source of vocab.txt / word_emb.npy")
    ap.add_argument("--test-split", default="dev", help="QASPER split written to test.json")
    ap.add_argument("--holdout-frac", type=float, default=0.15, help="share of QASPER-train papers for dev.json")
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--knn-k", type=int, default=3)
    ap.add_argument("--no-knn", action="store_true", help="ablation: only root + next edges")
    ap.add_argument("--no-features", action="store_true", help="ablation: no node_feat.npy")
    args = ap.parse_args()
    return cmd_export(args) if args.command == "export" else cmd_build(args)


if __name__ == "__main__":
    sys.exit(main())
