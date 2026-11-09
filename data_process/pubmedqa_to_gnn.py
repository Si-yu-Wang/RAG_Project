"""PubMedQA citation-graph corpus -> GNN-RAG (ReaRev) data, plus Dense / BM25 rankings and scoring.

Task (from scripts/21_build_pubmedqa_graph.py): cross-document evidence retrieval. The pool is the
WHOLE corpus (1,395 papers / 3,707 chunks: 951 PubMedQA source papers + 444 co-cited references);
gold = all chunks of the question's source paper. Unlike QASPER there is no in-paper scoping.

Split: the corpus has no train/test, so the 951 questions are split once with a fixed seed into
train / dev / test = 60 / 10 / 30 % (by question; each question has its own source paper).
dev selects ReaRev checkpoints; every method is reported on test.

Per-question ReaRev subgraph (GNN-RAG style retrieve-then-reason, label-free):
  * seed papers = papers of the dense top-``seed_k`` chunks; add their 1-hop ``cites`` neighbours;
  * candidates = all chunks of those papers, highest dense cosine first, capped at ``max_cand``;
  * nodes: virtual root ``Q::<qid>`` + one instance node per candidate ``<qid>|<chunk_id>``;
  * relations: 0 root -> candidate, 1 same-paper next chunk, 2 chunks of citing <-> cited papers,
    3 in-candidate BGE kNN (k=3, pairs not already linked);
  * node feature = BGE(q) * BGE(c) (row sum = dense cosine), root = 0; answers = gold chunks
    inside the candidates. train/dev keep only questions whose candidates contain gold.
Final GNN ranking = candidates by ReaRev probability, then the rest of the corpus by dense, so
every method ranks the same 3,707 chunks and is scored identically.

Subcommands: export | build | dense | bm25 | from-gnn | score | compare
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import re
import shutil
import statistics
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, config_hash, ensure_dir, iter_jsonl, log, read_lines, sha256_file, write_json, write_jsonl  # noqa: E402
from score_retrieval import question_scores  # noqa: E402

SRC = REPO / "data/processed/pubmedqa_graph"
OUT = REPO / "data/processed/pubmedqa_gnn"
SUPPORT = REPO / "data/processed/qasper_support"
CONFIG_NAME = "pubmedqa_citegraph_s1_v1"
RELATIONS = ["query.chunk.candidate", "paper.chunk.next", "paper.cites.paper", "chunk.similar"]
REL_ROOT, REL_NEXT, REL_CITES, REL_SIM = range(4)
KS = (1, 3, 5, 10)
METRICS = [f"{m}@{k}" for k in KS for m in ("hit", "recall", "mrr")] + [f"paper_hit@{k}" for k in KS]
DEPTH = 100
TOKEN = re.compile(r"\w+")


# ----------------------------------------------------------------------------- data
def load_corpus(src=SRC):
    chunks = pq.read_table(src / "chunks.parquet", columns=["chunk_id", "paper_id", "chunk_idx_in_paper", "text"]).to_pylist()
    chunks.sort(key=lambda c: (c["paper_id"], c["chunk_idx_in_paper"], c["chunk_id"]))
    papers = {}
    for c in chunks:
        papers.setdefault(c["paper_id"], []).append(c["chunk_id"])
    edges = pq.read_table(src / "graph_edges.parquet").to_pylist()
    nxt = [(e["src_id"], e["dst_id"]) for e in edges if e["edge_type"] == "adjacency"]
    # the source data has 2 self-citations (src == dst); they are not cross-paper links
    cites = [(e["src_id"], e["dst_id"]) for e in edges if e["edge_type"] == "cites" and e["src_id"] != e["dst_id"]]
    qa = pq.read_table(src / "qa.parquet").to_pylist()
    for q in qa:
        q["gold_chunk_ids"] = list(dict.fromkeys(q["gold_chunk_ids"]))
    qa.sort(key=lambda q: q["question_id"])
    return chunks, papers, nxt, cites, qa


def load_emb(prefix):
    x = np.load(str(prefix) + ".npy").astype(np.float32)
    ids = [i for i in read_lines(str(prefix) + ".ids.txt") if i]
    if len(ids) != len(x):
        raise ValueError(f"{prefix}: {len(ids)} ids vs {len(x)} rows")
    return x, ids


def aligned_embeddings(out, chunks, qa):
    cx, cids = load_emb(out / "_embed/chunks")
    qx, qids = load_emb(out / "_embed/queries")
    ci, qi = {k: i for i, k in enumerate(cids)}, {k: i for i, k in enumerate(qids)}
    miss = [c["chunk_id"] for c in chunks if c["chunk_id"] not in ci] + [q["question_id"] for q in qa if q["question_id"] not in qi]
    if miss:
        raise ValueError(f"{len(miss)} texts lack embeddings, e.g. {miss[:3]}")
    return cx[[ci[c["chunk_id"]] for c in chunks]], qx[[qi[q["question_id"]] for q in qa]]


def make_splits(qids, seed, dev_frac, test_frac):
    order = np.random.default_rng(seed).permutation(sorted(qids)).tolist()
    n_test, n_dev = round(test_frac * len(order)), round(dev_frac * len(order))
    return {q: ("test" if i < n_test else "dev" if i < n_test + n_dev else "train") for i, q in enumerate(order)}


# ----------------------------------------------------------------------------- export / build
def cmd_export(a):
    out = ensure_dir(Path(a.out) / "_embed")
    chunks, _, _, _, qa = load_corpus()
    n1 = write_jsonl(out / "chunks.jsonl", ({"id": c["chunk_id"], "text": c["text"]} for c in chunks))
    n2 = write_jsonl(out / "queries.jsonl", ({"id": q["question_id"], "text": q["question"]} for q in qa))
    log(f"wrote {n1} chunk texts and {n2} question texts to {out}; next: embed_bge.py (queries with --query), then build")
    return 0


def cmd_build(a):
    out = Path(a.out)
    for name in ("train.json", "dev.json", "test.json", "node_feat.npy", "entities.txt", "id_map.parquet", "splits.json"):
        if (out / name).exists():
            raise FileExistsError(f"{out / name} exists; build into a fresh folder")
    chunks, papers, nxt, cites, qa = load_corpus()
    cemb, qemb = aligned_embeddings(out, chunks, qa)
    cpos = {c["chunk_id"]: i for i, c in enumerate(chunks)}
    cpaper = [c["paper_id"] for c in chunks]
    paper_nb = {}
    for s, d in cites:
        paper_nb.setdefault(s, set()).add(d)
        paper_nb.setdefault(d, set()).add(s)
    next_pairs = {(min(cpos[s], cpos[d]), max(cpos[s], cpos[d])) for s, d in nxt if s in cpos and d in cpos}
    cite_pairs = {(min(s, d), max(s, d)) for s, d in cites}
    splits = make_splits([q["question_id"] for q in qa], a.seed, a.dev_frac, a.test_frac)
    cos = qemb @ cemb.T

    entity_keys, id_rows, feats, stats = [], [], [], {}
    records = {"train": [], "dev": [], "test": []}
    coverage = {k: [] for k in records}
    sizes = {k: [] for k in records}
    for qn, q in enumerate(qa):
        qid, split = q["question_id"], splits[q["question_id"]]
        order = np.argsort(-cos[qn], kind="stable")
        seed_papers = list(dict.fromkeys(cpaper[i] for i in order[: a.seed_k]))
        cand_papers = set(seed_papers)
        for p in seed_papers:
            cand_papers |= paper_nb.get(p, set())
        cand = sorted((cpos[c] for p in cand_papers for c in papers[p]), key=lambda i: (-cos[qn, i], i))[: a.max_cand]
        gold = set(q["gold_chunk_ids"])
        in_cand = [chunks[i]["chunk_id"] for i in cand if chunks[i]["chunk_id"] in gold]
        coverage[split].append(len(in_cand) / len(gold))
        if split != "test" and not in_cand:
            continue
        root = len(entity_keys)
        entity_keys.append(f"Q::{qid}")
        id_rows.append({"entity_id": root, "key": f"Q::{qid}", "kind": "root", "file": split,
                        "question_id": qid, "paper_id": q["source_paper_id"], "chunk_id": None})
        local = {}
        for i in cand:
            eid = len(entity_keys)
            key = f"{qid}|{chunks[i]['chunk_id']}"
            entity_keys.append(key)
            id_rows.append({"entity_id": eid, "key": key, "kind": "chunk", "file": split, "question_id": qid,
                            "paper_id": cpaper[i], "chunk_id": chunks[i]["chunk_id"]})
            local[i] = eid
        feats.append(np.zeros((1, cemb.shape[1]), np.float32))
        feats.append(qemb[qn][None, :] * cemb[cand])
        tuples = [[root, REL_ROOT, local[i]] for i in cand]
        linked = set()
        for x, y in sorted(next_pairs):
            if x in local and y in local:
                tuples.append([local[x], REL_NEXT, local[y]])
                linked.add((x, y))
        by_paper = {}
        for i in cand:
            by_paper.setdefault(cpaper[i], []).append(i)
        for p1, p2 in sorted(cite_pairs):
            for x in by_paper.get(p1, []):
                for y in by_paper.get(p2, []):
                    tuples.append([local[x], REL_CITES, local[y]])
                    linked.add((min(x, y), max(x, y)))
        if len(cand) > 1:
            m = cemb[cand]
            sim = m @ m.T
            np.fill_diagonal(sim, -np.inf)
            pairs = set()
            for r in range(len(cand)):
                for c in np.argsort(-sim[r], kind="stable")[: min(a.knn_k, len(cand) - 1)].tolist():
                    x, y = cand[r], cand[c]
                    pairs.add((min(x, y), max(x, y)))
            for x, y in sorted(pairs - linked):
                tuples.append([local[x], REL_SIM, local[y]])
        answers = [{"kb_id": f"{qid}|{g}", "text": f"{qid}|{g}"} for g in in_cand]
        records[split].append({"id": qid, "question": q["question"], "entities": [root], "answers": answers,
                               "subgraph": {"entities": [root] + [local[i] for i in cand], "tuples": tuples}})
        sizes[split].append((len(cand) + 1, len(tuples)))

    out.mkdir(parents=True, exist_ok=True)
    for split, rows in records.items():
        write_jsonl(out / f"{split}.json", rows, ascii_only=True)
        stats[split] = {"questions_total": sum(1 for v in splits.values() if v == split), "records": len(rows),
                        "gold_chunk_coverage_mean": round(float(np.mean(coverage[split])), 4),
                        "questions_with_any_gold_in_candidates": int(sum(c > 0 for c in coverage[split])),
                        "nodes_per_question_mean": round(statistics.mean(s[0] for s in sizes[split]), 1),
                        "tuples_per_question_mean": round(statistics.mean(s[1] for s in sizes[split]), 1)}
    feat = np.concatenate(feats).astype(np.float32)
    if feat.shape[0] != len(entity_keys):
        raise AssertionError("feature rows != entities")
    np.save(out / "node_feat.npy", feat)
    (out / "entities.txt").write_text("\n".join(entity_keys) + "\n", encoding="utf-8", newline="\n")
    (out / "relations.txt").write_text("\n".join(RELATIONS) + "\n", encoding="utf-8", newline="\n")
    for name in ("vocab.txt", "word_emb.npy"):
        shutil.copyfile(Path(a.support) / name, out / name)
    pq.write_table(pa.Table.from_pylist(id_rows), out / "id_map.parquet")
    write_json(out / "splits.json", splits)
    cfg = {"pool": "whole corpus (3,707 chunks)", "candidates": "papers of dense top-k chunks + 1-hop cites, capped",
           "seed_k": a.seed_k, "max_cand": a.max_cand, "knn_k": a.knn_k, "relations": RELATIONS,
           "split_seed": a.seed, "dev_frac": a.dev_frac, "test_frac": a.test_frac,
           "node_features": "bge(q)*bge(c)", "answers": "gold_chunk_ids inside candidates",
           "inputs_sha256": {n: sha256_file(SRC / f"{n}.parquet") for n in ("chunks", "graph_edges", "qa")}}
    manifest = {"config_name": CONFIG_NAME, "config": cfg, "config_hash": config_hash(cfg),
                "entities": len(entity_keys), "files": stats,
                "gnn_rag_flags": "--relation_word_emb False --lm sbert --entity_emb_file node_feat.npy --kge_frozen 1"}
    write_json(out / "manifest.json", manifest)
    log(json.dumps(manifest, indent=1))
    return 0


# ----------------------------------------------------------------------------- rankings
def test_questions(out, qa):
    splits = json.loads((Path(out) / "splits.json").read_text(encoding="utf-8"))
    return [q for q in qa if splits[q["question_id"]] == "test"]


def cmd_dense(a):
    out = Path(a.out_dir)
    chunks, _, _, _, qa = load_corpus()
    cemb, qemb = aligned_embeddings(out, chunks, qa)
    keep = {q["question_id"] for q in test_questions(out, qa)}
    rows = []
    for qn, q in enumerate(qa):
        if q["question_id"] not in keep:
            continue
        s = cemb @ qemb[qn]
        order = np.argsort(-s, kind="stable")[:DEPTH]
        rows.append({"question_id": q["question_id"], "method": a.method,
                     "ranked_chunk_ids": [chunks[i]["chunk_id"] for i in order], "scores": [float(s[i]) for i in order]})
    write_jsonl(a.out, rows)
    log(f"wrote {len(rows)} dense rankings to {a.out}")
    return 0


def cmd_bm25(a):
    chunks, _, _, _, qa = load_corpus()
    toks = [TOKEN.findall(c["text"].lower()) for c in chunks]
    tfs = [Counter(t) for t in toks]
    lens = np.array([len(t) for t in toks], dtype=np.float64)
    avg, n = lens.mean(), len(chunks)
    df = Counter(t for ts in toks for t in set(ts))
    k1, b = 1.2, 0.75
    rows = []
    for q in test_questions(a.out_dir, qa):
        s = np.zeros(n)
        for t in TOKEN.findall(q["question"].lower()):
            if t not in df:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            for i in range(n):
                f = tfs[i].get(t)
                if f:
                    s[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * lens[i] / avg))
        order = np.argsort(-s, kind="stable")[:DEPTH]
        rows.append({"question_id": q["question_id"], "method": "bm25",
                     "ranked_chunk_ids": [chunks[i]["chunk_id"] for i in order], "scores": [float(s[i]) for i in order]})
    write_jsonl(a.out, rows)
    log(f"wrote {len(rows)} bm25 rankings to {a.out}")
    return 0


def cmd_from_gnn(a):
    gnn = Path(a.gnn_dir)
    records = [r for r in iter_jsonl(gnn / "test.json") if r["entities"]]
    infos = list(iter_jsonl(a.info))
    if len(infos) != len(records):
        raise ValueError(f"{len(infos)} .info lines vs {len(records)} test records")
    dense = {r["question_id"]: r["ranked_chunk_ids"] for r in iter_jsonl(a.dense)}
    mapping = pq.read_table(gnn / "id_map.parquet", columns=["key", "chunk_id"]).to_pylist()
    rows = []
    for rec, info in zip(records, infos):
        qid = rec["id"]
        if "id" in info and info["id"] != qid:
            raise ValueError(f"alignment error: {info['id']} vs {qid}")
        cand = [mapping[e]["chunk_id"] for e in rec["subgraph"]["entities"][1:]]
        prob = {}
        for key, p in info.get("cand", []):
            q2, _, chunk = key.partition("|")
            if q2 == qid:
                prob[chunk] = float(p)
        ranked = sorted(cand, key=lambda c: -prob.get(c, 0.0))  # stable: ties keep dense order
        seen = set(ranked)
        ranked += [c for c in dense[qid] if c not in seen]
        rows.append({"question_id": qid, "method": a.method, "ranked_chunk_ids": ranked[:DEPTH],
                     "scores": [prob.get(c) for c in ranked[:DEPTH]]})
    write_jsonl(a.out, rows)
    log(f"wrote {len(rows)} GNN rankings (candidates by ReaRev, rest by dense) to {a.out}")
    return 0


# ----------------------------------------------------------------------------- scoring
def per_question(out_dir, path):
    chunks, _, _, _, qa = load_corpus()
    paper = {c["chunk_id"]: c["paper_id"] for c in chunks}
    ranking = {}
    for r in iter_jsonl(path):
        if r["question_id"] in ranking:
            raise ValueError(f"{path}: duplicate {r['question_id']}")
        ranking[r["question_id"]] = r["ranked_chunk_ids"]
    rows = []
    for q in test_questions(out_dir, qa):
        rank = ranking.get(q["question_id"])
        if rank is None or len(rank) < max(KS) or len(set(rank)) != len(rank) or any(c not in paper for c in rank):
            raise ValueError(f"{path}: bad/missing ranking for {q['question_id']}")
        vals = []
        sc = question_scores(set(q["gold_chunk_ids"]), [{c} for c in rank], KS)
        for k in KS:
            vals += list(sc[k])
        vals += [float(q["gold_paper_id"] in {paper[c] for c in rank[:k]}) for k in KS]
        rows.append((q["question_id"], vals))
    first = next(iter(iter_jsonl(path)))
    return first.get("method") or Path(path).stem, [r[0] for r in rows], np.array([r[1] for r in rows])


def scale(m):
    return 1.0 if m.startswith("mrr") else 100.0


def cmd_score(a):
    report, lines = {"split": "test", "metrics": METRICS, "methods": {}}, []
    head = ["hit@1", "recall@1", "hit@3", "recall@3", "mrr@3", "hit@5", "recall@5", "hit@10", "recall@10", "mrr@10", "paper_hit@1"]
    lines.append("| method | " + " | ".join(head) + " |")
    lines.append("|---|" + "---:|" * len(head))
    for path in a.rankings:
        method, qids, x = per_question(a.out_dir, path)
        means = {m: float(x[:, j].mean() * scale(m)) for j, m in enumerate(METRICS)}
        report["methods"][method] = {"file": str(path), "questions": len(qids), "metrics": means}
        lines.append(f"| {method} | " + " | ".join(f"{means[m]:.4f}" if m.startswith("mrr") else f"{means[m]:.2f}" for m in head) + " |")
    report["table_markdown"] = "\n".join(lines)
    print(f"PubMedQA test: {len(qids)} questions; hit/recall %, mrr 0-1; gold = all chunks of the source paper")
    print(report["table_markdown"])
    if a.out:
        write_json(a.out, report)
    return 0


def cmd_compare(a):
    ma, qa_, xa = per_question(a.out_dir, a.a)
    mb, qb, xb = per_question(a.out_dir, a.b)
    if qa_ != qb:
        raise ValueError("question sets differ")
    d = xb - xa
    idx = np.random.default_rng(a.seed).integers(0, len(d), size=(a.n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    out = {"a": ma, "b": mb, "n": len(d), "n_boot": a.n_boot, "diff_b_minus_a": {}}
    print(f"{mb} - {ma}  (n={len(d)}, paired bootstrap {a.n_boot})")
    for j, m in enumerate(METRICS):
        lo, hi = np.percentile(boots[:, j], [2.5, 97.5]) * scale(m)
        mean = d[:, j].mean() * scale(m)
        out["diff_b_minus_a"][m] = {"mean": float(mean), "ci95": [float(lo), float(hi)]}
        print(f"  {m:13s} {mean:+8.3f}  [{lo:+.3f}, {hi:+.3f}]")
    if a.out:
        write_json(a.out, out)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("export"); p.add_argument("--out", type=Path, default=OUT)
    p = sub.add_parser("build")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--support", type=Path, default=SUPPORT, help="vocab.txt / word_emb.npy for the ReaRev loader")
    p.add_argument("--seed", type=int, default=20261109)
    p.add_argument("--dev-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.30)
    p.add_argument("--seed-k", type=int, default=20)
    p.add_argument("--max-cand", type=int, default=150)
    p.add_argument("--knn-k", type=int, default=3)
    for name in ("dense", "bm25"):
        p = sub.add_parser(name)
        p.add_argument("--out-dir", type=Path, default=OUT)
        p.add_argument("--method", default=f"{name}_bge" if name == "dense" else "bm25")
        p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("from-gnn")
    p.add_argument("--gnn-dir", type=Path, default=OUT)
    p.add_argument("--info", type=Path, required=True)
    p.add_argument("--dense", type=Path, required=True, help="dense ranking file (fills the non-candidate tail)")
    p.add_argument("--method", default="gnn_rearev")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("score")
    p.add_argument("rankings", nargs="+")
    p.add_argument("--out-dir", type=Path, default=OUT)
    p.add_argument("--out", type=Path)
    p = sub.add_parser("compare")
    p.add_argument("a"); p.add_argument("b")
    p.add_argument("--out-dir", type=Path, default=OUT)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20261107)
    p.add_argument("--out", type=Path)
    a = ap.parse_args()
    return {"export": cmd_export, "build": cmd_build, "dense": cmd_dense, "bm25": cmd_bm25,
            "from-gnn": cmd_from_gnn, "score": cmd_score, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
