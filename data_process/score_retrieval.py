"""Unified retrieval scorer: one metric definition for every method and every dataset.

All methods are reduced to the same object: a RANKING of chunk_ids of the text-format dataset
(QASPER chunks, or WebQSP-text units). Scores are paragraph-level, exactly as member 2's
dense_rag.retrieval_metrics (union of para_ids of the top-K chunks vs gold_para_ids):

  Hit@K    = 1 if any gold paragraph is covered by the top-K chunks
  Recall@K = |gold ∩ ∪ para_ids(top-K)| / |gold|
  MRR@K    = 1 / rank of the first chunk covering a gold paragraph (0 if none in top-K)

Macro averages over a fixed denominator: questions with zero_gold_reason == '' (+ optional
reasons scored as misses, e.g. WebQSP 'answer_not_in_subgraph' for the 1,628-question view).
The candidate pool of a question is all chunks of its paper_id (= dense ``scope=paper``).

Ranking file (JSON lines), produced by the adapters below or by any custom method:
  {"question_id": "...", "method": "...", "ranked_chunk_ids": ["...", ...], "scores": [...]}

Subcommands
  score      score one or more ranking files              (+ --random-floor N, --named-only)
  compare    paired bootstrap CI for B - A
  from-dense convert member 2 predictions.jsonl -> ranking file
  from-gnn   convert a GNN-RAG <experiment>_test.info (run with --eps 2 for full rankings)
  baseline   label-free floors: random | order | degree (WebQSP) | bm25
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    QASPER_CHUNKS, REPO, WEBQSP_TEXT_DIR, iter_jsonl, log, read_lines, write_json, write_jsonl,
)

PRESETS = {
    "qasper": {"qa": REPO / "data/processed/qa.parquet", "chunks": QASPER_CHUNKS, "nodes": None},
    "webqsp": {"qa": "{d}/qa.parquet", "chunks": "{d}/chunks_slim.parquet", "nodes": "{d}/unit_nodes.parquet"},
}


# --------------------------------------------------------------------------- data
class Dataset:
    def __init__(self, name, split, qa=None, chunks=None, nodes=None, text_dir=None):
        p = PRESETS[name]
        d = Path(text_dir) if text_dir else WEBQSP_TEXT_DIR / split
        self.name, self.split = name, split
        self.qa_path = Path(qa) if qa else Path(str(p["qa"]).format(d=d))
        self.chunks_path = Path(chunks) if chunks else Path(str(p["chunks"]).format(d=d))
        self.nodes_path = Path(nodes) if nodes else (Path(str(p["nodes"]).format(d=d)) if p["nodes"] else None)
        cols = ["question_id", "paper_id", "split", "question", "gold_para_ids", "zero_gold_reason"]
        self.qa = [q for q in pq.read_table(self.qa_path, columns=cols).to_pylist() if q["split"] == split]
        if not self.qa:
            raise ValueError(f"no questions for split {split} in {self.qa_path}")
        self.qmap = {q["question_id"]: q for q in self.qa}
        ccols = ["chunk_id", "paper_id", "para_ids", "split"]
        crows = pq.read_table(self.chunks_path, columns=ccols).to_pylist()
        papers = {q["paper_id"] for q in self.qa}
        self.chunk_paras, self.chunk_paper, self.pool = {}, {}, {}
        for c in crows:
            if c["paper_id"] not in papers:
                continue
            self.chunk_paras[c["chunk_id"]] = frozenset(c["para_ids"])
            self.chunk_paper[c["chunk_id"]] = c["paper_id"]
            self.pool.setdefault(c["paper_id"], []).append(c["chunk_id"])
        self._named = None

    def named_units(self):
        if self._named is None:
            if not self.nodes_path:
                raise ValueError("--named-only needs a unit_nodes.parquet (WebQSP-text only)")
            t = pq.read_table(self.nodes_path, columns=["chunk_id", "is_named"]).to_pylist()
            self._named = {r["chunk_id"] for r in t if r["is_named"]}
        return self._named


def load_rankings(path):
    out = {}
    method = None
    for r in iter_jsonl(path):
        if r["question_id"] in out:
            raise ValueError(f"{path}: duplicate question {r['question_id']}")
        out[r["question_id"]] = r["ranked_chunk_ids"]
        method = method or r.get("method")
    return out, method or Path(path).stem


# --------------------------------------------------------------------------- metrics
def question_scores(gold, ranked_sets, ks):
    res = {}
    for k in ks:
        top = ranked_sets[:k]
        covered = set().union(*top) if top else set()
        inter = gold & covered
        rr = next((1.0 / r for r, s in enumerate(top, 1) if gold & s), 0.0)
        res[k] = (1.0 if inter else 0.0, len(inter) / len(gold), rr)
    return res


def per_question(ds, ranking, ks, miss_reasons=(), named_only=False, allow_short=False, strict=True):
    """Return (rows, excluded Counter). Each row: qid + metric values for every K."""
    rows, excluded = [], Counter()
    named = ds.named_units() if named_only else None
    kmax = max(ks)
    for q in ds.qa:
        qid, reason = q["question_id"], q["zero_gold_reason"]
        if reason != "" and reason not in miss_reasons:
            excluded[reason] += 1
            continue
        pool = ds.pool.get(q["paper_id"], [])
        if named is not None:
            pool = [c for c in pool if c in named]
        pool_set = set(pool)
        gold = set(q["gold_para_ids"])
        if named is not None:
            gold = gold & set().union(*(ds.chunk_paras[c] for c in pool)) if pool else set()
        rank = ranking.get(qid)
        if rank is None:
            if strict:
                raise ValueError(f"missing ranking for {qid}")
            rank = []
        seen = set()
        clean = []
        for c in rank:
            if c in seen:
                raise ValueError(f"{qid}: duplicate chunk {c}")
            seen.add(c)
            if c not in ds.chunk_paper or ds.chunk_paper[c] != q["paper_id"]:
                raise ValueError(f"{qid}: chunk {c} is outside the question's pool ({q['paper_id']})")
            if c in pool_set:
                clean.append(c)
        need = min(kmax, len(pool))
        if len(clean) < need and not allow_short:
            raise ValueError(f"{qid}: ranking has {len(clean)} pool chunks, needs {need} "
                             f"(use --allow-short to score truncated rankings as lower bounds)")
        row = {"question_id": qid, "zero_gold_reason": reason, "pool": len(pool), "n_gold": len(gold)}
        if reason != "" or not gold:
            for k in ks:
                row[f"hit@{k}"], row[f"recall@{k}"], row[f"mrr@{k}"] = 0.0, 0.0, 0.0
        else:
            sets = [ds.chunk_paras[c] for c in clean[:kmax]]
            for k, (h, r, m) in question_scores(gold, sets, ks).items():
                row[f"hit@{k}"], row[f"recall@{k}"], row[f"mrr@{k}"] = h, r, m
        rows.append(row)
    return rows, excluded


def summarize(rows, ks):
    out = {}
    n = len(rows)
    for k in ks:
        out[str(k)] = {"hit": 100 * statistics.mean(r[f"hit@{k}"] for r in rows) if n else None,
                       "recall": 100 * statistics.mean(r[f"recall@{k}"] for r in rows) if n else None,
                       "mrr": statistics.mean(r[f"mrr@{k}"] for r in rows) if n else None}
    return out


def random_floor(ds, ks, n_perm, seed, miss_reasons, named_only):
    rng = random.Random(seed)
    acc = None
    for _ in range(n_perm):
        ranking = {}
        named = ds.named_units() if named_only else None
        for q in ds.qa:
            pool = ds.pool.get(q["paper_id"], [])
            if named is not None:
                pool = [c for c in pool if c in named]
            ranking[q["question_id"]] = rng.sample(pool, min(max(ks), len(pool)))
        rows, _ = per_question(ds, ranking, ks, miss_reasons, named_only, allow_short=True)
        s = summarize(rows, ks)
        if acc is None:
            acc = {k: {m: 0.0 for m in v} for k, v in s.items()}
        for k, v in s.items():
            for m, x in v.items():
                acc[k][m] += x / n_perm
    return acc


DEFINITIONS = {
    "hit": "1 if any gold paragraph is covered by the top-K chunks (union of para_ids), macro %",
    "recall": "|gold paragraphs ∩ union of para_ids of top-K chunks| / |gold paragraphs|, macro %",
    "mrr": "1/rank of the first top-K chunk covering a gold paragraph, 0 if none; macro mean in [0,1]",
    "denominator": "questions with zero_gold_reason == '' plus any --count-as-miss reasons (scored 0)",
    "pool": "all chunks of the question's paper_id in the chunk table (dense scope=paper)",
}


def fmt_table(name_to_summary, ks):
    head = "| method | " + " | ".join(f"Hit@{k} | R@{k} | MRR@{k}" for k in ks) + " |"
    sep = "|---|" + "---:|" * (3 * len(ks))
    lines = [head, sep]
    for name, s in name_to_summary.items():
        cells = []
        for k in ks:
            v = s[str(k)]
            cells += [f"{v['hit']:.2f}", f"{v['recall']:.2f}", f"{v['mrr']:.4f}"]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- commands
def add_dataset_args(ap):
    ap.add_argument("--dataset", required=True, choices=sorted(PRESETS))
    ap.add_argument("--split", required=True, help="qasper: dev/train; webqsp: dev/test")
    ap.add_argument("--text-dir", default=None, help="webqsp-text split dir (default data/processed/webqsp_text/<split>)")
    ap.add_argument("--qa", default=None)
    ap.add_argument("--chunks", default=None)
    ap.add_argument("--nodes", default=None)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--count-as-miss", nargs="*", default=[],
                    help="zero_gold_reason values kept in the denominator with score 0 "
                         "(webqsp: answer_not_in_subgraph -> the 1,628 answerable view)")
    ap.add_argument("--named-only", action="store_true", help="webqsp sensitivity: pool = units of named nodes")


def get_ds(a):
    return Dataset(a.dataset, a.split, a.qa, a.chunks, a.nodes, a.text_dir)


def cmd_score(a):
    ds = get_ds(a)
    ks = sorted(set(a.ks))
    summaries, report = {}, {"dataset": a.dataset, "split": a.split, "ks": ks, "definitions": DEFINITIONS,
                             "count_as_miss": a.count_as_miss, "named_only": a.named_only, "methods": {}}
    for path in a.rankings:
        ranking, method = load_rankings(path)
        rows, excluded = per_question(ds, ranking, ks, tuple(a.count_as_miss), a.named_only,
                                      a.allow_short, strict=not a.allow_missing)
        s = summarize(rows, ks)
        summaries[method] = s
        report["methods"][method] = {"file": str(path), "scored_questions": len(rows),
                                     "excluded_by_reason": dict(excluded), "metrics": s}
        if a.per_question_dir:
            d = Path(a.per_question_dir)
            d.mkdir(parents=True, exist_ok=True)
            with open(d / f"{method}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    if a.random_floor:
        s = random_floor(ds, ks, a.random_floor, a.seed, tuple(a.count_as_miss), a.named_only)
        summaries[f"random (mean of {a.random_floor})"] = s
        report["methods"]["random"] = {"permutations": a.random_floor, "metrics": s}
    any_rows = next(iter(report["methods"].values()))
    report["table_markdown"] = fmt_table(summaries, ks)
    print(f"{a.dataset} / {a.split}: scored {any_rows.get('scored_questions', '?')} questions")
    print(report["table_markdown"])
    if a.out:
        write_json(a.out, report)
    return 0


def cmd_compare(a):
    ds = get_ds(a)
    ks = sorted(set(a.ks))
    ra, ma = load_rankings(a.a)
    rb, mb = load_rankings(a.b)
    rows_a, _ = per_question(ds, ra, ks, tuple(a.count_as_miss), a.named_only, a.allow_short)
    rows_b, _ = per_question(ds, rb, ks, tuple(a.count_as_miss), a.named_only, a.allow_short)
    if [r["question_id"] for r in rows_a] != [r["question_id"] for r in rows_b]:
        raise ValueError("question sets differ")
    rng = np.random.default_rng(a.seed)
    n = len(rows_a)
    idx = rng.integers(0, n, size=(a.n_boot, n))
    out = {"a": ma, "b": mb, "n": n, "n_boot": a.n_boot, "diff_b_minus_a": {}}
    print(f"{mb} - {ma}  (n={n}, paired bootstrap {a.n_boot})")
    for k in ks:
        for m in ("hit", "recall", "mrr"):
            key = f"{m}@{k}"
            d = np.array([rb_[key] - ra_[key] for ra_, rb_ in zip(rows_a, rows_b)])
            boots = d[idx].mean(axis=1)
            scale = 1.0 if m == "mrr" else 100.0
            lo, hi = np.percentile(boots, [2.5, 97.5]) * scale
            mean = d.mean() * scale
            out["diff_b_minus_a"][key] = {"mean": mean, "ci95": [lo, hi]}
            print(f"  {key:10s} {mean:+8.3f}  [{lo:+.3f}, {hi:+.3f}]")
    if a.out:
        write_json(a.out, out)
    return 0


def cmd_from_dense(a):
    rows = []
    for p in iter_jsonl(a.predictions):
        if p.get("status", "ok") != "ok":
            log(f"warning: {p['question_id']} status={p.get('status')}")
        hits = sorted(p.get("retrieved_chunks", []), key=lambda h: h["rank"])
        rows.append({"question_id": p["question_id"], "method": a.method,
                     "ranked_chunk_ids": [h["chunk_id"] for h in hits],
                     "scores": [h.get("score") for h in hits]})
    n = write_jsonl(a.out, rows)
    log(f"wrote {n} rankings to {a.out}")
    return 0


def cmd_from_gnn(a):
    gnn_dir = Path(a.gnn_dir)
    records = [r for r in iter_jsonl(gnn_dir / f"{a.gnn_file}.json") if len(r["entities"]) > 0]
    infos = list(iter_jsonl(a.info))
    if len(infos) != len(records):
        raise ValueError(f"{len(infos)} .info lines vs {len(records)} loadable records in {a.gnn_file}.json")
    unit_of = None
    if a.mode == "webqsp":
        text_dir = Path(a.text_dir)
        qpaper = {q["question_id"]: q["paper_id"] for q in
                  pq.read_table(text_dir / "qa.parquet", columns=["question_id", "paper_id"]).to_pylist()}
        unit_of = {}
        for r in pq.read_table(text_dir / "unit_nodes.parquet",
                               columns=["chunk_id", "paper_id", "node_key", "is_placeholder"]).to_pylist():
            if not r["is_placeholder"]:
                unit_of[(r["paper_id"], r["node_key"])] = r["chunk_id"]
    rows, dropped = [], Counter()
    for rec, info in zip(records, infos):
        qid = rec["id"]
        if "id" in info and info["id"] != qid:
            raise ValueError(f"alignment error: info id {info['id']} vs record {qid}")
        ranked, scores = [], []
        for key, prob in info.get("cand", []):
            if a.mode == "qasper":
                q2, _, chunk_id = key.partition("|")
                if q2 != qid:
                    raise ValueError(f"alignment error: candidate {key} in record {qid}")
            else:
                chunk_id = unit_of.get((qpaper[qid], key))
                if chunk_id is None:  # CVT / topic / non-unit node: not a candidate unit
                    dropped["non_unit_node"] += 1
                    continue
            ranked.append(chunk_id)
            scores.append(float(prob))
        rows.append({"question_id": qid, "method": a.method, "ranked_chunk_ids": ranked, "scores": scores})
    n = write_jsonl(a.out, rows)
    log(f"wrote {n} rankings to {a.out}; dropped {dict(dropped)}")
    return 0


TOKEN = re.compile(r"\w+")


def bm25_rank(query, docs, k1=1.2, b=0.75):
    toks = [TOKEN.findall(d.lower()) for d in docs]
    n = len(docs)
    avg = sum(map(len, toks)) / max(n, 1)
    df = Counter(t for ts in toks for t in set(ts))
    q = TOKEN.findall(query.lower())
    scores = []
    for ts in toks:
        tf = Counter(ts)
        s = 0.0
        for t in q:
            if t in tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(ts) / avg))
        scores.append(s)
    return scores


def cmd_baseline(a):
    ds = get_ds(a)
    rng = random.Random(a.seed)
    extra = {}
    if a.kind == "degree":
        if not ds.nodes_path:
            raise ValueError("degree baseline needs unit_nodes.parquet (webqsp)")
        extra = {r["chunk_id"]: r["degree"] for r in
                 pq.read_table(ds.nodes_path, columns=["chunk_id", "degree"]).to_pylist()}
    texts = {}
    if a.kind == "bm25":
        texts = {r["chunk_id"]: r["text"] for r in
                 pq.read_table(ds.chunks_path, columns=["chunk_id", "text"]).to_pylist() if r["chunk_id"] in ds.chunk_paper}
    rows = []
    for q in ds.qa:
        pool = list(ds.pool.get(q["paper_id"], []))  # file order = reading / canonical order
        if a.kind == "random":
            rng.shuffle(pool)
            scores = [None] * len(pool)
        elif a.kind == "order":
            scores = [None] * len(pool)
        elif a.kind == "degree":
            pool.sort(key=lambda c: -extra[c])  # stable: ties keep canonical order
            scores = [extra[c] for c in pool]
        else:
            s = bm25_rank(q["question"], [texts[c] for c in pool])
            order = sorted(range(len(pool)), key=lambda i: -s[i])
            pool = [pool[i] for i in order]
            scores = [s[i] for i in order]
        rows.append({"question_id": q["question_id"], "method": a.kind,
                     "ranked_chunk_ids": pool[: a.depth], "scores": scores[: a.depth]})
    n = write_jsonl(a.out, rows)
    log(f"wrote {n} {a.kind} rankings to {a.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("score")
    add_dataset_args(p)
    p.add_argument("rankings", nargs="+")
    p.add_argument("--allow-short", action="store_true", help="score rankings shorter than min(K, pool) (lower bounds)")
    p.add_argument("--allow-missing", action="store_true", help="questions without a ranking score 0")
    p.add_argument("--random-floor", type=int, default=0, help="also report the mean of N random rankings")
    p.add_argument("--seed", type=int, default=20261107)
    p.add_argument("--per-question-dir", default=None)
    p.add_argument("--out", default=None)

    p = sub.add_parser("compare")
    add_dataset_args(p)
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20261107)
    p.add_argument("--allow-short", action="store_true")
    p.add_argument("--out", default=None)

    p = sub.add_parser("from-dense")
    p.add_argument("--predictions", required=True)
    p.add_argument("--method", default="dense_bge")
    p.add_argument("--out", required=True)

    p = sub.add_parser("from-gnn")
    p.add_argument("--mode", required=True, choices=["qasper", "webqsp"])
    p.add_argument("--gnn-dir", required=True, help="the GNN data folder (holds <gnn-file>.json)")
    p.add_argument("--gnn-file", default="test")
    p.add_argument("--info", required=True, help="<checkpoint_dir>/<experiment>_test.info")
    p.add_argument("--text-dir", default=None, help="webqsp: data/processed/webqsp_text/<split>")
    p.add_argument("--method", default="gnn_rearev")
    p.add_argument("--out", required=True)

    p = sub.add_parser("baseline")
    add_dataset_args(p)
    p.add_argument("--kind", required=True, choices=["random", "order", "degree", "bm25"])
    p.add_argument("--depth", type=int, default=100)
    p.add_argument("--seed", type=int, default=20261107)
    p.add_argument("--out", required=True)

    a = ap.parse_args()
    if a.cmd == "from-gnn" and a.mode == "webqsp" and not a.text_dir:
        ap.error("--text-dir is required for --mode webqsp")
    return {"score": cmd_score, "compare": cmd_compare, "from-dense": cmd_from_dense,
            "from-gnn": cmd_from_gnn, "baseline": cmd_baseline}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
