"""Step 2: WebQSP (dense NSM subgraphs) -> the repo's text-RAG format (paragraphs/chunks/qa).

Design (config ``webqsp_u3path_v1``), chosen so that dense RAG and GNN-RAG rank the SAME
objects from the SAME pool and can be scored by the same paragraph-level metrics:

* Source graph: data/data/webqsp (GNN-RAG's own candidate pool; int ids into entities.txt).
* Document (= ``paper_id``): one per (split, topic-entity set). In this data a subgraph is a
  function of the topic set only, so questions with the same topics share one document
  (verified per document by a tuple fingerprint; a mismatch gets its own document).
* Unit (1 node = 1 paragraph = 1 chunk): one unit per CANDIDATE node, i.e. every subgraph node
  that is not a topic entity and not a CVT node. CVT = unnamed MID with >= 1 out-edge in the
  subgraph (Freebase compound-value nodes; they are never answers and GNN-RAG never needs
  them as answers). Unit text = up to ``max_paths`` shortest topic -> node paths, one per line:
      Natalie Portman | film.actor.film -> film.performance.character [film.performance.film: Star Wars] | Padme Amidala
  A CVT hop is merged into "r1 -> r2 [<= cvt_attrs sibling attributes]". "~r" marks an edge
  walked against its direction. Unnamed non-CVT nodes are written as their raw MID.
* Every choice (paths, relation labels, attributes, unit order) uses label-free deterministic
  sort keys; answers are only used afterwards to write the gold columns.
* Gold: units whose node is an answer (``answers[].kb_id``). ``zero_gold_reason``:
  '' | no_answer | answer_not_in_subgraph | answer_only_topic_entity | answer_cvt_node.

Outputs per split in data/processed/webqsp_text/<split>/:
  paragraphs.parquet, chunks.parquet (23 contract fields), chunks_slim.parquet (dense index input),
  qa.parquet (24 contract fields + answer_kb_ids, topic_mids), qa_with_gold.jsonl,
  unit_nodes.parquet (label-free sidecar: chunk -> node, used to map GNN node rankings),
  manifest.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    BGE_TOKENIZER, NAMES_DIR, WEBQSP_DIR, WEBQSP_TEXT_DIR, TokenCounter, config_hash, ensure_dir,
    is_mid, iter_jsonl, log, read_lines, sha1_text, sha256_file, write_json, write_jsonl,
)

CONFIG_NAME = "webqsp_u3path_v1"
UNREACHABLE_HOP = 99
RAW_MID_IN_TEXT = __import__("re").compile(r"(?<![\w.])[mg]\.[0-9a-z_]{2,}(?![\w.])")
PLACEHOLDER_HOP = 0

S = pa.large_string()
PARA_SCHEMA = pa.schema([
    ("para_id", S), ("paper_id", S), ("split", S), ("section_idx", pa.int64()), ("section_name", S),
    ("para_idx", pa.int64()), ("is_abstract", pa.bool_()), ("text_raw", S), ("text_clean", S),
    ("char_start", pa.int64()), ("char_end", pa.int64()), ("n_chars", pa.int64()), ("n_tokens", pa.int64()),
    ("offset_map", S),
])
CHUNK_SCHEMA = pa.schema([
    ("chunk_id", S), ("paper_id", S), ("split", S), ("section_idx", pa.int64()), ("section_name", S),
    ("section_path", S), ("sec_key", S), ("chunk_idx_in_section", pa.int64()), ("chunk_idx_in_paper", pa.int64()),
    ("text", S), ("n_tokens", pa.int64()), ("char_start", pa.int64()), ("char_end", pa.int64()),
    ("para_ids", pa.list_(pa.string())), ("para_spans", S), ("is_abstract", pa.bool_()),
    ("is_truncated", pa.bool_()), ("n_sentences", pa.int64()), ("n_sections", pa.int64()),
    ("prev_chunk_id", S), ("next_chunk_id", S), ("text_sha1", S), ("config_hash", S),
])
SLIM_FIELDS = ["chunk_id", "paper_id", "split", "text", "para_ids", "para_spans", "text_sha1", "config_hash"]
SLIM_SCHEMA = pa.schema([CHUNK_SCHEMA.field(n) for n in SLIM_FIELDS])
NODE_SCHEMA = pa.schema([
    ("chunk_id", S), ("para_id", S), ("paper_id", S), ("split", S), ("node_key", S), ("hop", pa.int64()),
    ("is_named", pa.bool_()), ("is_placeholder", pa.bool_()), ("n_paths", pa.int64()), ("degree", pa.int64()),
])
QA_SCHEMA = pa.schema([
    ("question_id", S), ("paper_id", S), ("split", S), ("question", S), ("n_answers", pa.int64()),
    ("answer_types", S), ("is_answerable", pa.bool_()), ("n_evidence", pa.int64()),
    ("n_evidence_float", pa.int64()), ("n_evidence_mapped", pa.int64()), ("n_evidence_failed", pa.int64()),
    ("evidence_levels", S), ("gold_para_ids", pa.list_(pa.string())), ("n_gold_paras", pa.int64()),
    ("zero_gold_reason", S), ("is_float_only", pa.bool_()), ("free_form_answers", pa.list_(pa.string())),
    ("extractive_spans", pa.list_(pa.string())),
    ("extractive_spans_located", pa.list_(pa.struct([("text", pa.string()), ("para_id", pa.string()),
                                                     ("char_start", pa.int64())]))),
    ("yes_no", pa.bool_()), ("writer_nlp_background", S), ("writer_topic_background", S),
    ("writer_paper_read", S), ("search_query", S),
    ("answer_kb_ids", pa.list_(pa.string())), ("topic_mids", pa.list_(pa.string())),
])


class Buffered:
    """Column-buffered parquet writer (keeps memory flat for ~1M-row splits)."""

    def __init__(self, path: Path, schema: pa.Schema, flush_rows: int = 50_000):
        self.schema, self.flush_rows = schema, flush_rows
        self.writer = pq.ParquetWriter(str(path), schema)
        self.cols = {f.name: [] for f in schema}
        self.n = 0
        self.total = 0

    def add(self, row: dict) -> None:
        for k, v in self.cols.items():
            v.append(row[k])
        self.n += 1
        if self.n >= self.flush_rows:
            self.flush()

    def flush(self) -> None:
        if self.n:
            self.writer.write_table(pa.Table.from_pydict(self.cols, schema=self.schema))
            self.total += self.n
            self.cols = {f.name: [] for f in self.schema}
            self.n = 0

    def close(self) -> int:
        self.flush()
        self.writer.close()
        return self.total


class Renderer:
    def __init__(self, entities, relations, names, cfg):
        self.entities, self.relations, self.names, self.cfg = entities, relations, names, cfg

    def disp(self, n: int) -> str:
        key = self.entities[n]
        if not is_mid(key):
            name = key  # literal value (date, number, code)
        else:
            name = self.names.get(key, key)
        cap = self.cfg["name_char_cap"]
        return name if len(name) <= cap else name[:cap].rstrip() + "..."

    def is_named(self, n: int) -> bool:
        key = self.entities[n]
        return (not is_mid(key)) or key in self.names


def build_doc(tuples: np.ndarray, topics: list[int], R: Renderer, cfg: dict):
    """Return (units, n_nodes) for one subgraph. Each unit: dict(node, hop, text_lines, named, degree)."""
    rel = R.relations
    out_nb = defaultdict(set)
    pair = defaultdict(set)
    adj = defaultdict(set)
    for h, r, t in tuples.tolist():
        if h == t:
            continue
        out_nb[h].add(t)
        pair[(h, t)].add(rel[r])
        pair[(t, h)].add("~" + rel[r])
        adj[h].add(t)
        adj[t].add(h)
    topic_set = set(topics)
    nodes = set(adj) | topic_set
    named = {n: R.is_named(n) for n in nodes}
    cvt = {n for n in nodes if n not in topic_set and not named[n] and is_mid(R.entities[n]) and out_nb.get(n)}
    disp_cache = {}

    def disp(n):
        d = disp_cache.get(n)
        if d is None:
            d = disp_cache[n] = R.disp(n)
        return d

    def nkey(n):
        return (disp(n).lower(), R.entities[n])

    def label(a, b):
        labs = sorted(pair[(a, b)])[: cfg["max_rels_per_pair"]]
        return " / ".join(labs)

    def cvt_attrs(c, exclude):
        items = []
        for u in adj[c]:
            if u in exclude or u == c:
                continue
            items.append((label(c, u), disp(u).lower(), R.entities[u], u))
        items.sort()
        return "; ".join(f"{lab}: {disp(u)}" for lab, _, _, u in items[: cfg["cvt_attrs"]])

    # multi-source undirected BFS from the topic set
    dist = {t: 0 for t in topics if t in nodes}
    frontier = sorted(dist, key=nkey)
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt

    def paths_to(v):
        """Up to max_paths shortest node paths topic -> v, deterministic (label-free) order."""
        res = []

        def back(node, suffix):
            if len(res) >= cfg["max_paths"]:
                return
            if dist[node] == 0:
                res.append([node] + suffix)
                return
            preds = [u for u in adj[node] if dist.get(u) == dist[node] - 1]
            preds.sort(key=nkey)
            for u in preds:
                back(u, [node] + suffix)
                if len(res) >= cfg["max_paths"]:
                    return

        back(v, [])
        return res

    def render(path):
        parts = [disp(path[0])]
        i, k = 1, len(path) - 1
        while i <= k:
            a, b = path[i - 1], path[i]
            lab = label(a, b)
            if b in cvt and i < k:
                c = path[i + 1]
                attrs = cvt_attrs(b, {a, c})
                parts.append(f"{lab} -> {label(b, c)}" + (f" [{attrs}]" if attrs else ""))
                parts.append(disp(c))
                i += 2
            else:
                parts.append(lab)
                parts.append(disp(b))
                i += 1
        return " | ".join(parts)

    units = []
    for v in nodes:
        if v in topic_set or v in cvt:
            continue
        if v in dist:
            lines = [render(p) for p in paths_to(v)]
            hop = dist[v]
        else:  # not connected to any topic (does not occur in the released data; kept for safety)
            lines = [disp(v)]
            hop = UNREACHABLE_HOP
        units.append({"node": v, "hop": hop, "lines": lines, "named": named[v], "degree": len(adj[v])})
    return units, len(nodes)


def fingerprint(tuples: np.ndarray) -> bytes:
    if len(tuples) == 0:
        return b""
    t = tuples[np.lexsort((tuples[:, 2], tuples[:, 1], tuples[:, 0]))]
    return t.astype(np.int64).tobytes()


def convert_split(split, args, R, cfg, chash, tok, out_root):
    src = Path(args.webqsp_dir) / f"{split}.json"
    out = ensure_dir(out_root / split)
    w_para = Buffered(out / "paragraphs.parquet", PARA_SCHEMA)
    w_chunk = Buffered(out / "chunks.parquet", CHUNK_SCHEMA)
    w_slim = Buffered(out / "chunks_slim.parquet", SLIM_SCHEMA)
    w_node = Buffered(out / "unit_nodes.parquet", NODE_SCHEMA)
    ents = R.entities

    docs = {}            # base doc key -> list of (fingerprint, paper_id)
    doc_units = {}       # paper_id -> {node_key: para_id}, chunk ids for gold
    doc_stats = []       # per doc: units
    qa_rows, gold_rows = [], []
    unit_tokens = []
    n_trunc = n_placeholder = n_unreach = n_rawmid_units = n_unnamed_node_units = n_fp_mismatch = 0
    reasons = Counter()
    units_per_q, gold_per_q = [], []

    for rec in iter_jsonl(src):
        qid = rec["id"]
        topics = [int(t) for t in rec["entities"]]
        topic_mids = sorted({ents[t] for t in topics})
        tuples = np.asarray(rec["subgraph"]["tuples"], dtype=np.int64).reshape(-1, 3)
        base = f"wqsp-{split}-" + ("+".join(topic_mids) if topic_mids else f"notopic-{qid}")
        fp = fingerprint(tuples)
        paper_id = None
        for fp0, pid in docs.get(base, []):
            if fp0 == fp:
                paper_id = pid
                break
        if paper_id is None:
            if base in docs:
                n_fp_mismatch += 1
                paper_id = f"{base}~{len(docs[base]) + 1}"
            else:
                paper_id = base
            docs.setdefault(base, []).append((fp, paper_id))
            # ---------- build the document ----------
            units, _ = build_doc(tuples, topics, R, cfg)
            for u in units:
                u["text"] = "\n".join(u["lines"])
            units.sort(key=lambda u: (u["hop"], u["text"], R.disp(u["node"]).lower(), ents[u["node"]]))
            if not units:
                label = ", ".join(R.disp(t) for t in topics) if topics else "no topic entity"
                units = [{"node": None, "hop": PLACEHOLDER_HOP, "lines": [f"(empty subgraph: {label})"],
                          "text": f"(empty subgraph: {label})", "named": False, "degree": 0}]
                n_placeholder += 1
            counts = tok.count_batch([u["text"] for u in units])
            for u, c in zip(units, counts):
                u["truncated"] = False
                if c > cfg["max_unit_tokens"]:
                    while len(u["lines"]) > 1 and c > cfg["max_unit_tokens"]:
                        u["lines"] = u["lines"][:-1]
                        u["text"] = "\n".join(u["lines"])
                        c = tok.count(u["text"])
                    if c > cfg["max_unit_tokens"]:
                        u["text"] = tok.truncate(u["text"], cfg["max_unit_tokens"])
                        c = tok.count(u["text"])
                    u["truncated"] = True
                    n_trunc += 1
                u["n_tokens"] = c
            # ---------- write rows ----------
            node2para, node2chunk = {}, {}
            hop_counter = Counter()
            ids = []
            for i, u in enumerate(units):
                para_id = f"{paper_id}#p{i:04d}"
                j = hop_counter[u["hop"]]
                hop_counter[u["hop"]] += 1
                chunk_id = f"{paper_id}#s{u['hop']:02d}#c{j:03d}"
                ids.append((para_id, chunk_id, j))
            cursor = 0
            for i, (u, (para_id, chunk_id, j)) in enumerate(zip(units, ids)):
                text = u["text"]
                n_chars = len(text)
                start, end = cursor, cursor + n_chars
                cursor = end + 2  # documents join units with a blank line
                hop = u["hop"]
                w_para.add({"para_id": para_id, "paper_id": paper_id, "split": split, "section_idx": hop,
                            "section_name": f"hop{hop}", "para_idx": i, "is_abstract": False,
                            "text_raw": text, "text_clean": text, "char_start": start, "char_end": end,
                            "n_chars": n_chars, "n_tokens": u["n_tokens"], "offset_map": ""})
                row = {"chunk_id": chunk_id, "paper_id": paper_id, "split": split, "section_idx": hop,
                       "section_name": f"hop{hop}", "section_path": f"hop{hop}", "sec_key": f"{hop:02d}",
                       "chunk_idx_in_section": j, "chunk_idx_in_paper": i, "text": text,
                       "n_tokens": u["n_tokens"], "char_start": start, "char_end": end,
                       "para_ids": [para_id], "para_spans": json.dumps([[para_id, 0, n_chars]]),
                       "is_abstract": False, "is_truncated": u["truncated"], "n_sentences": len(u["lines"]),
                       "n_sections": 1,
                       "prev_chunk_id": ids[i - 1][1] if i > 0 else None,
                       "next_chunk_id": ids[i + 1][1] if i + 1 < len(ids) else None,
                       "text_sha1": sha1_text(text), "config_hash": chash}
                w_chunk.add(row)
                w_slim.add({k: row[k] for k in SLIM_FIELDS})
                node_key = ents[u["node"]] if u["node"] is not None else ""
                w_node.add({"chunk_id": chunk_id, "para_id": para_id, "paper_id": paper_id, "split": split,
                            "node_key": node_key, "hop": hop, "is_named": bool(u["named"]),
                            "is_placeholder": u["node"] is None, "n_paths": len(u["lines"]),
                            "degree": u["degree"]})
                unit_tokens.append(u["n_tokens"])
                if u["node"] is not None:
                    node2para[node_key] = para_id
                    node2chunk[node_key] = chunk_id
                    if not u["named"]:
                        n_unnamed_node_units += 1
                    if RAW_MID_IN_TEXT.search(text):
                        n_rawmid_units += 1
                    if hop == UNREACHABLE_HOP:
                        n_unreach += 1
            doc_units[paper_id] = (node2para, node2chunk, len(units))
            doc_stats.append(len(units))
        # ---------- question row ----------
        node2para, node2chunk, n_units = doc_units[paper_id]
        answers = rec.get("answers", [])
        kb_ids = [a["kb_id"] for a in answers]
        texts = [a["text"] if a.get("text") else a["kb_id"] for a in answers]
        gold_nodes = [k for k in dict.fromkeys(kb_ids) if k in node2para]
        gold_para = [node2para[k] for k in gold_nodes]
        gold_chunk = [node2chunk[k] for k in gold_nodes]
        if not kb_ids:
            reason = "no_answer"
        elif gold_para:
            reason = ""
        else:
            topic_keys = {ents[t] for t in topics}
            sub_nodes = set(np.unique(tuples[:, [0, 2]]).tolist()) if len(tuples) else set()
            sub_keys = {ents[n] for n in sub_nodes}
            if any(k in topic_keys for k in kb_ids):
                reason = "answer_only_topic_entity"
            elif any(k in sub_keys for k in kb_ids):
                reason = "answer_cvt_node"
            else:
                reason = "answer_not_in_subgraph"
        reasons[reason or "eligible"] += 1
        kinds = {("kb_entity" if is_mid(k) else "kb_literal") for k in kb_ids}
        atype = "none" if not kinds else (kinds.pop() if len(kinds) == 1 else "kb_mixed")
        row = {"question_id": qid, "paper_id": paper_id, "split": split, "question": rec["question"],
               "n_answers": len(kb_ids), "answer_types": atype, "is_answerable": bool(kb_ids),
               "n_evidence": len(gold_para), "n_evidence_float": 0, "n_evidence_mapped": len(gold_nodes),
               "n_evidence_failed": len(set(kb_ids)) - len(gold_nodes), "evidence_levels": "kg_node",
               "gold_para_ids": gold_para, "n_gold_paras": len(gold_para), "zero_gold_reason": reason,
               "is_float_only": False, "free_form_answers": texts, "extractive_spans": [],
               "extractive_spans_located": [], "yes_no": None, "writer_nlp_background": "",
               "writer_topic_background": "", "writer_paper_read": "", "search_query": "",
               "answer_kb_ids": kb_ids, "topic_mids": topic_mids}
        qa_rows.append(row)
        gold_rows.append({**row, "gold_chunk_ids": gold_chunk, "n_gold_chunks": len(gold_chunk)})
        units_per_q.append(n_units)
        if reason == "":
            gold_per_q.append(len(gold_para))

    n_rows = w_chunk.close()
    w_para.close(), w_slim.close(), w_node.close()
    pq.write_table(pa.Table.from_pylist(qa_rows, schema=QA_SCHEMA), out / "qa.parquet")
    write_jsonl(out / "qa_with_gold.jsonl", gold_rows)

    def pct(xs, q):
        return float(np.percentile(xs, q)) if xs else None

    ceilings = {}
    for k in (1, 3, 5, 10, 20):
        ceilings[f"recall@{k}"] = (round(100 * statistics.mean(min(k, g) / g for g in gold_per_q), 2)
                                   if gold_per_q else None)
    stats = {
        "split": split, "questions": len(qa_rows), "documents": len(doc_stats), "units": n_rows,
        "tokens_total": int(sum(unit_tokens)),
        "tokens_per_unit": {"mean": round(statistics.mean(unit_tokens), 2), "median": pct(unit_tokens, 50),
                            "p99": pct(unit_tokens, 99), "max": max(unit_tokens)},
        "units_per_question": {"mean": round(statistics.mean(units_per_q), 2), "median": pct(units_per_q, 50),
                               "p90": pct(units_per_q, 90), "max": max(units_per_q)},
        "units_per_document": {"mean": round(statistics.mean(doc_stats), 2), "median": pct(doc_stats, 50),
                               "max": max(doc_stats)},
        "zero_gold_reason": dict(reasons),
        "eligible_questions": reasons.get("eligible", 0),
        "gold_units_per_eligible_question": ({"mean": round(statistics.mean(gold_per_q), 2),
                                              "median": pct(gold_per_q, 50), "p90": pct(gold_per_q, 90),
                                              "max": max(gold_per_q)} if gold_per_q else None),
        "perfect_ranking_ceiling": ceilings,
        "truncated_units": n_trunc, "placeholder_units": n_placeholder, "unreachable_units": n_unreach,
        "units_with_raw_mid_text": n_rawmid_units,
        "units_whose_node_is_unnamed": n_unnamed_node_units,
        "fingerprint_mismatch_docs": n_fp_mismatch,
    }
    manifest = {
        "config_name": CONFIG_NAME, "config": cfg, "config_hash": chash,
        "tokenizer": tok.backend, "split": split, "chunks": n_rows,
        "files": {name: sha256_file(out / name) for name in
                  ("paragraphs.parquet", "chunks.parquet", "chunks_slim.parquet", "qa.parquet",
                   "qa_with_gold.jsonl", "unit_nodes.parquet")},
        "stats": stats,
    }
    write_json(out / "manifest.json", manifest)
    log(json.dumps(stats, indent=1))
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=["dev", "test"], choices=["train", "dev", "test"])
    ap.add_argument("--webqsp-dir", default=str(WEBQSP_DIR))
    ap.add_argument("--names", default=str(NAMES_DIR / "mid2name.tsv"))
    ap.add_argument("--out", default=str(WEBQSP_TEXT_DIR))
    ap.add_argument("--max-paths", type=int, default=3)
    ap.add_argument("--max-rels-per-pair", type=int, default=2)
    ap.add_argument("--cvt-attrs", type=int, default=3)
    ap.add_argument("--name-char-cap", type=int, default=200)
    ap.add_argument("--max-unit-tokens", type=int, default=510)
    args = ap.parse_args()

    wq = Path(args.webqsp_dir)
    entities = read_lines(wq / "entities.txt")
    relations = read_lines(wq / "relations.txt")
    names = {}
    with open(args.names, encoding="utf-8") as f:
        next(f)
        for line in f:
            mid, name, _src = line.rstrip("\n").split("\t")
            names[mid] = name
    log(f"loaded {len(entities):,} entities, {len(relations):,} relations, {len(names):,} names")
    cfg = {
        "unit": "candidate_node_paths", "doc_scope": "split+topic_set",
        "cvt_rule": "unnamed MID with >=1 out-edge in the subgraph, not a topic entity",
        "max_paths": args.max_paths, "max_rels_per_pair": args.max_rels_per_pair,
        "cvt_attrs": args.cvt_attrs, "inverse_mark": "~", "unnamed_render": "raw_mid",
        "name_char_cap": args.name_char_cap, "max_unit_tokens": args.max_unit_tokens,
        "path_order": "predecessor (name.lower, key)", "unit_order": "hop, text, name.lower, key",
        "names_sha256": sha256_file(args.names),
        "source_sha256": {s: sha256_file(wq / f"{s}.json") for s in ("train", "dev", "test")},
        "converter_version": "1.0.0",
    }
    chash = config_hash(cfg)
    tok = TokenCounter(BGE_TOKENIZER)
    R = Renderer(entities, relations, names, cfg)
    out_root = ensure_dir(args.out)
    summary = {}
    for split in args.splits:
        log(f"=== {split} ===")
        summary[split] = convert_split(split, args, R, cfg, chash, tok, out_root)
    write_json(out_root / "summary.json", {"config_name": CONFIG_NAME, "config_hash": chash, "splits": summary})
    return 0


if __name__ == "__main__":
    sys.exit(main())
