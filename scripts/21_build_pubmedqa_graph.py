"""Build chunks / graph edges / QA for the expanded PubMedQA citation graph.

Consumes the output of ``20_expand_pubmedqa_corpus.py``:
    papers.parquet  (paper_id, role, pmid, title, year, abstract, has_abstract)
    edges.parquet   (src_paper, dst_paper, edge_type='cites')

Produces
--------
chunks.parquet        contract-compatible chunk table (split='test')
graph_edges.parquet   unified edges: cites (paper-level) + adjacency (chunk-level)
qa.parquet            question + gold_paper_id + gold_chunk_ids + final_decision
report.json           corpus / graph / qa statistics

Retrieval pool is the WHOLE corpus (no paper scoping); gold is the source
paper's chunk(s) -- this is the CG-RAG-style cross-document task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from tokenizers import Tokenizer

SENT_RE = re.compile(r"(?<=[.!?])\s+")
CFG = {
    "chunk_size": 512,
    "overlap_tokens": 102,
    "min_chunk_tokens": 64,
    "section_boundary": "hard",
    "chunker_version": "pubmedqa-graph-1.0",
}


def cfg_hash() -> str:
    return hashlib.sha1(repr(sorted(CFG.items())).encode()).hexdigest()[:12]


def split_sentences(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in SENT_RE.finditer(text):
        end = m.start()
        if end > start:
            spans.append((start, end))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans or [(0, len(text))]


def pack_chunks(text: str, tok: Tokenizer, max_tokens: int, overlap: int):
    """Sentence-packed chunks with token overlap. Returns [(a, b, n_tokens, truncated)]."""
    if not text.strip():
        return []
    sents = split_sentences(text)
    out = []
    i, n_s = 0, len(sents)
    while i < n_s:
        a = sents[i][0]
        # a single sentence longer than the budget -> hard character split
        if len(tok.encode(text[a : sents[i][1]]).ids) > max_tokens:
            lo, hi = a, sents[i][1]
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if len(tok.encode(text[a:mid]).ids) <= max_tokens:
                    lo = mid
                else:
                    hi = mid
            b = lo
            out.append((a, b, len(tok.encode(text[a:b]).ids), True))
            sents[i] = (b, sents[i][1])
            if b >= sents[i][1]:
                i += 1
            continue
        # pack as many whole sentences as fit
        j = i
        while j + 1 < n_s and len(tok.encode(text[a : sents[j + 1][1]]).ids) <= max_tokens:
            j += 1
        b = sents[j][1]
        out.append((a, b, len(tok.encode(text[a:b]).ids), False))
        if j + 1 >= n_s:
            break
        # overlap: back up to the sentence where the tail reaches ~overlap tokens
        k = j
        while k > i and len(tok.encode(text[sents[k][0] : b]).ids) < overlap:
            k -= 1
        i = max(k, i + 1)
    return out


def para_rows_for_paper(paper: pd.Series, raw: dict) -> list[dict]:
    """Build paragraph rows: source -> CONTEXTS, reference -> abstract."""
    rows = []
    if paper["role"] == "source" and paper["pmid"] in raw:
        rec = raw[paper["pmid"]]
        ctxs = rec.get("CONTEXTS") or []
        labels = rec.get("LABELS") or []
        for i, txt in enumerate(ctxs):
            rows.append(
                {
                    "section_idx": i,
                    "section_name": labels[i] if i < len(labels) else f"Section {i}",
                    "is_abstract": False,
                    "text_clean": (txt or "").strip(),
                }
            )
    else:
        txt = (paper["abstract"] or "").strip()
        if not txt:
            txt = (paper["title"] or "").strip()
        rows.append({"section_idx": -1, "section_name": "Abstract", "is_abstract": True, "text_clean": txt})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", default="data/processed/pubmedqa_graph")
    ap.add_argument("--raw", default="data/raw/pubmedqa/ori_pqal.json")
    ap.add_argument("--tokenizer", default="data/raw/bge_tok/tokenizer.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    corpus = Path(args.corpus_dir)
    out = Path(args.out) if args.out else corpus
    out.mkdir(parents=True, exist_ok=True)

    papers = pd.read_parquet(corpus / "papers.parquet")
    cites = pd.read_parquet(corpus / "edges.parquet")
    raw = json.loads(Path(args.raw).read_text(encoding="utf-8"))

    if args.limit:
        src_ids = set(papers[papers.role == "source"].head(args.limit).paper_id)
        papers = papers[papers.paper_id.isin(src_ids | set(papers[papers.role == "reference"].paper_id))]
        # keep refs that are cited by the limited sources (best effort)

    tok = Tokenizer.from_file(args.tokenizer)
    h = cfg_hash()
    max_t, ov = CFG["chunk_size"], CFG["overlap_tokens"]

    chunk_rows: list[dict] = []
    paper_chunks: dict[str, list[str]] = {}

    for p in papers.itertuples(index=False):
        prs = para_rows_for_paper(pd.Series(p._asdict()), raw)
        flat = "\n\n".join(r["text_clean"] for r in prs)
        flat_off, acc = [], 0
        for r in prs:
            flat_off.append(acc)
            acc += len(r["text_clean"]) + 2

        per_sec: dict[int, int] = {}
        for pi, r in enumerate(prs):
            text = r["text_clean"]
            if not text:
                continue
            pid = f"{p.paper_id}#p{pi:04d}"
            for (a, b, n, trunc) in pack_chunks(text, tok, max_t, ov):
                sec = r["section_idx"]
                idx = per_sec.get(sec, 0)
                per_sec[sec] = idx + 1
                sec_key = "abs" if sec == -1 else f"{sec:02d}"
                cid = f"{p.paper_id}#s{sec_key}#c{idx:03d}"
                chunk_rows.append(
                    {
                        "chunk_id": cid,
                        "paper_id": p.paper_id,
                        "split": "test",
                        "section_idx": sec,
                        "section_name": r["section_name"],
                        "section_path": r["section_name"],
                        "sec_key": sec_key,
                        "chunk_idx_in_section": idx,
                        "chunk_idx_in_paper": len(paper_chunks.get(p.paper_id, [])),
                        "text": text[a:b],
                        "n_tokens": n,
                        "char_start": flat_off[pi] + a,
                        "char_end": flat_off[pi] + b,
                        "para_ids": [pid],
                        "para_spans": json.dumps([[pid, a, b]]),
                        "is_abstract": bool(r["is_abstract"]),
                        "is_truncated": bool(trunc),
                        "n_sentences": len(split_sentences(text[a:b])),
                        "n_sections": 1,
                        "prev_chunk_id": None,
                        "next_chunk_id": None,
                        "text_sha1": hashlib.sha1(text[a:b].encode()).hexdigest(),
                        "config_hash": h,
                    }
                )
                paper_chunks.setdefault(p.paper_id, []).append(cid)

    chunks = pd.DataFrame(chunk_rows)
    # link prev/next within paper (O(n) via positional index)
    pos = {cid: i for i, cid in enumerate(chunks.chunk_id)}
    for pid, cids in paper_chunks.items():
        for i, cid in enumerate(cids):
            j = pos[cid]
            if i > 0:
                chunks.at[j, "prev_chunk_id"] = cids[i - 1]
            if i + 1 < len(cids):
                chunks.at[j, "next_chunk_id"] = cids[i + 1]

    # ---- graph edges ---------------------------------------------------
    paper_ids = set(papers.paper_id)
    n_cites_raw = len(cites)
    cites = cites[cites.src_paper.isin(paper_ids) & cites.dst_paper.isin(paper_ids)]
    edges = [{"src_id": r.src_paper, "dst_id": r.dst_paper, "edge_type": "cites", "level": "paper", "weight": 1.0}
             for r in cites.itertuples()]
    for r in chunks.itertuples():
        if r.next_chunk_id:
            edges.append({"src_id": r.chunk_id, "dst_id": r.next_chunk_id, "edge_type": "adjacency", "level": "chunk", "weight": 1.0})
    edges_df = pd.DataFrame(edges)

    # ---- QA ------------------------------------------------------------
    src = papers[papers.role == "source"][["paper_id", "pmid"]].copy()
    qa_rows = []
    for s in src.itertuples():
        rec = raw.get(s.pmid)
        if not rec:
            continue
        gold = paper_chunks.get(s.paper_id, [])
        if not gold:
            continue
        qa_rows.append(
            {
                "question_id": s.pmid,
                "question": rec.get("QUESTION", ""),
                "source_paper_id": s.paper_id,
                "gold_paper_id": s.paper_id,
                "gold_chunk_ids": gold,
                "final_decision": rec.get("final_decision"),
            }
        )
    qa = pd.DataFrame(qa_rows)

    chunks.to_parquet(out / "chunks.parquet", index=False)
    edges_df.to_parquet(out / "graph_edges.parquet", index=False)
    qa.to_parquet(out / "qa.parquet", index=False)

    report = {
        "papers": int(papers.paper_id.nunique()),
        "chunks": int(len(chunks)),
        "chunks_per_paper_mean": round(len(chunks) / max(papers.paper_id.nunique(), 1), 2),
        "cites_edges": int((edges_df.edge_type == "cites").sum()),
        "cites_dropped_dangling": n_cites_raw - int((edges_df.edge_type == "cites").sum()),
        "adjacency_edges": int((edges_df.edge_type == "adjacency").sum()),
        "qa_questions": int(len(qa)),
        "qa_with_gold": int((qa.gold_chunk_ids.str.len() > 0).sum()) if len(qa) else 0,
        "config_hash": h,
    }
    (out / "report_graph.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[REPORT]")
    for k, v in report.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
