"""A5/A6 / script 03: build chunks for one config, validate, and map gold evidence.

Usage:
    python scripts/03_build_chunks.py --chunk-size 512 --overlap 0.2 --name size512_ov0.2

Outputs (per config):
    data/processed/chunks/<name>/chunks.parquet
    data/processed/chunks/<name>/chunks.jsonl
    data/processed/chunks/<name>/qa_with_gold.jsonl
    data/processed/chunks/<name>/manifest.json
    docs/chunking_stats_<name>.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rag_data.chunker import ChunkConfig, chunk_paper  # noqa: E402
from rag_data.tokenizer import DEFAULT_TOKENIZER_FILE, TokenCounter  # noqa: E402


def jsonable(obj):
    """Convert pandas/numpy containers into plain JSON types, recursively.

    This is not cosmetic. ``df.to_dict("records")`` hands back ndarrays for list columns, and
    ``json.dumps(..., default=str)`` then silently serialises them as their **repr string**
    (``"['a' 'b']"``) -- valid JSON, wrong type, and invisible to anyone who does not inspect
    the file. The same path turns a missing ``prev_chunk_id`` into ``NaN``, which strict JSON
    parsers reject outright. Every delivered ``.jsonl`` goes through here.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, np.generic):
        return jsonable(obj.item())
    if isinstance(obj, np.ndarray):
        return [jsonable(x) for x in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if obj is pd.NaT:
        return None
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return str(obj)


def write_jsonl(path: Path, records: list[dict]) -> int:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(jsonable(rec), ensure_ascii=False, allow_nan=False) + "\n")
    return len(records)



def df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(str(int(v)) if v.is_integer() else f"{v:g}")
            else:
                cells.append(str(v).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def validate(chunks: list[dict], para: pd.DataFrame) -> dict:
    ids = [c["chunk_id"] for c in chunks]
    dup = len(ids) - len(set(ids))
    empty = sum(1 for c in chunks if not c["text"].strip())
    over = sum(1 for c in chunks if c["n_tokens"] > c["chunk_size_limit"])
    multi_sec = sum(1 for c in chunks if c["n_sections"] > 1)

    # every paragraph must be fully covered, without gaps
    by_paper_spans: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for c in chunks:
        for pid, a, b in c["para_spans"]:
            by_paper_spans[c["paper_id"]][pid].append((a, b))
    gaps = 0
    content_gaps = 0
    checked = 0
    for r in para.itertuples():
        spans = sorted(by_paper_spans.get(r.paper_id, {}).get(r.para_id, []))
        if not spans:
            continue
        checked += 1
        cover = 0
        text = r.text_clean or ""
        for a, b in spans:
            if a > cover:
                gaps += 1
                # a single space/newline falling exactly on a chunk boundary is harmless,
                # losing real characters is not
                if text[cover:a].strip():
                    content_gaps += 1
            cover = max(cover, b)
        if cover < len(text):
            gaps += 1
            if text[cover:].strip():
                content_gaps += 1

    links_bad = 0
    for x, y in zip(chunks, chunks[1:]):
        if x["paper_id"] == y["paper_id"]:
            if x.get("next_chunk_id") != y["chunk_id"] or y.get("prev_chunk_id") != x["chunk_id"]:
                links_bad += 1

    return {
        "chunks": len(chunks),
        "duplicate_chunk_ids": dup,
        "empty_chunks": empty,
        "over_budget_chunks": over,
        "chunks_spanning_sections": multi_sec,
        "paragraphs_checked": checked,
        "paragraphs_with_boundary_gaps": gaps,
        "paragraphs_with_content_gaps": content_gaps,
        "broken_neighbour_links": links_bad,
    }


def gold_mapping(chunks: list[dict], qa: pd.DataFrame) -> tuple[list[dict], dict]:
    para_to_chunks: dict[str, list[str]] = defaultdict(list)
    chunk_by_id: dict[str, dict] = {}
    for c in chunks:
        chunk_by_id[c["chunk_id"]] = c
        for pid in c["para_ids"]:
            para_to_chunks[pid].append(c["chunk_id"])

    out: list[dict] = []
    n_with_gold = 0
    n_unreachable = 0
    co_split_hit = 0
    co_split_denom = 0
    for r in qa.to_dict("records"):
        g = r.get("gold_para_ids")
        gold_paras = list(g) if g is not None else []
        cids: list[str] = []
        seen: set[str] = set()
        for p in gold_paras:
            for cid in para_to_chunks.get(p, []):
                if cid not in seen:
                    seen.add(cid)
                    cids.append(cid)
        reachable = len(cids) > 0
        if gold_paras:
            n_with_gold += 1
            if not reachable:
                n_unreachable += 1

        # co-split: does a single chunk hold both an answer span and a gold paragraph?
        loc = r.get("extractive_spans_located")
        located = [s for s in (list(loc) if loc is not None else [])
                   if isinstance(s, dict) and s.get("para_id")]
        if located and gold_paras:
            co_split_denom += 1
            ok = False
            for s in located:
                for cid in para_to_chunks.get(s["para_id"], []):
                    c = chunk_by_id[cid]
                    if any(p in c["para_ids"] for p in gold_paras):
                        ok = True
                        break
                if ok:
                    break
            co_split_hit += int(ok)

        rec = dict(r)
        rec["gold_chunk_ids"] = cids
        rec["n_gold_chunks"] = len(cids)
        out.append(rec)

    stats = {
        "questions": len(out),
        "questions_with_gold_paras": n_with_gold,
        "questions_gold_unreachable": n_unreachable,
        "co_split_denom": co_split_denom,
        "co_split_hit": co_split_hit,
        "co_split_rate": round(co_split_hit / max(co_split_denom, 1) * 100, 2),
    }
    return out, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--out-dir", default="docs")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_FILE)
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--min-chunk-tokens", type=int, default=64)
    ap.add_argument("--section-boundary", default="soft", choices=["hard", "soft"])
    ap.add_argument("--no-abstract", action="store_true")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    cfg = ChunkConfig(
        chunk_size=args.chunk_size,
        overlap_ratio=args.overlap,
        min_chunk_tokens=args.min_chunk_tokens,
        section_boundary=args.section_boundary,
        include_abstract=not args.no_abstract,
    )
    name = args.name or f"size{cfg.chunk_size}_ov{cfg.resolved_overlap()}_{cfg.section_boundary}"
    proc = Path(args.processed_dir)
    out_dir = proc / "chunks" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    counter = TokenCounter(args.tokenizer)
    para = pd.read_parquet(proc / "paragraphs.parquet")
    qa = pd.read_parquet(proc / "qa.parquet")
    print(f"[ok] paragraphs={len(para)} qa={len(qa)} tokenizer={counter.backend}")

    by_paper: dict[str, list[dict]] = defaultdict(list)
    for r in para.to_dict("records"):
        by_paper[r["paper_id"]].append(r)

    all_chunks: list[dict] = []
    for paper_id, rows in by_paper.items():
        rows.sort(key=lambda r: (r["section_idx"], r["para_idx"]))
        split = rows[0]["split"]
        for c in chunk_paper(rows, cfg, counter, paper_id, split):
            d = dict(c.__dict__)
            d["chunk_size_limit"] = cfg.chunk_size
            all_chunks.append(d)
    all_chunks.sort(key=lambda d: (d["split"], d["paper_id"], d["chunk_idx_in_paper"]))
    print(f"[ok] built {len(all_chunks)} chunks")

    checks = validate(all_chunks, para)
    chunk_df = pd.DataFrame(all_chunks)
    out_cols = [c for c in chunk_df.columns if c != "chunk_size_limit"]
    chunk_df = chunk_df[out_cols].copy()
    # pyarrow cannot infer a mixed (str, int, int) tuple column; keep it as JSON text.
    chunk_df["para_spans"] = chunk_df["para_spans"].apply(
        lambda x: json.dumps([list(t) for t in x], ensure_ascii=False)
    )
    chunk_df.to_parquet(out_dir / "chunks.parquet", index=False)
    # parquet keeps ``para_spans`` as JSON text (pyarrow cannot infer a tuple column), but the
    # jsonl rendering must carry a real array -- otherwise the two formats disagree on type.
    jsonl_records = chunk_df.to_dict("records")
    for rec in jsonl_records:
        if isinstance(rec["para_spans"], str):
            rec["para_spans"] = json.loads(rec["para_spans"])
    n_jsonl = write_jsonl(out_dir / "chunks.jsonl", jsonl_records)
    print(f"[ok] wrote chunks.jsonl rows={n_jsonl}")

    qa_gold, gstats = gold_mapping(all_chunks, qa)
    write_jsonl(out_dir / "qa_with_gold.jsonl", qa_gold)

    n_tok = int(chunk_df["n_tokens"].sum())
    body_tok = int(para["n_tokens"].sum())
    stats = {
        "config_name": name,
        "config": cfg.to_dict(),
        "config_hash": cfg.hash(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tokenizer": counter.backend,
        "chunks": len(chunk_df),
        "papers": int(chunk_df["paper_id"].nunique()),
        "tokens_total": n_tok,
        "body_tokens": body_tok,
        "index_blowup": round(n_tok / max(body_tok, 1), 3),
        "avg_tokens": round(float(chunk_df["n_tokens"].mean()), 1),
        "median_tokens": int(chunk_df["n_tokens"].median()),
        "p95_tokens": int(chunk_df["n_tokens"].quantile(0.95)),
        "truncated_chunks": int(chunk_df["is_truncated"].sum()),
        "checks": checks,
        "gold": gstats,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    per_split = (
        chunk_df.groupby("split")
        .agg(chunks=("chunk_id", "size"), tokens=("n_tokens", "sum"),
             avg_tokens=("n_tokens", "mean"))
        .reset_index()
    )
    per_split["avg_tokens"] = per_split["avg_tokens"].round(1)

    L = [f"# Chunking 统计 · `{name}`\n"]
    L.append(f"> {stats['generated_at']}　|　tokenizer `{counter.backend}`")
    L.append(f"> 配置：size={cfg.chunk_size}，overlap={cfg.resolved_overlap()}，"
             f"min={cfg.min_chunk_tokens}，section={cfg.section_boundary}，"
             f"abstract={'含' if cfg.include_abstract else '不含'}\n")
    L.append("## 1. 规模\n")
    L.append(df_to_md(per_split))
    L.append("")
    L.append(f"- chunk 总数 **{stats['chunks']:,}**，覆盖 {stats['papers']} 篇论文")
    L.append(f"- token 总量 {stats['tokens_total']:,}（正文 {stats['body_tokens']:,}）"
             f"→ **索引放大 {stats['index_blowup']}×**")
    L.append(f"- 长度：mean {stats['avg_tokens']} / median {stats['median_tokens']} / p95 {stats['p95_tokens']}")
    L.append(f"- 硬切（`is_truncated`）chunk：{stats['truncated_chunks']}")
    L.append("")
    L.append("## 2. 不变量校验\n")
    L.append(df_to_md(pd.DataFrame([{"check": k, "value": v} for k, v in checks.items()])))
    L.append("")
    L.append("## 3. gold 映射（A7）\n")
    L.append(df_to_md(pd.DataFrame([{"metric": k, "value": v} for k, v in gstats.items()])))
    L.append("")
    L.append(f"> `co_split_rate` = 抽取式答案与它的证据段落落在**同一个 chunk** 的比例。"
             f"这个指标比平均 recall 更能解释坏 case：证据在、答案被切走 → 模型必错。")
    L.append("")
    (Path(args.out_dir) / f"chunking_stats_{name}.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[ok] wrote {out_dir}")
    print(f"[ok] checks={checks}")
    print(f"[ok] gold={gstats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
