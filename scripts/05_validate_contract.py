"""A9 / script 05: validate a delivered chunk config against the published contract.

This is the gate every downstream member runs before trusting the data, and the gate M1 runs
before tagging a release. It answers three questions that a schema document cannot:

    is the table structurally sound?      (keys, ids, chains, offsets, token budget)
    is the evidence reconstructable?      (coverage, gold reachability)
    is anything leaked into public data?  (answer-derived columns)

Usage:
    python scripts/05_validate_contract.py --chunks-dir data/processed/chunks/size512_ov102_soft
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from rag_data import validate as V  # noqa: E402

ICON = {V.PASS: "✅", V.WARN: "⚠️", V.FAIL: "❌"}


def load_chunks(chunks_dir: Path) -> tuple[list[dict], dict]:
    df = pd.read_parquet(chunks_dir / "chunks.parquet")
    rows = df.to_dict("records")
    for r in rows:
        raw = r["para_spans"]
        r["para_spans"] = json.loads(raw) if isinstance(raw, str) else raw
        r["para_spans"] = [tuple(x) for x in r["para_spans"]]
        r["para_ids"] = [str(p) for p in (list(r["para_ids"]) if r["para_ids"] is not None else [])]
    manifest = {}
    mpath = chunks_dir / "manifest.json"
    if mpath.exists():
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    return rows, manifest


def load_qa(path: Path) -> list[dict]:
    df = pd.read_parquet(path)
    rows = df.to_dict("records")
    for r in rows:
        for k in ("gold_para_ids", "free_form_answers", "extractive_spans",
                  "extractive_spans_located"):
            if k in r:
                v = r[k]
                r[k] = [x for x in v] if v is not None else []
    return rows


def load_gold_chunks(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rec["gold_chunk_ids"] = list(rec.get("gold_chunk_ids") or [])
            rec["gold_para_ids"] = list(rec.get("gold_para_ids") or [])
            out.append(rec)
    return out


def md_table(findings: list[V.Finding]) -> str:
    lines = ["| 状态 | 检查项 | 结果 |", "|---|---|---|"]
    for f in findings:
        lines.append(f"| {ICON[f.status]} {f.status} | `{f.check}` | {f.detail} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--chunks-dir", default=None,
                    help="default: data/processed/chunks/size512_ov102_soft")
    ap.add_argument("--out", default="docs/contract/contract_validation.md")
    args = ap.parse_args()

    proc = Path(args.processed_dir)
    cdir = Path(args.chunks_dir) if args.chunks_dir else proc / "chunks" / "size512_ov102_soft"

    chunks, manifest = load_chunks(cdir)
    para = pd.read_parquet(proc / "paragraphs.parquet")
    para_len = {r["para_id"]: len(r["text_clean"] or "") for r in para.to_dict("records")}
    para_text = {r["para_id"]: (r["text_clean"] or "") for r in para.to_dict("records")}
    qa = load_qa(proc / "qa.parquet")
    gold = load_gold_chunks(cdir / "qa_with_gold.jsonl")

    limit = (manifest.get("config") or {}).get("chunk_size")
    findings: list[V.Finding] = []
    findings.append(V.finding("artifact.chunks", V.PASS if chunks else V.FAIL,
                             f"{len(chunks):,} chunks from {cdir.name}"))
    findings += V.check_columns(chunks, V.CHUNK_REQUIRED, V.FORBIDDEN_IN_PUBLIC)
    findings += V.check_chunk_ids(chunks)
    findings += V.check_ordering(chunks)
    findings += V.check_neighbours(chunks)
    findings += V.check_token_budget(chunks, limit)
    findings += V.check_para_spans(chunks, para_len)
    findings += V.check_coverage(chunks, para_len, para_text)
    findings += V.check_text_sha1(chunks)
    findings += V.check_config_hash(chunks, manifest.get("config_hash"))

    chunk_ids = {r["chunk_id"] for r in chunks}
    para_ids = set(para_len)
    qa_findings: list[V.Finding] = []
    n_with_gold = sum(1 for g in gold if g.get("gold_chunk_ids"))
    qa_findings.append(V.finding(
        "artifact.qa", V.PASS if qa else V.FAIL,
        f"{len(qa):,} questions, {n_with_gold:,} with non-empty gold_chunk_ids "
        f"(= the only valid recall denominator)"))
    qa_findings += V.check_columns(qa, V.QA_REQUIRED)
    qa_findings += V.check_gold(gold or qa, chunk_ids, para_ids)

    all_findings = findings + qa_findings
    n_pass, n_warn, n_fail = V.summarise(all_findings)

    L = [f"# 契约校验 · `{cdir.name}`\n"]
    L.append(f"> {len(chunks):,} chunks / {len(qa):,} 题　|　"
             f"结果：**{n_pass} PASS，{n_warn} WARN，{n_fail} FAIL**\n")
    if n_fail:
        L.append("> ❌ **本配置未通过契约校验，不可交付给下游。**\n")
    else:
        L.append("> ✅ 全部结构性与防泄漏检查通过。\n")

    L.append("## 1. chunks 表\n")
    L.append(md_table(findings))
    L.append("")
    L.append("## 2. 评测表（M5 专用）\n")
    L.append(md_table(qa_findings))
    L.append("")

    L.append("## 3. 配置\n")
    if manifest:
        cfg = manifest.get("config", {})
        L.append(df_to_md(pd.DataFrame([{
            "chunk_size": cfg.get("chunk_size"),
            "overlap_ratio": cfg.get("overlap_ratio"),
            "overlap_tokens": cfg.get("resolved_overlap_tokens"),
            "min_chunk_tokens": cfg.get("min_chunk_tokens"),
            "section_boundary": cfg.get("section_boundary"),
            "include_abstract": cfg.get("include_abstract"),
            "chunker_version": cfg.get("chunker_version"),
            "config_hash": manifest.get("config_hash"),
            "tokenizer": manifest.get("tokenizer"),
        }])))
        L.append("")
        g = manifest.get("gold", {})
        if g:
            L.append(f"- `co_split_rate` = {g.get('co_split_rate')}% "
                     f"（{g.get('co_split_hit')}/{g.get('co_split_denom')}）")
            L.append(f"- 有 gold 段落的题目 {g.get('questions_with_gold_paras')} 题，"
                     f"不可达 {g.get('questions_gold_unreachable')} 题")
            L.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"[ok] {cdir.name}: PASS={n_pass} WARN={n_warn} FAIL={n_fail} -> {out}")
    return 1 if n_fail else 0


def df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
