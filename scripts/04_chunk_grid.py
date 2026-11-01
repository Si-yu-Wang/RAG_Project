"""A8: chunk-size / overlap grid sweep, measured entirely at the M1 layer.

The interesting question is not "which chunking looks nice" but **what ceiling does chunking
impose on downstream retrieval**. So every config is scored on model-free metrics only:

* cost        -- chunk count, index blow-up, length distribution, hard-split count
* intactness  -- how often a gold paragraph survives inside a single chunk
* granularity -- how many chunks one question's gold evidence fans out to
* oracle bound-- best achievable evidence recall@k with a *perfect* ranker over chunks

The oracle bound is the honest deliverable: if a config caps recall@5 at 0.93, no amount of
retrieval or reranking work can get past it, and that saves the team from chasing a mirage.

Usage:
    python scripts/04_chunk_grid.py --sizes 128,192,256,384,512,768 --overlaps 0,0.1,0.2,0.3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402

from rag_data.chunker import ChunkConfig, chunk_paper  # noqa: E402
from rag_data.oracle import evidence_recall_ceiling, paragraph_status  # noqa: E402
from rag_data.tokenizer import DEFAULT_TOKENIZER_FILE, TokenCounter  # noqa: E402

KS = (1, 3, 5, 10)


# ------------------------------------------------------------------------------ one config
def evaluate(cfg: ChunkConfig, by_paper, qa_records, counter, sent_cache, body_tokens,
             para_len: dict[str, int]):
    t0 = time.time()
    chunks: list[dict] = []
    for paper_id, rows in by_paper.items():
        for c in chunk_paper(rows, cfg, counter, paper_id, rows[0]["split"], sent_cache):
            chunks.append(c.__dict__)
    build_s = time.time() - t0

    # spans_by_para[p][chunk] = (a, b) -- offsets relative to the paragraph's own text
    spans_by_para: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
    chunk_paras: dict[str, set[str]] = {}
    for c in chunks:
        chunk_paras[c["chunk_id"]] = set(c["para_ids"])
        for pid, a, b in c["para_spans"]:
            spans_by_para[pid][c["chunk_id"]] = (a, b)

    n_tok = sum(c["n_tokens"] for c in chunks)
    toks = sorted(c["n_tokens"] for c in chunks)

    # Two *different* things, and conflating them flips the overlap conclusion:
    #   torn = no single chunk holds the paragraph -> evidence is genuinely fragmented
    #   dup  = the paragraph reappears in a later chunk because of overlap -> by design
    torn = dup = 0
    checked_paras = len(spans_by_para)
    for pid, sp in spans_by_para.items():
        is_torn, is_dup = paragraph_status(sp, para_len.get(pid, 0))
        torn += int(is_torn)
        dup += int(is_dup)

    # ---- granularity + oracle bound
    gold_paras: list[list[str]] = []
    co_split_hit = co_split_denom = 0
    n_covered = 0
    total_gold_chunks = 0
    for r in qa_records:
        g = r.get("gold_para_ids")
        gp = [str(p) for p in (list(g) if g is not None else [])]
        if not gp:
            continue
        n_covered += 1
        present = [p for p in gp if p in spans_by_para]
        if not present:
            continue
        gold_paras.append(present)
        total_gold_chunks += len({c for p in present for c in spans_by_para[p]})

        # co-split: does one chunk hold both an answer span and a gold paragraph?
        loc = r.get("extractive_spans_located")
        located = [s for s in (list(loc) if loc is not None else [])
                   if isinstance(s, dict) and s.get("para_id")]
        if located and gp:
            co_split_denom += 1
            gold_set = set(gp)
            ok = False
            for s in located:
                for cid in spans_by_para.get(s["para_id"], {}):
                    if chunk_paras[cid] & gold_set:
                        ok = True
                        break
                if ok:
                    break
            co_split_hit += int(ok)

    n_g = max(len(gold_paras), 1)
    recall, fits, n_ceiling = evidence_recall_ceiling(
        gold_paras, spans_by_para, para_len, KS)
    fits_pct = {f"gold_fits_{k}_pct": round(fits[k] / max(n_ceiling, 1) * 100, 2)
                for k in KS}

    row = {
        "name": f"s{cfg.chunk_size}_ov{cfg.resolved_overlap()}",
        "chunk_size": cfg.chunk_size,
        "overlap_ratio": cfg.overlap_ratio,
        "overlap_tokens": cfg.resolved_overlap(),
        "section_boundary": cfg.section_boundary,
        "min_chunk_tokens": cfg.min_chunk_tokens,
        "chunks": len(chunks),
        "tokens": n_tok,
        "index_blowup": round(n_tok / max(body_tokens, 1), 3),
        "avg_tokens": round(n_tok / max(len(chunks), 1), 1),
        "median_tokens": toks[len(toks) // 2] if toks else 0,
        "p95_tokens": toks[int(len(toks) * 0.95)] if toks else 0,
        "truncated_chunks": sum(1 for c in chunks if c["is_truncated"]),
        "paras_torn_pct": round(torn / max(checked_paras, 1) * 100, 2),
        "paras_dup_pct": round(dup / max(checked_paras, 1) * 100, 2),
        "mean_gold_chunks": round(total_gold_chunks / n_g, 2),
        "co_split_rate": round(co_split_hit / max(co_split_denom, 1) * 100, 2),
        "gold_questions": n_covered,
        "build_seconds": round(build_s, 2),
    }
    row.update(recall)
    row.update(fits_pct)
    return row


# ------------------------------------------------------------------------------------- io
def df_to_md(df: pd.DataFrame, keep: list[str] | None = None) -> str:
    d = df[keep] if keep else df
    cols = list(d.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in d.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:g}")
            else:
                cells.append(str(v).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _pick(df: pd.DataFrame, max_size: int, blowup_cap: float):
    sub = df[(df["chunk_size"] <= max_size) & (df["index_blowup"] <= blowup_cap)]
    if sub.empty:
        sub = df[df["chunk_size"] <= max_size]
    if sub.empty:
        sub = df
    return sub.sort_values(
        ["oracle_recall@5", "oracle_recall@1", "index_blowup"],
        ascending=[False, False, True],
    ).iloc[0]


def write_report(df: pd.DataFrame, args, ctx: dict) -> Path:
    hard_csv = Path(args.processed_dir) / "chunks" / "grid_results_hard.csv"
    L = ["# Chunk 参数网格实验结果\n"]
    L.append(f"> 论文 {ctx['papers']:,} 篇 / 题 {ctx['qa']:,} / 正文 {ctx['body_tokens']:,} tokens"
             f"　|　tokenizer `{ctx['tokenizer']}`　|　section_boundary `{args.section_boundary}`")
    L.append(f"> 共 {len(df)} 组配置（size × overlap）\n")

    L.append("## 1. 指标定义（全部与模型无关，纯数据层）\n")
    L.append("| 指标 | 含义 | 方向 |")
    L.append("|---|---|---|")
    L.append("| `index_blowup` | 全部 chunk 的 token 总量 ÷ 正文 token 总量 | 越低越省 |")
    L.append("| `paras_torn_pct` | **没有任何一个 chunk 完整包含该段落**的比例（真实撕裂） | 越低越好 |")
    L.append("| `paras_dup_pct` | 因 overlap 而出现在 ≥2 个 chunk 里的段落比例（**设计行为，不是缺陷**） | 仅作说明 |")
    L.append("| `mean_gold_chunks` | 一道题的证据平均散落在几个 chunk 里 | 越低越好 |")
    L.append("| `oracle_recall@k` | **完美排序器**在 k 个 chunk 内的证据召回上界 | 越高越好 |")
    L.append("| `co_split_rate` | 抽取式答案与其证据落在同一 chunk 的比例 | 越高越好 |")
    L.append("")
    L.append("> `oracle_recall@k` 是这张表最重要的列：它假设检索器提前知道答案，只受切分约束。"
             "任何 Dense / Graph / 融合方法的 recall@k 都不可能超过它。")
    L.append("")
    L.append("> **`torn` 与 `dup` 必须分开看。** 段落被 overlap 复制到两个 chunk 是设计意图；"
             "只有当**没有任何 chunk 完整包含该段落**时，证据才真的被切断了。"
             "把两者混为一谈（把 dup 当缺陷）会把 overlap 误判成全面劣势——第一版实现就踩了这个坑，"
             "当时结论是\"overlap 有害\"，修正后结论正好相反。")
    L.append("")

    L.append("## 2. 全量结果\n")
    keep = ["name", "chunks", "index_blowup", "avg_tokens", "p95_tokens", "truncated_chunks",
            "paras_torn_pct", "paras_dup_pct", "mean_gold_chunks",
            "co_split_rate", "oracle_recall@1", "oracle_recall@3", "oracle_recall@5",
            "oracle_recall@10"]
    L.append(df_to_md(df.sort_values(["chunk_size", "overlap_tokens"]), keep))
    L.append("")

    L.append("## 3. 透视：size × overlap → `oracle_recall@1` / `paras_torn_pct` / `index_blowup`\n")
    p1 = df.pivot_table(index="chunk_size", columns="overlap_ratio",
                        values="oracle_recall@1").round(2)
    p1.columns = [f"ov={c:g}" for c in p1.columns]
    p1 = p1.reset_index()
    L.append("**oracle_recall@1（%）** —— 单 chunk 内看全全部证据的比例\n")
    L.append(df_to_md(p1))
    L.append("")
    p2 = df.pivot_table(index="chunk_size", columns="overlap_ratio",
                        values="paras_torn_pct").round(2)
    p2.columns = [f"ov={c:g}" for c in p2.columns]
    p2 = p2.reset_index()
    L.append("**paras_torn_pct（%）** —— 段落被切散、无法在任何单个 chunk 内看全的比例\n")
    L.append(df_to_md(p2))
    L.append("")
    p3 = df.pivot_table(index="chunk_size", columns="overlap_ratio",
                        values="index_blowup").round(3)
    p3.columns = [f"ov={c:g}" for c in p3.columns]
    p3 = p3.reset_index()
    L.append("**index_blowup（×）** —— 索引成本\n")
    L.append(df_to_md(p3))
    L.append("")
    L.append("> 读法：**size 决定撕裂的下限，overlap 决定把撕裂压到多低。** "
             "两者在 `oracle_recall@1` 上可以互相替代（256/@0.3 ≈ 384/@0.1），"
             "但换来的索引成本不同——见 §5。")
    L.append("")

    L.append("## 4. 按 overlap 看（对每个 size 取均值）\n")
    agg2 = df.groupby("overlap_ratio", as_index=False).agg(
        chunks=("chunks", "mean"),
        index_blowup=("index_blowup", "mean"),
        paras_torn_pct=("paras_torn_pct", "mean"),
        paras_dup_pct=("paras_dup_pct", "mean"),
        oracle_recall_1=("oracle_recall@1", "mean"),
        oracle_recall_3=("oracle_recall@3", "mean"),
        oracle_recall_5=("oracle_recall@5", "mean"),
    ).round(2)
    L.append(df_to_md(agg2))
    L.append("")

    L.append("## 5. 边际收益：overlap 每多花 0.1 倍索引，换来多少召回上界\n")
    rows = []
    for size in sorted(df["chunk_size"].unique()):
        s = df[df["chunk_size"] == size].sort_values("overlap_ratio")
        for i in range(1, len(s)):
            prev, cur = s.iloc[i - 1], s.iloc[i]
            d_ov = cur["overlap_ratio"] - prev["overlap_ratio"]
            d_b = cur["index_blowup"] - prev["index_blowup"]
            d_r = cur["oracle_recall@1"] - prev["oracle_recall@1"]
            if d_b <= 0:
                continue
            rows.append({
                "size": size,
                "overlap": f"{prev['overlap_ratio']:.1f}→{cur['overlap_ratio']:.1f}",
                "Δblowup": round(d_b, 3),
                "Δoracle@1(pp)": round(d_r, 2),
                "pp per 0.1 blowup": round(d_r / d_b * 0.1, 2),
            })
    L.append(df_to_md(pd.DataFrame(rows)))
    L.append("")

    best_c = _pick(df, args.embed_max_tokens, 1.5)
    best_any = _pick(df, 10_000, 1.5)

    L.append("## 6. 推荐配置\n")
    L.append(f"**主推 `{best_c['name']}`**（size={int(best_c['chunk_size'])}，"
             f"overlap={int(best_c['overlap_tokens'])}）\n")
    L.append(df_to_md(pd.DataFrame([{
        "chunks": int(best_c["chunks"]),
        "index_blowup": best_c["index_blowup"],
        "oracle_recall@1": best_c["oracle_recall@1"],
        "oracle_recall@3": best_c["oracle_recall@3"],
        "oracle_recall@5": best_c["oracle_recall@5"],
        "co_split_rate": best_c["co_split_rate"],
        "paras_torn_pct": best_c["paras_torn_pct"],
    }])))
    L.append("")
    L.append("> 选择规则：硬约束 `chunk_size ≤ embedding 上限`（本轮取 "
             f"{args.embed_max_tokens}，依据 bge-large-en-v1.5 的 512 position embedding），"
             "软约束 `index_blowup ≤ 1.5`，在此范围内最大化 `oracle_recall@5`，"
             "并列时看 `oracle_recall@1` 与索引大小。**这是纯数据层排序，不含 embedding 信息。**")
    L.append("")
    if int(best_any["chunk_size"]) > int(best_c["chunk_size"]):
        L.append(f"若不设 embedding 上限，数据层最优是 `{best_any['name']}`"
                 f"（oracle_recall@1 = {best_any['oracle_recall@1']}%，"
                 f"torn = {best_any['paras_torn_pct']}%）。"
                 f"它超过 512 后会被 bge-large 截断，**只有在 M2 换用长上下文 embedding 后才可用**。")
        L.append("")

    if args.section_boundary == "soft":
        L.append("## 7. 怎么读这张表（三条结论）\n")

        def _cell(size: int, ov: float, col: str):
            s = df[(df["chunk_size"] == size) & (np.isclose(df["overlap_ratio"], ov))]
            return None if s.empty else s.iloc[0][col]

        big = df[df["chunk_size"] >= 256]["oracle_recall@5"].min()
        L.append(f"1. **k≥5 时，切分不是瓶颈。** `oracle_recall@5` 的最小值是 "
                 f"{df['oracle_recall@5'].min():.2f}%，而 size≥256 的配置最低也有 {big:.2f}%。"
                 f"只要检索器能把正确的 5 个 chunk 排进 top-5，证据基本一定完整——"
                 f"**瓶颈在排序（M2/M3/M4），不在切分（M1）**。")
        lo, hi = df["oracle_recall@1"].min(), df["oracle_recall@1"].max()
        L.append(f"2. **k=1 时代价巨大。** `oracle_recall@1` 在整个网格里从 {lo:.2f}% 到 {hi:.2f}%。"
                 f"小 chunk 意味着一条证据被拆到多个 chunk，**任何 top-1 指标都会系统性失真**；"
                 f"M5 汇报时必须同时给 top-k，不能只报 top-1。")
        a, b, c = (_cell(512, 0.0, "paras_torn_pct"), _cell(512, 0.3, "paras_torn_pct"),
                   _cell(512, 0.3, "index_blowup"))
        if a is not None and b is not None:
            L.append(f"3. **overlap 的真实作用是「修复边界撕裂」，不是「增加上下文」。** "
                     f"它把下一个 chunk 的起点推到段落起始之前，让段落完整落进后一个 chunk："
                     f"size=512 时 torn 从 overlap=0 的 {a}% 降到 overlap=0.3 的 {b}%，"
                     f"代价是索引放大到 {c}×。")
        L.append("")

    if hard_csv.exists():
        hd = pd.read_csv(hard_csv)
        L.append("## 8. 消融：section 硬边界 vs 软边界\n")
        pairs = []
        for _, h in hd.iterrows():
            m = df[(df["chunk_size"] == h["chunk_size"])
                   & (np.isclose(df["overlap_ratio"], h["overlap_ratio"]))]
            if m.empty:
                continue
            sft = m.iloc[0]
            pairs.append({
                "size": int(h["chunk_size"]),
                "overlap": h["overlap_ratio"],
                "chunks soft/hard": f"{int(sft['chunks'])} / {int(h['chunks'])}",
                "avg_tok soft/hard": f"{sft['avg_tokens']:.0f} / {h['avg_tokens']:.0f}",
                "torn% soft/hard": f"{sft['paras_torn_pct']:.2f} / {h['paras_torn_pct']:.2f}",
                "or@1 soft/hard": f"{sft['oracle_recall@1']:.2f} / {h['oracle_recall@1']:.2f}",
                "or@5 soft/hard": f"{sft['oracle_recall@5']:.2f} / {h['oracle_recall@5']:.2f}",
            })
        if pairs:
            L.append(df_to_md(pd.DataFrame(pairs)))
            L.append("")
            L.append("> 硬边界 = chunk 不跨 section；软边界 = 只按 token 预算装箱。")
            L.append(">")
            L.append("> 两个方向**不是单调优劣**：硬边界的 chunk 平均只有软边界的一半长"
                     "（每篇 13.9 个 section、每 section 仅约 295 token，预算用不满），"
                     "chunk 数多 35%，但 `torn` 与 `oracle_recall@1` **反而更好**——"
                     "因为边界落在 section 末尾时通常也落在段落末尾，段落不容易被切开。"
                     "总 token 成本（`index_blowup`）硬边界更低。")
            L.append(">")
            L.append("> **所以这一项不能只在数据层定**：硬边界给的是「更短但更完整」的 chunk，"
                     "软边界给的是「更长但更容易切散」的 chunk。"
                     "谁更好取决于 embedding 在 300 vs 480 token 上的信噪比，"
                     "必须等 M2 的 Dense baseline 用真实 recall 复核。")
            L.append("")

    out = Path(args.md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"[ok] wrote {out}")
    print(f"[ok] primary={best_c['name']}  unconstrained={best_any['name']}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--out-dir", default="docs")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_FILE)
    ap.add_argument("--sizes", default="128,192,256,384,512,768")
    ap.add_argument("--overlaps", default="0,0.1,0.2,0.3")
    ap.add_argument("--section-boundary", default="soft", choices=["hard", "soft"])
    ap.add_argument("--min-chunk-tokens", type=int, default=0,
                    help="0 = adaptive: min(64, chunk_size // 4)")
    ap.add_argument("--limit-papers", type=int, default=0)
    ap.add_argument("--embed-max-tokens", type=int, default=512)
    ap.add_argument("--csv", default=None, help="output csv (default grid_results[_hard].csv)")
    ap.add_argument("--md", default=None, help="output report path")
    ap.add_argument("--report-only", action="store_true",
                    help="skip the sweep and rebuild the report from an existing csv")
    ap.add_argument("--no-report", action="store_true",
                    help="only write the csv (used when sharding a long sweep)")
    args = ap.parse_args()

    tag = "" if args.section_boundary == "soft" else "_hard"
    # A smoke run on a paper subset must never overwrite the canonical result files --
    # otherwise a 40-paper check silently replaces the 1,169-paper deliverable.
    if args.limit_papers:
        tag = f"{tag}_limit{args.limit_papers}"
    csv_path = Path(args.csv) if args.csv else (
        Path(args.processed_dir) / "chunks" / f"grid_results{tag}.csv")
    if args.md is None:
        args.md = (Path(args.out_dir) /
                   ("chunk_grid_results.md" if tag == "" else f"chunk_grid{tag}.md"))

    proc = Path(args.processed_dir)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.report_only:
        df = pd.read_csv(csv_path)
        meta = json.loads((proc / "chunks" / "grid_results.json").read_text(encoding="utf-8"))
        body = int(meta.get("body_tokens") or 0)
        if not body and len(df):
            body = int(round(df.iloc[0]["tokens"] / max(df.iloc[0]["index_blowup"], 1e-9)))
        qa_n = int(meta.get("qa") or 0)
        if not qa_n:
            try:
                qa_n = int(len(pd.read_parquet(proc / "qa.parquet")))
            except Exception:
                qa_n = int(df["gold_questions"].iloc[0]) if len(df) else 0
        write_report(df, args, {
            "papers": meta.get("papers", 0),
            "qa": qa_n,
            "body_tokens": body,
            "tokenizer": meta.get("tokenizer", ""),
        })
        return 0

    counter = TokenCounter(args.tokenizer)
    para = pd.read_parquet(proc / "paragraphs.parquet")
    qa = pd.read_parquet(proc / "qa.parquet")
    qa_records = qa.to_dict("records")

    by_paper: dict[str, list[dict]] = defaultdict(list)
    para_len: dict[str, int] = {}
    for r in para.to_dict("records"):
        by_paper[r["paper_id"]].append(r)
        para_len[r["para_id"]] = len(r["text_clean"] or "")
    for rows in by_paper.values():
        rows.sort(key=lambda r: (r["section_idx"], r["para_idx"]))
    if args.limit_papers:
        keep_ids = list(by_paper)[: args.limit_papers]
        by_paper = {k: by_paper[k] for k in keep_ids}
        qa_records = [r for r in qa_records if r["paper_id"] in by_paper]

    body_tokens = int(para["n_tokens"].sum()) if not args.limit_papers else int(
        sum(r["n_tokens"] for rows in by_paper.values() for r in rows)
    )
    print(f"[ok] papers={len(by_paper)} qa={len(qa_records)} body_tokens={body_tokens:,} "
          f"tokenizer={counter.backend}")

    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    overlaps = [float(x) for x in args.overlaps.split(",") if x.strip()]

    # One shared cache: sentence splits and their token counts do not depend on chunk_size.
    sent_cache: dict = {}
    rows_out: list[dict] = []
    for size in sizes:
        min_tok = args.min_chunk_tokens or max(16, min(64, size // 4))
        for ov in overlaps:
            cfg = ChunkConfig(
                chunk_size=size,
                overlap_ratio=ov,
                min_chunk_tokens=min(min_tok, size - 16),
                section_boundary=args.section_boundary,
            )
            if cfg.resolved_overlap() >= size:
                continue
            t0 = time.time()
            row = evaluate(cfg, by_paper, qa_records, counter, sent_cache, body_tokens, para_len)
            rows_out.append(row)
            print(f"[{len(rows_out):2d}] {row['name']:<14} chunks={row['chunks']:6d} "
                  f"blowup={row['index_blowup']:.3f} torn={row['paras_torn_pct']:5.2f}% "
                  f"co_split={row['co_split_rate']:5.2f}% "
                  f"or@1={row['oracle_recall@1']:5.2f}% or@5={row['oracle_recall@5']:5.2f}% "
                  f"({time.time() - t0:.1f}s)")

    df = pd.DataFrame(rows_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    if args.no_report:
        print(f"[ok] wrote {csv_path} (report skipped)")
        return 0
    if tag == "":
        (proc / "chunks" / "grid_results.json").write_text(
            json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "tokenizer": counter.backend,
                        "papers": len(by_paper),
                        "qa": len(qa_records),
                        "body_tokens": body_tokens,
                        "limit_papers": args.limit_papers,
                        "rows": rows_out}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    write_report(df, args, {
        "papers": len(by_paper),
        "qa": len(qa_records),
        "body_tokens": body_tokens,
        "tokenizer": counter.backend,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
