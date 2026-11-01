"""P2+P3 / script 01: clean the corpus and flatten it into paragraphs.parquet.

Usage:
    python scripts/01_prepare_corpus.py --data-dir data/raw/qasper \
        --processed-dir data/processed --out-dir docs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from rag_data.cleaner import Cleaner, CleanConfig, decode_segments, nfkc_is_charwise  # noqa: E402
from rag_data.flattener import SEP, flatten_paper  # noqa: E402
from rag_data.loader import load_split  # noqa: E402
from rag_data.tokenizer import DEFAULT_TOKENIZER_FILE, TokenCounter  # noqa: E402

SPLIT_FILES = {
    "train": "qasper-train-v0.3.json",
    "dev": "qasper-dev-v0.3.json",
}


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


def validate(rows: list[dict], flat_by_paper: dict[str, str], n_sample: int = 200) -> dict:
    """The lossless contract: slice(flat, start, end) must equal text_clean."""
    ids = [r["para_id"] for r in rows]
    dup = [k for k, v in Counter(ids).items() if v > 1]

    bad_slice = 0
    for r in rows[:n_sample]:
        flat = flat_by_paper[r["paper_id"]]
        if flat[r["char_start"] : r["char_end"]] != r["text_clean"]:
            bad_slice += 1

    # offset map must be monotonic and cover the cleaned text exactly once
    bad_map = 0
    for r in rows[:2000]:
        segs = decode_segments(r["offset_map"], len(r["text_clean"]))
        cover = 0
        ok = True
        for s in segs:
            if s.clean_start != cover or s.clean_end < s.clean_start:
                ok = False
                break
            cover = s.clean_end
        if not ok or cover != len(r["text_clean"]):
            bad_map += 1

    return {
        "duplicate_para_ids": len(dup),
        "slice_mismatch": bad_slice,
        "slice_checked": min(n_sample, len(rows)),
        "offset_map_mismatch": bad_map,
        "offset_map_checked": min(2000, len(rows)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw/qasper")
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--out-dir", default="docs")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_FILE)
    ap.add_argument("--ref-mode", default="placeholder", choices=["keep", "placeholder", "drop"])
    ap.add_argument("--no-abstract", action="store_true", help="exclude the abstract section")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    proc_dir = Path(args.processed_dir)
    proc_dir.mkdir(parents=True, exist_ok=True)

    counter = TokenCounter(args.tokenizer)
    cfg = CleanConfig(ref_mode=args.ref_mode)
    cleaner = Cleaner(cfg)
    print(f"[ok] tokenizer={counter.backend} ref_mode={cfg.ref_mode}")

    papers = []
    for split, fname in SPLIT_FILES.items():
        p = data_dir / fname
        if p.exists():
            sub = load_split(str(p), split)
            papers.extend(sub)
            print(f"[ok] {split}: {len(sub)} papers")
    if not papers:
        print("[error] no data")
        return 1

    all_rows: list[dict] = []
    flat_by_paper: dict[str, str] = {}
    agg = Counter()
    per_split: dict[str, Counter] = {}
    empty_by_split: Counter = Counter()
    examples: list[tuple[int, dict]] = []
    nfkc_violations = 0
    nfkc_checked = 0

    for paper in papers:
        rows, st, flat = flatten_paper(
            paper, cleaner, counter=None, include_abstract=not args.no_abstract
        )
        flat_by_paper[paper.paper_id] = flat
        all_rows.extend(rows)
        ps = per_split.setdefault(paper.split, Counter())
        ps["papers"] += 1
        ps["paragraphs"] += st.paragraphs
        ps["empty_skipped"] += st.empty_skipped
        ps["chars_raw"] += st.chars_raw
        ps["chars_clean"] += st.chars_clean
        ps["changed"] += st.changed
        for k, v in (st.rule_hits or {}).items():
            agg[k] += v
        empty_by_split[paper.split] += st.empty_skipped
        for r in rows:
            if r["offset_map"]:
                n_edits = r["offset_map"].count(";") + 1
                if len(examples) < 3 or n_edits > examples[-1][0]:
                    examples.append((n_edits, r))
                    examples.sort(key=lambda x: -x[0])
                    examples = examples[:3]

    # batched token counts (one encode call per 256 paragraphs)
    texts = [r["text_clean"] for r in all_rows]
    counts = counter.count_batch(texts)
    for r, c in zip(all_rows, counts):
        r["n_tokens"] = c

    # nfkc per-char assumption, checked on real data
    for r in all_rows[:3000]:
        nfkc_checked += 1
        if not nfkc_is_charwise(r["text_raw"]):
            nfkc_violations += 1

    df = pd.DataFrame(all_rows)
    out_parquet = proc_dir / "paragraphs.parquet"
    df.to_parquet(out_parquet, index=False)
    print(f"[ok] wrote {out_parquet} rows={len(df)}")

    # LaTeX-heavy paragraphs: S2ORC keeps display equations verbatim. They are pure noise
    # for dense retrieval but we do not drop them silently -- we measure and flag them.
    dollar_counts = df["text_clean"].str.count(r"\$")
    math_mask = dollar_counts >= 4
    math_stats = {
        "paragraphs": int(math_mask.sum()),
        "pct_of_paragraphs": round(float(math_mask.mean()) * 100, 2),
        "tokens": int(df.loc[math_mask, "n_tokens"].sum()),
        "pct_of_tokens": round(
            float(df.loc[math_mask, "n_tokens"].sum()) / max(int(df["n_tokens"].sum()), 1) * 100, 2
        ),
    }

    checks = validate(all_rows, flat_by_paper)

    manifest = {
        "artifact": "paragraphs.parquet",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "clean_config": cfg.to_dict(),
        "include_abstract": not args.no_abstract,
        "tokenizer": counter.backend,
        "separator": SEP,
        "split_files": {k: str(data_dir / v) for k, v in SPLIT_FILES.items() if (data_dir / v).exists()},
        "stats": {
            "papers": len(papers),
            "paragraphs": len(all_rows),
            "tokens": int(df["n_tokens"].sum()),
            "chars_clean": int(df["n_chars"].sum()),
            "rule_hits": dict(agg),
            "latex_heavy": math_stats,
        },
        "checks": checks,
    }
    (proc_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------------- report ----------------
    tot_raw = sum(ps["chars_raw"] for ps in per_split.values())
    tot_clean = sum(ps["chars_clean"] for ps in per_split.values())
    L: list[str] = []
    L.append("# P2/P3 清洗与展平报告\n")
    L.append(f"> 生成时间：{manifest['generated_at']}　|　tokenizer：`{counter.backend}`")
    L.append(f"> 配置：`ref_mode={cfg.ref_mode}`，NFKC={cfg.nfkc}，去零宽={cfg.strip_zero_width}，"
             f"连字符合并={cfg.join_hyphenated}，换行合并={cfg.join_newlines}，"
             f"空白压缩={cfg.collapse_spaces}，抽象段={'排除' if args.no_abstract else '包含'}\n")

    L.append("## 1. 规模\n")
    rows_md = []
    for split, ps in sorted(per_split.items()):
        rows_md.append({
            "split": split,
            "papers": ps["papers"],
            "paragraphs": ps["paragraphs"],
            "empty_skipped": ps["empty_skipped"],
            "chars_raw": ps["chars_raw"],
            "chars_clean": ps["chars_clean"],
            "changed_paras": ps["changed"],
        })
    L.append(df_to_md(pd.DataFrame(rows_md)))
    L.append("")
    L.append(f"- 段落总数 **{len(all_rows):,}**，token 总数 **{int(df['n_tokens'].sum()):,}**")
    L.append(f"- 字符数 {tot_raw:,} → {tot_clean:,}（**-{(1 - tot_clean / max(tot_raw, 1)) * 100:.2f}%**）")
    L.append(f"- 被清洗改动过的段落：**{sum(ps['changed'] for ps in per_split.values()):,}** "
             f"（{sum(ps['changed'] for ps in per_split.values()) / max(len(all_rows), 1) * 100:.2f}%）")
    L.append("")

    L.append("## 2. 各条规则的命中次数\n")
    L.append(df_to_md(pd.DataFrame([{"rule": k, "hits": v} for k, v in agg.most_common()])))
    L.append("")

    L.append("## 3. 校验（无损契约）\n")
    L.append(df_to_md(pd.DataFrame([{"check": k, "value": v} for k, v in checks.items()])))
    L.append("")
    L.append(f"- NFKC 逐字符等价性抽查 {nfkc_checked:,} 段，违反 {nfkc_violations} 段")
    L.append("")

    L.append("## 4. LaTeX 密集段落（待决策）\n")
    L.append(df_to_md(pd.DataFrame([{"metric": k, "value": v} for k, v in math_stats.items()])))
    L.append("")
    L.append("> 判据：段落内 `$` 出现 ≥ 4 次。这些是 S2ORC 原样保留的行间公式，"
             "对 BGE 是纯噪声（占据 token 预算但不携带可检索语义），对 BM25 也只会贡献 `nabla`、`Theta` 这类符号串。"
             "**是否在切分前剔除，需要和 M2/M5 一起拍板**：剔除会损失少量公式型问题的可检索性，不剔则会稀释 chunk 的信噪比。")
    L.append("")

    L.append("## 5. 清洗样例（改动最多的 3 段）\n")
    for n_edits, r in examples:
        L.append(f"**{r['para_id']}**（{n_edits} 个编辑段，{r['n_tokens']} tokens）\n")
        L.append("原文：\n")
        L.append("> " + r["text_raw"][:400].replace("\n", " ") + "\n")
        L.append("清洗后：\n")
        L.append("> " + r["text_clean"][:400] + "\n")
    (out_dir / "cleaning_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[ok] wrote {out_dir / 'cleaning_report.md'}")
    print(f"[ok] checks: {checks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
