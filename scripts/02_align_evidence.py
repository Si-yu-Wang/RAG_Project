"""A4 / script 02: align gold evidence onto paragraph ids -> qa.parquet + qa_dataset.jsonl.

Usage:
    python scripts/02_align_evidence.py --data-dir data/raw/qasper \
        --processed-dir data/processed --out-dir docs
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from rag_data.evidence_aligner import AlignConfig, align_paper  # noqa: E402
from rag_data.loader import load_split  # noqa: E402

SPLIT_FILES = {"train": "qasper-train-v0.3.json", "dev": "qasper-dev-v0.3.json"}


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw/qasper")
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--out-dir", default="docs/dataset")
    ap.add_argument("--no-fuzzy", action="store_true")
    ap.add_argument("--fuzzy-threshold", type=int, default=88)
    args = ap.parse_args()

    data_dir, proc, out_dir = Path(args.data_dir), Path(args.processed_dir), Path(args.out_dir)
    cfg = AlignConfig(use_fuzzy=not args.no_fuzzy, fuzzy_threshold=args.fuzzy_threshold)

    para = pd.read_parquet(proc / "paragraphs.parquet")
    by_paper: dict[str, list[dict]] = defaultdict(list)
    for r in para.to_dict("records"):
        by_paper[r["paper_id"]].append(r)
    print(f"[ok] paragraphs loaded: {len(para)} rows / {len(by_paper)} papers")

    all_records: list[dict] = []
    levels_total: Counter[str] = Counter()
    missing_papers: list[str] = []
    for split, fname in SPLIT_FILES.items():
        path = data_dir / fname
        if not path.exists():
            print(f"[warn] missing {path}")
            continue
        for paper in load_split(str(path), split):
            rows = by_paper.get(paper.paper_id)
            if not rows:
                missing_papers.append(paper.paper_id)
                continue
            recs, levels = align_paper(paper.qas, rows, split, cfg)
            for r in recs:
                r["paper_id"] = paper.paper_id
            all_records.extend(recs)
            levels_total.update(levels)
        print(f"[ok] {split}: aligned")

    df = pd.DataFrame(all_records)
    df = df.sort_values(["split", "paper_id", "question_id"]).reset_index(drop=True)
    df.to_parquet(proc / "qa.parquet", index=False)
    # jsonl for humans / Member 5; identical rows, list columns preserved
    with open(proc / "qa_dataset.jsonl", "w", encoding="utf-8") as f:
        for rec in df.to_dict("records"):
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    print(f"[ok] wrote qa.parquet + qa_dataset.jsonl rows={len(df)}")

    textual = sum(v for k, v in levels_total.items() if k not in {"float", "empty"})
    mapped = textual - levels_total.get("fail", 0)
    align_rate = mapped / max(textual, 1) * 100

    per_split = []
    for split, g in df.groupby("split"):
        t = int(g["n_evidence"].sum())
        fl = int(g["n_evidence_float"].sum())
        mp = int(g["n_evidence_mapped"].sum())
        tx = t - fl
        per_split.append({
            "split": split,
            "questions": len(g),
            "answerable": int(g["is_answerable"].sum()),
            "unanswerable_only": int((~g["is_answerable"]).sum()),
            "textual_evidence": tx,
            "mapped": mp,
            "align_rate_pct": round(mp / max(tx, 1) * 100, 2),
            "questions_with_gold": int((g["n_gold_paras"] > 0).sum()),
            "questions_zero_gold": int((g["n_gold_paras"] == 0).sum()),
            "float_only_questions": int(g["is_float_only"].sum()),
        })

    span_total = sum(len(x) for x in df["extractive_spans"])
    span_located = sum(
        1 for x in df["extractive_spans_located"] for s in x if s.get("para_id")
    )
    zero_gold_answerable = int(((df["n_gold_paras"] == 0) & df["is_answerable"]).sum())

    manifest = {
        "artifact": "qa.parquet",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "align_config": cfg.to_dict(),
        "stats": {
            "questions": len(df),
            "evidence_items": int(df["n_evidence"].sum()),
            "levels": dict(levels_total),
            "align_rate_pct": round(align_rate, 2),
            "papers_in_index": len(by_paper),
            "papers_missing_from_index": len(missing_papers),
        },
    }
    (proc / "manifest_qa.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    L: list[str] = []
    L.append("# A4 证据对齐报告\n")
    L.append(f"> 生成时间：{manifest['generated_at']}")
    L.append(f"> 配置：`use_fuzzy={cfg.use_fuzzy}`，阈值 {cfg.fuzzy_threshold}，section 规则 {cfg.section_rule}\n")

    L.append("## 1. 逐 split 结果\n")
    L.append(df_to_md(pd.DataFrame(per_split)))
    L.append("")

    L.append("## 2. 证据条目的对齐路径分布\n")
    rows = []
    total = sum(levels_total.values())
    for k, v in levels_total.most_common():
        rows.append({"evidence_level": k, "count": v, "pct": round(v / max(total, 1) * 100, 2)})
    L.append(df_to_md(pd.DataFrame(rows)))
    L.append("")
    L.append(f"- **文本证据对齐率 {align_rate:.2f}%**（分母 = 全部 evidence 条目 − float − 空）")
    L.append(f"- 仍有 **{levels_total.get('fail', 0)}** 条无法对齐")
    L.append("")

    L.append("## 3. 关键校验\n")
    L.append(df_to_md(pd.DataFrame([
        {"check": "可回答但 gold 为空的题目数", "value": zero_gold_answerable},
        {"check": "只有图表证据（float_only）的题目数", "value": int(df["is_float_only"].sum())},
        {"check": "抽取式答案片段总数", "value": span_total},
        {"check": "其中能定位到段落", "value": f"{span_located} "
                                            f"({span_located / max(span_total, 1) * 100:.1f}%)"},
        {"check": "语料索引缺失的论文数", "value": len(missing_papers)},
    ])))
    L.append("")
    if zero_gold_answerable:
        L.append(f"> ⚠️ 有 **{zero_gold_answerable}** 条可回答问题的 gold 为空，M5 计算 recall 时应把它们单独列出，"
                 "不能直接当 0 分（否则会系统性拉低指标）。字段 `zero_gold_reason` 给出了原因，分布如下：\n")
        zr = (df[df["n_gold_paras"] == 0]
              .groupby(["zero_gold_reason", "is_answerable"]).size()
              .unstack(fill_value=0).reset_index())
        zr.columns = ["zero_gold_reason"] + [
            "answerable" if c is True else "unanswerable_only" for c in zr.columns[1:]
        ]
        for c in ("answerable", "unanswerable_only"):
            if c not in zr.columns:
                zr[c] = 0
        L.append(df_to_md(zr[["zero_gold_reason", "answerable", "unanswerable_only"]]))
        L.append("")
        L.append(f"- 因此**可用于计算 recall 的题目数 = {int((df['n_gold_paras'] > 0).sum())}** "
                 f"（3,598 − {int((df['n_gold_paras'] == 0).sum())} 条 gold 为空）")
        L.append("")
        L.append("| reason | 含义 | 建议处理 |")
        L.append("|---|---|---|")
        L.append("| `no_evidence` | 标注者没选任何证据段（绝大多数是 yes/no 题，如"
                 "「他们用了预训练模型吗？」）。**注意含大量 unanswerable 题** | **从 recall 分母剔除**，只在答案级指标里出现 |")
        L.append("| `float_only` | 证据是图/表（`FLOAT SELECTED`），正文里没有对应文本 | **从 recall 分母剔除** |")
        L.append("| `align_fail` | 文本证据但确实对不上（仅 "
                 f"{int(df['zero_gold_reason'].eq('align_fail').sum())} 题） | 人工抽检后可接受 |")
        L.append("")

    L.append("## 4. section 规则的作用\n")
    sec_hits = levels_total.get("section", 0)
    L.append(f"- 靠 `section_name` 命中的证据条目：**{sec_hits}** 条")
    L.append("- QASPER 的 `section_name` 本身是 S2ORC 层级路径（`\"Experiments ::: Ablation\"`），"
             f"全语料 **15,155 / 61,317（24.7%）** 的段落带 `:::` 层级；"
             "这条规则把它们解析成「该小节的**全部段落**」，同时给 M3 免费提供了层级边。")
    L.append("")

    L.append("## 5. 字段说明（给 M5）\n")
    L.append("| 字段 | 说明 |")
    L.append("|---|---|")
    for k, v in [
        ("gold_para_ids", "该题所有标注者证据的段落并集（**recall 的唯一依据**）"),
        ("evidence_levels", "本题证据走过的对齐路径，逗号分隔"),
        ("is_answerable", "任一标注者给出非 unanswerable 答案"),
        ("is_float_only", "证据只有图表，正文里不存在 → 应从 recall 分母剔除"),
        ("extractive_spans_located", "抽取式答案片段 + 所在 para_id + 段内偏移（用于算「答案与证据是否同 chunk」）"),
        ("answer_types", "本题各标注者的答案类型集合（7.8% 的题不一致，聚合口径需 M5 定）"),
    ]:
        L.append(f"| `{k}` | {v} |")
    L.append("")

    (out_dir / "evidence_alignment_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[ok] wrote {out_dir / 'evidence_alignment_report.md'}")
    print(f"[ok] align_rate={align_rate:.2f}%  zero_gold_answerable={zero_gold_answerable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
