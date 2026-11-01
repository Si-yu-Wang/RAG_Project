"""P1 / script 00: profile the QASPER corpus and emit docs/dataset/dataset_profile.md.

Usage:
    python scripts/00_profile_dataset.py --data-dir data/raw/qasper --out-dir docs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Keep matplotlib's font cache inside the project so we never touch ~/.matplotlib
# (its cache refresh deletes stale files, which trips the sandbox delete guard).
_MPL = Path(__file__).resolve().parents[1] / "data" / "cache" / "mpl"
_MPL.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import pandas as pd  # noqa: E402

from rag_data.loader import load_split  # noqa: E402
from rag_data.profiling import build_profile  # noqa: E402
from rag_data.tokenizer import DEFAULT_TOKENIZER_FILE, TokenCounter  # noqa: E402

SPLIT_FILES = {
    "train": "qasper-train-v0.3.json",
    "dev": "qasper-dev-v0.3.json",
}


def df_to_md(df: pd.DataFrame, index: bool = False) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:g}")
            else:
                cells.append(str(v).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _plt():
    """Lazy matplotlib import: figures are optional and must never block the report."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_hist(values, path: Path, title: str, xlabel: str, bins: int = 60) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=140)
    ax.hist(values, bins=bins, color="#3b7dd8", edgecolor="white", linewidth=0.4)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("papers", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_bar(labels, values, path: Path, title: str, xlabel: str = "", rotate: int = 0) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=140)
    ax.bar([str(x) for x in labels], values, color="#d8793b", edgecolor="white", linewidth=0.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("count", fontsize=9)
    ax.tick_params(labelsize=8)
    if rotate:
        plt.setp(ax.get_xticklabels(), rotation=rotate, ha="right")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw/qasper")
    ap.add_argument("--out-dir", default="docs/dataset")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_FILE,
                    help="path to tokenizer.json, or a HF model name")
    ap.add_argument("--no-figures", action="store_true", help="skip PNG generation")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    papers = []
    per_split_counts = {}
    for split, fname in SPLIT_FILES.items():
        p = data_dir / fname
        if not p.exists():
            print(f"[warn] missing {p}, skipped")
            continue
        sub = load_split(str(p), split)
        per_split_counts[split] = len(sub)
        papers.extend(sub)
        print(f"[ok] {split}: {len(sub)} papers")

    if not papers:
        print("[error] no data loaded")
        return 1

    counter = TokenCounter(args.tokenizer)
    print(f"[ok] tokenizer backend = {counter.backend}")
    stats, tables = build_profile(papers, counter)
    stats["per_split_files"] = per_split_counts
    stats["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ---- figures (optional; must never block the report) -----------------
    FIG = {"ok": not args.no_figures}
    if FIG["ok"]:
        try:
            body_tokens = [counter.count(p.body_text()) for p in papers]
            save_hist(body_tokens, fig_dir / "p1_paper_body_tokens.png",
                      "Body tokens per paper (BGE tokenizer)", "tokens")

            at = tables["answer_type"]
            save_bar(at["answer_type"], at["count"], fig_dir / "p1_answer_type.png",
                     "Answer annotation types", "type")

            per_q = []
            for p in papers:
                for qa_ in p.qas:
                    per_q.append(min(len(qa_.get("answers") or []), 8))
            vc = pd.Series(per_q).value_counts().sort_index()
            save_bar(vc.index, vc.to_numpy(), fig_dir / "p1_answers_per_question.png",
                     "Answers per question (>=8 bucketed to 8)", "answers")

            eb = tables["evidence_buckets"]
            save_bar(eb["bucket"], eb["count"], fig_dir / "p1_evidence_buckets.png",
                     "Evidence item resolution against paragraphs", "bucket", rotate=20)
        except Exception as exc:  # pragma: no cover
            FIG["ok"] = False
            print(f"[warn] figures skipped: {type(exc).__name__}: {exc}")
    IMG = (lambda name, alt: f"\n![{alt}](figures/{name})\n") if FIG["ok"] else (lambda n, a: "\n")

    # ---- json -----------------------------------------------------------
    with open(out_dir / "profile_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)

    # ---- markdown -------------------------------------------------------
    sc, st, qa, ev = stats["scale"], stats["structure"], stats["qa"], stats["evidence"]
    L: list[str] = []
    L.append("# QASPER 数据集摸底报告（P1）\n")
    L.append(f"> 生成时间：{stats['generated_at']}　|　tokenizer：`{counter.backend}`")
    L.append("> 本报告只做**测量**，不做任何清洗或改写；所有数字都可用 "
             "`python scripts/00_profile_dataset.py` 复现。\n")

    L.append("## 1. 规模总览\n")
    L.append(df_to_md(tables["split_summary"]))
    L.append("")
    L.append(f"- 全语料正文 **{sc['total_body_tokens']:,}** tokens（含摘要 "
             f"{sc['total_abstract_tokens']:,}）")
    L.append(f"- 问题 **{qa['n_questions']:,}** 条，答案标注 **{qa['n_answers']:,}** 条")
    L.append("")

    L.append("## 2. 顶层字段（真实 dump，不是文档抄的）\n")
    L.append(df_to_md(tables["top_level_keys"]))
    L.append("")

    L.append("## 3. 文本规模分布\n")
    rows = []
    for k in ("body_tokens", "body_chars", "abstract_tokens"):
        d = sc[k]
        rows.append({"metric": k, **{kk: d[kk] for kk in ("n", "mean", "p50", "p90", "p99", "max")}})
    L.append(df_to_md(pd.DataFrame(rows)))
    L.append(IMG("p1_paper_body_tokens.png", "paper body tokens"))

    L.append("## 4. 文档结构\n")
    rows = []
    for k in ("sections_per_paper", "paras_per_section", "paragraph_tokens"):
        d = st[k]
        rows.append({"metric": k, **{kk: d[kk] for kk in ("mean", "p50", "p90", "p99", "max")}})
    L.append(df_to_md(pd.DataFrame(rows)))
    L.append("")
    L.append(df_to_md(pd.DataFrame([
        {"check": "总段落数", "value": f"{st['total_paragraphs']:,}"},
        {"check": "空段落数（text.strip()==''）",
         "value": f"{st['empty_paragraphs']} ({st['empty_paragraph_pct']}%)"},
        {"check": "无 section 的论文", "value": st["papers_with_zero_sections"]},
        {"check": "无摘要的论文", "value": st["papers_with_empty_abstract"]},
        {"check": "**存在 section_name 重名的论文**",
         "value": f"{st['papers_with_dup_section_name']} "
                  f"({st['papers_with_dup_section_name_pct']}%)"},
        {"check": "重复名称的额外出现次数", "value": st["extra_dup_occurrences"]},
    ])))
    L.append("")

    L.append("## 5. 问题与答案\n")
    rows = []
    for k in ("qas_per_paper", "answers_per_question", "evidence_per_question",
              "highlight_per_question"):
        d = qa[k]
        rows.append({"metric": k, **{kk: d[kk] for kk in ("mean", "p50", "p90", "p99", "max")}})
    L.append(df_to_md(pd.DataFrame(rows)))
    L.append("")
    L.append(df_to_md(pd.DataFrame([
        {"check": "没有答案的问题数", "value": qa["questions_without_answers"]},
        {"check": "违反『仅一个答案字段非空』的标注数",
         "value": f"{qa['answers_violating_single_type_rule']} "
                  f"({qa['answers_violating_pct']}%)"},
    ])))
    L.append("")
    L.append("**单个标注的答案类型分布**\n")
    L.append(df_to_md(tables["answer_type"]))
    L.append(IMG("p1_answer_type.png", "answer type"))
    L.append("**单个问题的答案类型组合**（多标注者可能给出不同类型）\n")
    L.append(df_to_md(tables["question_type_combo"].head(15)))
    L.append(IMG("p1_answers_per_question.png", "answers per question"))

    L.append("## 6. Evidence 形态与可对齐性 ★\n")
    L.append("这是 P6 能否顺利的唯一判据。\n")
    L.append(df_to_md(pd.DataFrame([
        {"bucket": "全部 evidence 条目", "count": ev["total_evidence_items"]},
        {"bucket": "`FLOAT SELECTED`（图表证据）", "count": f"{ev['float_items']} ({ev['float_pct_of_all']}%)"},
        {"bucket": "空字符串", "count": ev["empty_items"]},
        {"bucket": "**文本证据（对齐全部分母）**", "count": ev["textual_items"]},
        {"bucket": "└ 原文精确匹配", "count": f"{ev['exact_raw']}"},
        {"bucket": "└ 归一化后精确匹配", "count": f"{ev['exact_norm']}"},
        {"bucket": "└ 仅子串（containment）匹配", "count": f"{ev['substring']}"},
        {"bucket": "└ 命中摘要而非正文", "count": f"{ev['abstract']}"},
        {"bucket": "└ **完全失败**", "count": f"{ev['fail']}"},
        {"bucket": "**文本证据可对齐率**", "count": f"**{ev['alignable_pct_of_textual']}%**"},
        {"bucket": "highlighted_evidence 落在某段落内", "count": f"{ev['highlight_in_paragraph_pct']}% "
                                                                 f"({ev['highlight_items']} 条)"},
    ])))
    L.append(IMG("p1_evidence_buckets.png", "evidence buckets"))
    if ev["sample_failures"]:
        L.append("**对齐失败样例（供人工判读）**\n")
        for pid, txt in ev["sample_failures"]:
            L.append(f"- `{pid}` … {txt}")
        L.append("")

    L.append("## 7. S2ORC 占位符（BIBREF / TABREF / FIGREF）\n")
    aa = stats["artifacts"]
    L.append(df_to_md(pd.DataFrame([
        {"type": k, "occurrences": v} for k, v in aa["total_occurrences"].items()
    ])))
    L.append("")
    L.append(f"- 含占位符的段落：**{aa['paragraphs_with_artifacts']:,}** "
             f"（占全部段落 {aa['paragraphs_with_artifacts_pct']}%）")
    L.append(f"- 每篇论文占位符数：均值 {aa['per_paper']['mean']}，p90 {aa['per_paper']['p90']}，"
             f"max {aa['per_paper']['max']}")
    if aa["sample_paragraph"]:
        L.append("\n样例：\n")
        L.append("> " + aa["sample_paragraph"][:300].replace("\n", " "))
    L.append("")

    L.append("## 8. 写作元数据（Eval 组可做分层分析）\n")
    for f in ("meta_nlp_background", "meta_topic_background", "meta_paper_read"):
        L.append(f"**{f.removeprefix('meta_')}**\n")
        L.append(df_to_md(tables[f]))
        L.append("")
    L.append(f"- `search_query` 非空比例：**{stats['search_query_filled_pct']}%**")
    L.append(f"- 每篇论文图表数：均值 {stats['figures']['mean']}，p90 {stats['figures']['p90']}，"
             f"max {stats['figures']['max']}")
    L.append("")

    (out_dir / "dataset_profile.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[ok] wrote {out_dir / 'dataset_profile.md'}")
    print(f"[ok] wrote {out_dir / 'profile_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
