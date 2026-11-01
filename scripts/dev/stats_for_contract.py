import json
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\Users\NTU\6405\project\RAG_Project")
P = ROOT / "data" / "processed"
C = P / "chunks" / "size512_ov102_soft"

qa = pd.read_parquet(P / "qa.parquet")
lines = []
lines.append(f"questions total: {len(qa)}")
lines.append(f"is_answerable True: {int(qa['is_answerable'].sum())}  "
             f"False: {int((~qa['is_answerable']).sum())}")
lines.append("")
lines.append("zero_gold_reason x is_answerable:")
ct = pd.crosstab(qa["zero_gold_reason"].replace("", "(has gold)"), qa["is_answerable"])
lines.append(ct.to_string())
lines.append("")
lines.append(f"answer_types distribution (question level, first type):")
lines.append(qa["answer_types"].str.split(",").str[0].value_counts().to_string())
lines.append("")
lines.append(f"recall-eligible (gold_para_ids non-empty): "
             f"{int((qa['n_gold_paras'] > 0).sum())}")
lines.append(f"multi-gold-paragraph questions: {int((qa['n_gold_paras'] > 1).sum())}")
lines.append("")
gold = []
with open(C / "qa_with_gold.jsonl", encoding="utf-8") as f:
    for line in f:
        gold.append(json.loads(line))
g = pd.DataFrame([{"qid": r["question_id"], "n": len(r.get("gold_chunk_ids") or []),
                   "ng": r.get("n_gold_paras", 0)} for r in gold])
lines.append(f"gold chunk count: mean {g['n'].mean():.2f} median {g['n'].median():.0f} "
             f"p95 {g['n'].quantile(0.95):.0f} max {g['n'].max()}")
lines.append(f"questions with exactly 1 gold chunk: {int((g['n'] == 1).sum())} "
             f"({(g['n'] == 1).mean() * 100:.1f}%)")
lines.append("")
para = pd.read_parquet(P / "paragraphs.parquet")
lines.append(f"paragraphs: {len(para)}  papers: {para['paper_id'].nunique()}")
lines.append(f"paragraph tokens: mean {para['n_tokens'].mean():.1f} "
             f"median {para['n_tokens'].median():.0f} p95 {para['n_tokens'].quantile(0.95):.0f} "
             f"max {para['n_tokens'].max()}")
lines.append(f"splits: {para.groupby('split')['paper_id'].nunique().to_dict()}")
lines.append(f"section parents (:::) paragraphs: "
             f"{int(para['section_name'].str.contains(':::').sum())}")

(Path(r"C:\Users\zxt123\AppData\Local\Temp") / "_stats.txt").write_text(
    "\n".join(lines), encoding="utf-8")
print("ok")
