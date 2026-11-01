import json
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\Users\NTU\6405\project\RAG_Project")
P = ROOT / "data" / "processed"
lines = []

for name in ("pubmedqa_512_ov102", "pubmedqa_128_ov26"):
    df = pd.read_parquet(P / "pubmedqa" / "chunks" / name / "chunks.parquet")
    per = df.groupby("paper_id").size()
    vc = per.value_counts().sort_index()
    lines.append(f"=== {name}: chunks={len(df)} papers={per.size} "
                 f"chunks/paper mean={per.mean():.2f} max={per.max()}")
    lines.append("  distribution: " + ", ".join(f"{k}->{v}" for k, v in vc.items()))
    lines.append(f"  tokens/chunk mean={df['n_tokens'].mean():.1f} "
                 f"median={df['n_tokens'].median():.0f} max={df['n_tokens'].max()}")
    lines.append(f"  chunks with 1 paragraph: {int((df['para_ids'].apply(len) == 1).sum())} "
                 f"({(df['para_ids'].apply(len) == 1).mean() * 100:.1f}%)")

pm = json.loads((P / "pubmedqa" / "manifest.json").read_text(encoding="utf-8"))
lines.append("")
lines.append(f"pubmedqa manifest: questions={pm['questions']} paragraphs={pm['paragraphs']} "
             f"tokens={pm['tokens']} paras/paper={pm['paragraphs_per_paper']} "
             f"tokens/paper={pm['tokens_per_paper']}")
lines.append(f"decisions: {pm['decision_distribution']}")
lines.append(f"cleaner hits: {pm['cleaner_hits']}")
lines.append(f"para token stats: {pm['paragraph_tokens']}")

qs = json.loads((P / "manifest.json").read_text(encoding="utf-8"))
lines.append("")
lines.append(f"qasper paragraphs: {qs.get('paragraphs')} tokens={qs.get('tokens')}")
qn = json.loads((P / "manifest_qa.json").read_text(encoding="utf-8"))
lines.append(f"qasper qa keys: {list(qn.keys())[:12]}")

(Path(r"C:\Users\zxt123\AppData\Local\Temp") / "_b3.txt").write_text(
    "\n".join(lines), encoding="utf-8")
print("ok")
