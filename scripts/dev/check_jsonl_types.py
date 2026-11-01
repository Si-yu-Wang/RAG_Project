import json
from pathlib import Path

ROOT = Path(r"D:\Users\NTU\6405\project\RAG_Project")
C = ROOT / "data" / "processed" / "chunks" / "size512_ov102_soft"
P = ROOT / "data" / "processed"

lines = []
with open(C / "qa_with_gold.jsonl", encoding="utf-8") as f:
    rec = json.loads(f.readline())
for k in ("gold_para_ids", "gold_chunk_ids", "free_form_answers",
          "extractive_spans", "extractive_spans_located"):
    v = rec.get(k)
    lines.append(f"qa_with_gold  {k:<24} type={type(v).__name__:<6} sample={repr(v)[:80]}")

with open(P / "qa_dataset.jsonl", encoding="utf-8") as f:
    rec2 = json.loads(f.readline())
for k in ("gold_para_ids", "free_form_answers", "extractive_spans",
          "extractive_spans_located"):
    v = rec2.get(k)
    lines.append(f"qa_dataset    {k:<24} type={type(v).__name__:<6} sample={repr(v)[:80]}")

with open(C / "chunks.jsonl", encoding="utf-8") as f:
    c = json.loads(f.readline())
for k in ("para_ids", "para_spans", "prev_chunk_id"):
    v = c.get(k)
    lines.append(f"chunks.jsonl  {k:<24} type={type(v).__name__:<6} sample={repr(v)[:80]}")

(Path(r"C:\Users\zxt123\AppData\Local\Temp") / "_types.txt").write_text(
    "\n".join(lines), encoding="utf-8")
print("ok")
