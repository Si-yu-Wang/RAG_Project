"""Merge sharded grid CSVs into the canonical result file."""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

C = Path(r"D:\Users\NTU\6405\project\RAG_Project\data\processed\chunks")
shards = [C / "grid_results_a.csv", C / "grid_results_b.csv"]
parts = [pd.read_csv(p) for p in shards if p.exists()]
df = pd.concat(parts, ignore_index=True).sort_values(["chunk_size", "overlap_ratio"])
df.to_csv(C / "grid_results.csv", index=False)
meta = {
    "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "tokenizer": "tokenizers-file:tokenizer.json",
    "papers": 1169,
    "qa": 3598,
    "body_tokens": int(round(df.iloc[0]["tokens"] / df.iloc[0]["index_blowup"])),
    "limit_papers": 0,
    "sharded": True,
    "rows": df.to_dict("records"),
}
(C / "grid_results.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print("merged rows:", len(df))
print("sizes:", sorted(df["chunk_size"].unique().tolist()))
print("body_tokens:", meta["body_tokens"])
