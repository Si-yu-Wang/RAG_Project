"""Measure every file under data/ so we can decide what must ship vs. what is too big."""
from pathlib import Path

ROOT = Path(r"D:\Users\NTU\6405\project\RAG_Project")
data = ROOT / "data"

rows = []
for p in sorted(data.rglob("*")):
    if p.is_file():
        rows.append((p.relative_to(ROOT).as_posix(), p.stat().st_size))

rows.sort(key=lambda t: -t[1])
total = sum(s for _, s in rows)
print(f"TOTAL data/ = {total/1e6:.2f} MB, {len(rows)} files\n")
for rel, s in rows:
    print(f"{s/1e6:9.2f} MB  {s:>12,}  {rel}")
