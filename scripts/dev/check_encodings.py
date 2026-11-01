"""Guard: every text file in the repo must be UTF-8 with no NUL bytes.

Why this exists: the editor preserves a file's existing encoding on overwrite, so a single
UTF-16 source file stays UTF-16 forever. Git then treats it as binary -- no diffs, no review,
and GitHub renders mojibake. It is invisible until someone opens the file on another machine.
"""

import subprocess
from pathlib import Path

ROOT = Path(r"D:\Users\NTU\6405\project\RAG_Project")
EXTS = {".md", ".py", ".json", ".csv", ".toml", ".txt", ".cfg", ".yml", ".yaml"}

tracked = subprocess.run(["git", "ls-files"], cwd=str(ROOT),
                         capture_output=True).stdout.decode("utf-8").splitlines()

bad = []
checked = 0
for rel in tracked:
    p = ROOT / rel
    if p.suffix.lower() not in EXTS or not p.exists():
        continue
    checked += 1
    b = p.read_bytes()
    if b.count(0):
        bad.append((rel, "NUL bytes", len(b)))
        continue
    try:
        b.decode("utf-8")
    except UnicodeDecodeError as e:
        bad.append((rel, f"not utf-8: {e}", len(b)))

print("checked_files:", checked)
print("bad_files:", len(bad))
for rel, why, size in bad:
    print("  BAD", rel, size, why)
