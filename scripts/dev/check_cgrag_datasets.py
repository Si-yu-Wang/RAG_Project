"""Check which datasets CG-RAG (arXiv:2501.15067) actually evaluates on.

Answers the question "does the project dataset (QASPER) appear in CG-RAG at all?"
by counting mentions in the full text rather than trusting our memory of the paper.
"""

from __future__ import annotations

import re
import sys

PDF = (
    r"D:\xwechat_files\wxid_s7kfjny52r4712_dfa5\temp\RWTemp\2026-10"
    r"\c3af6d1fed584bb67ad1be029d90d00e\2501.15067v1.pdf"
)
TERMS = [
    "QASPER", "Qasper", "PubMedQA", "PapersWithCode", "Papers with Code",
    "SQuAD", "Natural Questions", "TriviaQA", "HotpotQA", "BEIR", "MS MARCO",
    "BioASQ", "SciQ", "ARC", "QEvasion", "LitQA", "MuSiQue",
]

def main() -> int:
    import fitz

    doc = fitz.open(PDF)
    if doc.needs_pass:
        doc.authenticate("1010")
    text = "\n".join(page.get_text("text") for page in doc)
    flat = re.sub(r"\s+", " ", text)

    out = [f"pages={len(doc)} chars={len(flat)}"]
    out.append("--- term hit counts ---")
    for t in TERMS:
        n = len(re.findall(re.escape(t), flat, flags=re.IGNORECASE))
        out.append(f"{t:22s} {n}")
    out.append("--- dataset sentence contexts ---")
    for kw in ("PubMedQA-1k", "PapersWithCodeQA"):
        for m in list(re.finditer(re.escape(kw), flat))[:2]:
            s = max(0, m.start() - 160)
            out.append(f"[{kw}] ...{flat[s:m.end() + 160]}...")
            out.append("")
    with open(sys.argv[1], "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
