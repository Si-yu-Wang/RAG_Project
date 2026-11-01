"""Check whether PubMedQA-1k is obtainable, and what its schema looks like.

Only a feasibility probe: we need to know (a) the raw file is reachable, (b) how many
records it has, (c) whether each record carries a context paragraph list (so we can decide
how it fits our paragraph/chunk schema).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

CANDIDATES = [
    "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json",
    "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqau.json",
]


def head(url: str) -> str:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "curl/8"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return f"OK {r.status} len={r.headers.get('Content-Length')}"
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}"
    except Exception as e:  # noqa: BLE001
        return type(e).__name__


def main() -> int:
    out = []
    for u in CANDIDATES:
        out.append(f"{head(u)}  {u}")

    # Try the main file in full, to report the real schema.
    url = CANDIDATES[0]
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode("utf-8"))
        out.append(f"\nrecords={len(data)}")
        pmid = next(iter(data))
        rec = data[pmid]
        out.append(f"top-level keys of a record: {sorted(rec.keys())}")
        ctx = rec.get("CONTEXTS") or []
        out.append(f"len(CONTEXTS)={len(ctx)}  first context: {str(ctx[0])[:200] if ctx else None}")
        out.append(f"avg context chars={sum(len(' '.join(c)) for c in ctx)/max(len(data),1):.0f}")
        out.append(f"sample QUESTION: {str(rec.get('QUESTION'))[:150]}")
        out.append(f"sample final_decision: {rec.get('final_decision')}")
        out.append(f"sample LONG_ANSWER: {str(rec.get('LONG_ANSWER'))[:150]}")
        out.append(f"labels: {sorted({v.get('final_decision') for v in data.values()})}")
    except Exception as e:  # noqa: BLE001
        out.append(f"\nfetch failed: {type(e).__name__}: {e}")

    open(os.path.join(os.environ["TEMP"], "_pubmedqa.txt"), "w", encoding="utf-8").write("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
