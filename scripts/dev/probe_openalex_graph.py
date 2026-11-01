"""Second probe: OpenAlex instead of Semantic Scholar.

The S2 probe showed two problems -- heavy 429 rate limiting, and **zero** references
pointing back into the QASPER corpus. This probe checks whether OpenAlex (no API key,
generous limits, DOI-level `referenced_works`) does better, and measures how many
in-corpus edges we can get via *bibliographic coupling* (two papers sharing a reference).

Usage:
    python scripts/dev/probe_openalex_graph.py --n 20
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://api.openalex.org/works"
UA = "Mozilla/5.0 (rag-project; mailto:student@example.edu)"


def get(url: str, timeout: int = 30) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            return {"_error": f"HTTP {e.code}"}
        except Exception as e:  # noqa: BLE001
            return {"_error": type(e).__name__}
    return {"_error": "HTTP 429 x3"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw/qasper")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--mailto", default="rag-project@example.edu")
    args = ap.parse_args()

    dev = Path(args.data_dir) / "qasper-dev-v0.3.json"
    raw = json.loads(dev.read_text(encoding="utf-8"))
    keys = list(raw)[: args.n]

    rows = []
    for i, pid in enumerate(keys, 1):
        title = (raw[pid].get("title") or "").strip()
        q = urllib.parse.urlencode(
            {"search": title, "per-page": 1, "select": "id,title,referenced_works", "mailto": args.mailto}
        )
        data = get(f"{BASE}?{q}")
        if not data or "_error" in data or not data.get("results"):
            rows.append({"pid": pid, "status": "fail", "nref": 0, "refs": set()})
            print(f"[{i}/{len(keys)}] {pid} FAIL {title[:40]}")
            continue
        w = data["results"][0]
        refs = set(w.get("referenced_works") or [])
        rows.append({"pid": pid, "status": "ok", "nref": len(refs), "refs": refs,
                     "oa_id": w.get("id")})
        print(f"[{i}/{len(keys)}] {pid} refs={len(refs)} {title[:40]}")
        time.sleep(0.12)

    ok = [r for r in rows if r["status"] == "ok"]
    oa_ids = {r["oa_id"]: r["pid"] for r in ok}
    n_in = 0
    for r in ok:
        n_in += len({x for x in r["refs"] if x in oa_ids})
    pairs = 0
    shared_sum = 0
    for i, a in enumerate(ok):
        for b in ok[i + 1:]:
            s = len(a["refs"] & b["refs"])
            if s:
                pairs += 1
                shared_sum += s
    total_pairs = len(ok) * (len(ok) - 1) // 2

    out = [
        f"sampled={len(keys)} ok={len(ok)}",
        f"mean_referenced_works={sum(r['nref'] for r in ok) / max(len(ok),1):.1f}",
        f"direct_edges_into_sample={n_in}",
        f"papers_with_in_sample_citation={sum(1 for r in ok if any(x in oa_ids for x in r['refs']))}/{len(ok)}",
        f"coupling_pairs={pairs}/{total_pairs} ({pairs/max(total_pairs,1)*100:.1f}%) "
        f"mean_shared={shared_sum/max(pairs,1):.1f}",
    ]
    Path(os.path.join(os.environ["TEMP"], "_oa_probe.txt")).write_text("\n".join(out), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
