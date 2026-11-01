"""Probe whether QASPER can be given a real citation graph.

QASPER ships text only -- no references. The plan is to fetch each paper's reference list
from the Semantic Scholar Graph API (QASPER `paper_id` is already an arXiv id) and turn it
into graph edges for Member 3.

Two things must be measured before promising anything:

1. **coverage** -- how many references can we resolve to an arXiv id at all;
2. **density** -- of the edges we can build, how many have *both* endpoints inside our
   1,169-paper corpus. Direct citations are usually sparse, because a 2019 NLP paper cites
   work from 2015-2018 that QASPER does not contain. So we also measure **bibliographic
   coupling** (two papers sharing at least one reference), which yields in-corpus edges
   without needing the cited paper to be in the corpus.

Usage:
    python scripts/dev/probe_citation_graph.py --n 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

S2 = "https://api.semanticscholar.org/graph/v1/paper/arXiv:{pid}?fields=title,references.title,references.externalIds,references.year"
UA = "Mozilla/5.0 (rag-project citation probe)"


def fetch(pid: str, timeout: int = 30) -> dict | None:
    req = urllib.request.Request(S2.format(pid=pid), headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"_error": f"HTTP {e.code}"}
    except Exception as e:  # noqa: BLE001
        return {"_error": type(e).__name__}


def ref_arxiv_ids(paper: dict) -> set[str]:
    out: set[str] = set()
    for ref in paper.get("references") or []:
        ext = (ref or {}).get("externalIds") or {}
        aid = ext.get("ArXiv")
        if aid:
            out.add(str(aid).strip().lower())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw/qasper")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--sleep", type=float, default=1.1, help="seconds between API calls")
    ap.add_argument("--out", default="docs/experiments/citation_graph_probe.md")
    args = ap.parse_args()

    dev = Path(args.data_dir) / "qasper-dev-v0.3.json"
    train = Path(args.data_dir) / "qasper-train-v0.3.json"
    corpus: dict[str, str] = {}
    for p in (train, dev):
        if p.exists():
            for pid in json.loads(p.read_text(encoding="utf-8")):
                corpus[pid.strip().lower()] = p.stem
    sample = [pid for pid in list(corpus)[: args.n]]

    rows = []
    errors = 0
    for i, pid in enumerate(sample, 1):
        data = fetch(pid)
        if not data or "_error" in data:
            errors += 1
            rows.append({"pid": pid, "status": (data or {}).get("_error", "fail"),
                         "n_refs": 0, "n_ref_arxiv": 0, "n_in_corpus": 0, "refs": set()})
            print(f"[{i}/{len(sample)}] {pid} ERROR {(data or {}).get('_error')}")
            time.sleep(args.sleep)
            continue
        refs = data.get("references") or []
        aids = ref_arxiv_ids(data)
        in_corpus = {a for a in aids if a in corpus}
        rows.append({"pid": pid, "status": "ok", "n_refs": len(refs),
                     "n_ref_arxiv": len(aids), "n_in_corpus": len(in_corpus), "refs": aids})
        print(f"[{i}/{len(sample)}] {pid} refs={len(refs)} arxiv={len(aids)} in_corpus={len(in_corpus)}")
        time.sleep(args.sleep)

    ok = [r for r in rows if r["status"] == "ok"]
    n_refs = sum(r["n_refs"] for r in ok)
    n_arxiv = sum(r["n_ref_arxiv"] for r in ok)
    n_in = sum(r["n_in_corpus"] for r in ok)
    with_any = sum(1 for r in ok if r["n_in_corpus"] > 0)

    # direct edges among the sample
    direct = set()
    for r in ok:
        for a in r["refs"]:
            if a in corpus:
                direct.add((r["pid"], a))

    # bibliographic coupling inside the sample
    pairs = 0
    weight_sum = 0
    for i, a in enumerate(ok):
        for b in ok[i + 1:]:
            shared = len(a["refs"] & b["refs"])
            if shared:
                pairs += 1
                weight_sum += shared

    out = []
    out.append(f"sampled={len(sample)} ok={len(ok)} errors={errors}")
    out.append(f"refs_total={n_refs} refs_with_arxiv={n_arxiv} "
               f"({n_arxiv / max(n_refs,1) * 100:.1f}%)")
    out.append(f"refs_pointing_into_corpus={n_in} ({n_in / max(n_arxiv,1) * 100:.1f}% of arxiv refs)")
    out.append(f"papers_with_at_least_one_in_corpus_ref={with_any}/{len(ok)} "
               f"({with_any / max(len(ok),1) * 100:.1f}%)")
    out.append(f"mean_in_corpus_refs_per_paper={n_in / max(len(ok),1):.2f}")
    out.append(f"direct_edges_among_sample={len(direct)}")
    pairs_total = len(ok) * (len(ok) - 1) // 2
    out.append(f"coupling_pairs={pairs}/{pairs_total} "
               f"({pairs / max(pairs_total,1) * 100:.1f}%), mean_shared_refs={weight_sum / max(pairs,1):.1f}")
    Path(os.path.join(os.environ["TEMP"], "_cite_probe.txt")).write_text(
        "\n".join(out), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
