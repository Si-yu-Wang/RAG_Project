"""Expand PubMedQA-1k into a citation-graph corpus (CG-RAG style).

Stages
------
P0  PMID -> OpenAlex work (+ referenced_works)
P1  cited paper -> cited paper edges
P2  expand corpus (source papers + all references), rebuild abstracts,
    optional backfill from PubMed E-utilities

Outputs (under --out)
----------------------
source_papers.parquet  pmid, openalex_id, title, year, n_refs, referenced_works
papers.parquet         paper_id, role, pmid, title, year, abstract, has_abstract
edges.parquet          src_paper, dst_paper, edge_type='cites'
report.json            counts / coverage / timing

Usage
-----
    python scripts/20_expand_pubmedqa_corpus.py --limit 20
    python scripts/20_expand_pubmedqa_corpus.py            # full 1,000
"""

from __future__ import annotations

import argparse
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd

socket.setdefaulttimeout(20)  # hard cap on DNS/connect/read stalls
BASE = "https://api.openalex.org/works"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
UA = "Mozilla/5.0 (rag-project; mailto:student@example.edu)"


def _get(url: str, timeout: int = 20, tries: int = 3) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429 or 500 <= e.code < 600:
                time.sleep(5 * (attempt + 1))
                continue
            return None
        except Exception:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
    return None


def _cached(cache_dir: Path, key: str, url: str) -> dict | None:
    p = cache_dir / f"{key}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    d = _get(url)
    if d is not None:
        p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return d


def rebuild_abstract(inv: dict | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; rebuild the text."""
    if not inv:
        return ""
    pos: dict[int, str] = {}
    for word, ps in inv.items():
        for p in ps:
            pos[p] = word
    return " ".join(pos[i] for i in sorted(pos))


def wid(x: str | None) -> str | None:
    """'https://openalex.org/W123' -> 'W123'."""
    return x.rsplit("/", 1)[-1] if x else None


def fetch_pmid(pmid: str, cache_dir: Path, mailto: str) -> dict | None:
    q = urllib.parse.urlencode(
        {
            "filter": f"ids.pmid:{pmid}",
            "select": "id,title,publication_year,referenced_works",
            "mailto": mailto,
        }
    )
    d = _cached(cache_dir, f"src_{pmid}", f"{BASE}?{q}")
    if not d:
        return None
    res = d.get("results") or []
    return res[0] if res else None


def fetch_batch(ids: list[str], cache_dir: Path, batch_idx: int, mailto: str) -> list[dict]:
    q = urllib.parse.urlencode(
        {
            "filter": "ids.openalex:" + "|".join(ids),
            "select": "id,title,abstract_inverted_index,ids,publication_year",
            "per-page": 50,
            "mailto": mailto,
        }
    )
    d = _cached(cache_dir, f"batch50_{batch_idx:05d}", f"{BASE}?{q}")
    return (d or {}).get("results") or []


def backfill_pubmed(pmid: str, cache_dir: Path) -> str:
    """Fetch an abstract from PubMed when OpenAlex has none."""
    import xml.etree.ElementTree as ET

    p = cache_dir / f"pm_{pmid}.txt"
    if p.exists():
        return p.read_text(encoding="utf-8")
    url = f"{EUTILS}?db=pubmed&id={pmid}&retmode=xml"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    text = ""
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            root = ET.fromstring(r.read())
        parts = [el.text or "" for el in root.iter("AbstractText")]
        text = " ".join(x.strip() for x in parts if x.strip())
    except Exception:  # noqa: BLE001
        text = ""
    p.write_text(text, encoding="utf-8")
    time.sleep(0.35)  # E-utilities: <= 3 req/s
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/pubmedqa/ori_pqal.json")
    ap.add_argument("--out", default="data/processed/pubmedqa_graph")
    ap.add_argument("--cache", default="data/raw/openalex_cache")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--mailto", default="rag-project@example.edu")
    ap.add_argument("--backfill", action="store_true", help="fill missing abstracts from PubMed")
    ap.add_argument(
        "--min-cited",
        type=int,
        default=1,
        help="only fetch references cited by >= N source papers (2 = coupling core; 1 = all)",
    )
    args = ap.parse_args()

    out = Path(args.out)
    cache = Path(args.cache)
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    raw = json.loads(Path(args.raw).read_text(encoding="utf-8"))
    pmids = list(raw.keys())
    if args.limit:
        pmids = pmids[: args.limit]
    print(f"[P0] source papers: {len(pmids)}")

    # ---- P0 + P1 -------------------------------------------------------
    src_rows, edges = [], []
    for i, pmid in enumerate(pmids, 1):
        had = (cache / f"src_{pmid}.json").exists()
        w = fetch_pmid(pmid, cache, args.mailto)
        if not w:
            print(f"  [{i}/{len(pmids)}] PMID {pmid} NOT FOUND", flush=True)
            if not had:
                time.sleep(0.12)
            continue
        oa = wid(w["id"])
        refs = [wid(x) for x in (w.get("referenced_works") or []) if wid(x)]
        src_rows.append(
            {
                "pmid": pmid,
                "openalex_id": oa,
                "title": w.get("title") or "",
                "year": w.get("publication_year"),
                "n_refs": len(refs),
                "referenced_works": refs,
            }
        )
        edges.extend({"src_paper": oa, "dst_paper": r, "edge_type": "cites"} for r in refs)
        if i % 25 == 0:
            print(f"  [{i}/{len(pmids)}] resolved", flush=True)
        if not had:
            time.sleep(0.12)

    src = pd.DataFrame(src_rows)
    print(f"[P1] source resolved {len(src)}/{len(pmids)}, cites edges {len(edges)}")

    # ---- P2 ------------------------------------------------------------
    cited = Counter(e["dst_paper"] for e in edges)
    ref_ids = sorted({d for d, c in cited.items() if c >= args.min_cited} - set(src["openalex_id"]))
    print(f"[P2] references to fetch (cited >= {args.min_cited}): {len(ref_ids)}", flush=True)

    ref_rows: list[dict] = []
    B = 50
    for i in range(0, len(ref_ids), B):
        chunk = ref_ids[i : i + B]
        works = fetch_batch(chunk, cache, i // B, args.mailto)
        for w in works:
            ab = rebuild_abstract(w.get("abstract_inverted_index"))
            pm = None
            ids = w.get("ids") or {}
            if ids.get("pmid"):
                pm = ids["pmid"].rsplit("/", 1)[-1]
            ref_rows.append(
                {
                    "paper_id": wid(w["id"]),
                    "role": "reference",
                    "pmid": pm,
                    "title": w.get("title") or "",
                    "year": w.get("publication_year"),
                    "abstract": ab,
                    "has_abstract": bool(ab),
                }
            )
        if (i // B) % 10 == 0:
            print(f"  [P2] batch {i // B + 1}/{(len(ref_ids) + B - 1) // B} rows={len(ref_rows)}", flush=True)
        time.sleep(0.3)

    # source rows (text = CONTEXTS, richer than the OpenAlex abstract)
    src_papers = []
    for r in src.itertuples():
        rec = raw[r.pmid]
        ctx = " ".join(rec.get("CONTEXTS") or [])
        src_papers.append(
            {
                "paper_id": r.openalex_id,
                "role": "source",
                "pmid": r.pmid,
                "title": r.title,
                "year": r.year,
                "abstract": ctx,
                "has_abstract": bool(ctx),
            }
        )

    papers = pd.concat([pd.DataFrame(src_papers), pd.DataFrame(ref_rows)], ignore_index=True)
    papers = papers.drop_duplicates(subset=["paper_id"], keep="first")

    # ---- optional backfill --------------------------------------------
    backfilled = 0
    if args.backfill:
        miss = papers[(~papers.has_abstract) & papers.pmid.notna()]
        print(f"[P2] backfilling {len(miss)} papers from PubMed ...")
        for j, r in enumerate(miss.itertuples(), 1):
            ab = backfill_pubmed(str(r.pmid), cache)
            if ab:
                papers.loc[papers.paper_id == r.paper_id, "abstract"] = ab
                papers.loc[papers.paper_id == r.paper_id, "has_abstract"] = True
                backfilled += 1
            if j % 25 == 0:
                print(f"  backfilled {j}/{len(miss)}")

    # ---- write ---------------------------------------------------------
    src.to_parquet(out / "source_papers.parquet", index=False)
    papers.to_parquet(out / "papers.parquet", index=False)
    pd.DataFrame(edges).to_parquet(out / "edges.parquet", index=False)

    n_ref = int((papers.role == "reference").sum())
    n_abs = int(papers.has_abstract.sum())
    report = {
        "source_papers": int(len(src)),
        "source_requested": len(pmids),
        "references_fetched": n_ref,
        "corpus_papers": int(len(papers)),
        "papers_with_abstract": n_abs,
        "abstract_coverage_pct": round(n_abs / max(len(papers), 1) * 100, 1),
        "refs_abstract_coverage_pct": round(
            papers[papers.role == "reference"].has_abstract.mean() * 100, 1
        )
        if n_ref
        else None,
        "backfilled": backfilled,
        "cites_edges": len(edges),
        "mean_refs_per_source": round(src.n_refs.mean(), 1) if len(src) else None,
        "unique_refs": len(cnt),
        "min_cited": args.min_cited,
        "refs_selected": len(ref_ids),
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[REPORT]")
    for k, v in report.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
