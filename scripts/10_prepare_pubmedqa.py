"""B1–B2 / script 10: PubMedQA-1k → the same paragraph / QA schema as QASPER.

Why this exists: QASPER is the main line. PubMedQA is a cheap second domain used to answer one
question -- **does the pipeline generalise, or is it tuned to one dataset?** It is deliberately
NOT a place to run a chunk grid or build a graph, because the data cannot support either:

* each record is a single abstract (~2-3 paragraphs, a few hundred tokens), so ``chunk_size``
  has almost nothing to decide;
* there is no paragraph-level evidence annotation, only a yes/no/maybe decision, so evidence
  recall has no denominator and only answer-level accuracy is computable.

Both limits are *findings*, not obstacles, and they are quantified in
``docs/dataset/pubmedqa_report.md``. Emitting the same schema lets the identical downstream code run
unchanged, which is exactly what a smoke test is for.

Usage:
    python scripts/10_prepare_pubmedqa.py --raw-dir data/raw/pubmedqa \
        --processed-dir data/processed/pubmedqa --out-dir docs
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from rag_data.cleaner import Cleaner  # noqa: E402
from rag_data.tokenizer import DEFAULT_TOKENIZER_FILE, TokenCounter  # noqa: E402

URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
SPLIT = "test"  # PQA-L is the expert-labelled 1k set; the original paper uses it as test


def download(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    dst = raw_dir / "ori_pqal.json"
    if dst.exists() and dst.stat().st_size > 1_000_000:
        print(f"[ok] cached {dst} ({dst.stat().st_size/1e6:.2f} MB)")
        return dst
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    dst.write_bytes(data)
    print(f"[ok] downloaded {dst} ({len(data)/1e6:.2f} MB)")
    return dst


def context_paragraphs(rec: dict) -> list[str]:
    """``CONTEXTS`` is a list of paragraph strings; some releases nest sentences inside.

    Normalise both shapes to ``list[str]`` of paragraphs, dropping empties. Getting this wrong
    silently produces one giant paragraph or one-sentence paragraphs, and both change the
    "can this corpus even be chunked" answer.
    """
    out: list[str] = []
    for item in rec.get("CONTEXTS") or []:
        if isinstance(item, str):
            text = item
        elif isinstance(item, (list, tuple)):
            text = " ".join(str(x) for x in item)
        else:
            text = str(item)
        text = " ".join(text.split())
        if text:
            out.append(text)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/pubmedqa")
    ap.add_argument("--processed-dir", default="data/processed/pubmedqa")
    ap.add_argument("--out-dir", default="docs/dataset")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_FILE)
    args = ap.parse_args()

    raw_path = download(Path(args.raw_dir))
    data = json.loads(raw_path.read_text(encoding="utf-8"))
    print(f"[ok] records={len(data)}")

    counter = TokenCounter(args.tokenizer)
    cleaner = Cleaner()

    para_rows: list[dict] = []
    qa_rows: list[dict] = []
    rule_hits: dict[str, int] = {}
    ctx_counts: list[int] = []

    for pmid, rec in data.items():
        paras = context_paragraphs(rec)
        ctx_counts.append(len(paras))
        flat_cursor = 0
        para_ids: list[str] = []
        for i, ptext in enumerate(paras):
            res = cleaner.clean(ptext)
            for k, v in res.hits.items():
                rule_hits[k] = rule_hits.get(k, 0) + v
            pid = f"{pmid}#p{i:04d}"
            para_ids.append(pid)
            para_rows.append({
                "para_id": pid,
                "paper_id": pmid,
                "split": SPLIT,
                "section_idx": 0,
                "section_name": "Context",
                "para_idx": i,
                "is_abstract": True,
                "text_raw": ptext,
                "text_clean": res.text,
                "char_start": flat_cursor,
                "char_end": flat_cursor + len(res.text),
                "n_chars": len(res.text),
                "n_tokens": counter.count(res.text),
                "offset_map": "",
            })
            flat_cursor += len(res.text) + 2  # "\n\n" separator, same as the flattener

        decision = (rec.get("final_decision") or "").strip().lower()
        yes_no = True if decision == "yes" else (False if decision == "no" else None)
        long_answer = rec.get("LONG_ANSWER") or ""

        qa_rows.append({
            "question_id": pmid,
            "paper_id": pmid,
            "split": SPLIT,
            "question": rec.get("QUESTION") or "",
            "n_answers": 1,
            # PubMedQA is a three-way decision, not QASPER's four answer types. Keeping a
            # distinct label means nobody accidentally pools it with QASPER's distribution.
            "answer_types": "yes_no_maybe",
            "is_answerable": True,
            "n_evidence": 0,
            "n_evidence_float": 0,
            "n_evidence_mapped": 0,
            "n_evidence_failed": 0,
            "evidence_levels": "none",
            # Deliberately empty: PubMedQA annotates a decision, not supporting paragraphs.
            # Filling it with "all paragraphs" would make evidence recall trivially 100%.
            "gold_para_ids": [],
            "n_gold_paras": 0,
            "zero_gold_reason": "no_evidence",
            "is_float_only": False,
            "free_form_answers": [long_answer] if long_answer else [],
            "extractive_spans": [],
            "extractive_spans_located": [],
            "yes_no": yes_no,
            "writer_nlp_background": "",
            "writer_topic_background": "",
            "writer_paper_read": "",
            "search_query": "",
            "decision": decision,
        })

    proc = Path(args.processed_dir)
    proc.mkdir(parents=True, exist_ok=True)
    para = pd.DataFrame(para_rows)
    qa = pd.DataFrame(qa_rows)
    para.to_parquet(proc / "paragraphs.parquet", index=False)
    qa.to_parquet(proc / "qa.parquet", index=False)

    with open(proc / "qa_dataset.jsonl", "w", encoding="utf-8") as f:
        for rec in qa.to_dict("records"):
            rec["free_form_answers"] = [str(x) for x in rec["free_form_answers"]]
            rec["gold_para_ids"] = []
            rec["extractive_spans"] = []
            rec["extractive_spans_located"] = []
            if rec.get("yes_no") is None:
                rec["yes_no"] = None
            f.write(json.dumps(rec, ensure_ascii=False, allow_nan=False) + "\n")

    n_par = len(para)
    stats = {
        "dataset": "PubMedQA (PQA-L, 1k)",
        "source": URL,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": SPLIT,
        "questions": len(qa),
        "papers": int(para["paper_id"].nunique()),
        "paragraphs": n_par,
        "tokens": int(para["n_tokens"].sum()),
        "paragraphs_per_paper": {
            "mean": round(n_par / max(len(qa), 1), 2),
            "max": int(max(ctx_counts)) if ctx_counts else 0,
            "distribution": {str(k): int(v) for k, v in
                             pd.Series(ctx_counts).value_counts().sort_index().items()},
        },
        "paragraph_tokens": {
            "mean": round(float(para["n_tokens"].mean()), 1),
            "median": int(para["n_tokens"].median()),
            "p95": int(para["n_tokens"].quantile(0.95)),
            "max": int(para["n_tokens"].max()),
        },
        "tokens_per_paper": {
            "mean": round(float(para.groupby("paper_id")["n_tokens"].sum().mean()), 1),
            "max": int(para.groupby("paper_id")["n_tokens"].sum().max()),
        },
        "decision_distribution": {str(k): int(v) for k, v in
                                  qa["decision"].value_counts().items()},
        "cleaner_hits": rule_hits,
        "gold_paragraphs_available": False,
        "note": "PubMedQA 标注的是 yes/no/maybe 判定，不是证据段落，"
                "因此 gold_para_ids 全部为空、zero_gold_reason=no_evidence。",
    }
    (proc / "manifest.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] wrote {proc} paragraphs={len(para)} qa={len(qa)} "
          f"tokens={stats['tokens']:,}")
    print(f"[ok] paragraphs/paper mean={stats['paragraphs_per_paper']['mean']} "
          f"max={stats['paragraphs_per_paper']['max']}")
    print(f"[ok] tokens/paper mean={stats['tokens_per_paper']['mean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
