"""Evaluation-only boundary. Never import gold into a retriever or generator."""
import argparse
import json
from pathlib import Path
from .data import read_records, sha256


def export_requests(qa_path, output, split=None):
    rows = read_records(qa_path)
    requests, seen = [], set()
    for row in rows:
        if split is not None and row.get("split") != split:
            continue
        request = {key: row[key] for key in ("question_id", "paper_id", "question")}
        if "split" in row:
            request["split"] = row["split"]
        if any(not isinstance(request[k], str) or not request[k].strip() for k in ("question_id", "paper_id", "question")):
            raise ValueError("Invalid request identifiers/question")
        if request["question_id"] in seen:
            raise ValueError("Duplicate question_id")
        seen.add(request["question_id"])
        requests.append(request)
    if not requests:
        raise ValueError("No questions selected")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "x", encoding="utf-8") as f:
        for row in requests:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"count": len(requests), "split": split, "source_sha256": sha256(qa_path)}


def paragraph_recall(qa_rows, predictions, top_k=5):
    """Macro paragraph recall over eligible questions; missing predictions are errors."""
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be positive")
    by_id = {}
    for p in predictions:
        if p["question_id"] in by_id:
            raise ValueError("Duplicate prediction ID")
        by_id[p["question_id"]] = p
    values, seen = [], set()
    for q in qa_rows:
        if q["question_id"] in seen:
            raise ValueError("Duplicate QA ID")
        seen.add(q["question_id"])
        if q["zero_gold_reason"] != "":
            continue
        gold = set(q["gold_para_ids"])
        if not gold:
            raise ValueError("Eligible question has empty gold")
        p = by_id[q["question_id"]]
        if p["paper_id"] != q["paper_id"] or p["config"]["top_k"] < top_k:
            raise ValueError("Prediction paper/top_k mismatch")
        hit = set()
        for chunk in p["retrieved_chunks"][:top_k]:
            hit.update(chunk["para_ids"])
        values.append(len(hit & gold) / len(gold))
    return {"top_k": top_k, "eligible_questions": len(values),
            "macro_paragraph_recall": sum(values)/len(values) if values else None,
            "excluded_questions": len(qa_rows)-len(values)}


def main():
    parser = argparse.ArgumentParser(description="Evaluation-side export of label-free requests")
    parser.add_argument("--qa", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split")
    args = parser.parse_args()
    print(json.dumps(export_requests(args.qa, args.output, args.split), indent=2))


if __name__ == "__main__":
    main()
