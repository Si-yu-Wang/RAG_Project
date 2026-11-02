"""Evaluation-only boundary. Never import gold into a retriever or generator."""
import argparse
import json
import sys
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


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Existing `--qa ... --output ...` export commands remain valid.
    if argv and argv[0].startswith('--') and argv[0] not in ('--help', '-h'):
        argv.insert(0, 'export')
    parser = argparse.ArgumentParser(description="Export label-free requests or evaluate retrieval")
    sub = parser.add_subparsers(dest='command', required=True)
    export = sub.add_parser('export', help='Export label-free requests (evaluation-side)')
    export.add_argument('--qa', required=True)
    export.add_argument('--output', required=True)
    export.add_argument('--split')
    score = sub.add_parser('score', help='Validate a full split and compute paragraph Recall and chunk MRR')
    score.add_argument('--qa', required=True)
    score.add_argument('--predictions', required=True)
    score.add_argument('--index', required=True, help='Index bundle used as the authoritative chunk corpus')
    score.add_argument('--split', default='dev')
    score.add_argument('--ks', nargs='+', type=int, default=[1, 3, 5, 10])
    score.add_argument('--output', required=True, help='New metrics JSON file')
    score.add_argument('--report', help='Optional new Markdown report')
    args = parser.parse_args(argv)
    if args.command == 'export':
        print(json.dumps(export_requests(args.qa, args.output, args.split), indent=2))
        return
    from .retrieval_metrics import evaluate, render_report
    outputs = [Path(args.output)] + ([Path(args.report)] if args.report else [])
    if len({p.resolve() for p in outputs}) != len(outputs):
        parser.error('--output and --report must be different paths')
    for output in outputs:
        if output.exists():
            parser.error(f'Output already exists: {output}')
    report = evaluate(args.qa, args.predictions, args.index, args.split, args.ks)
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
    with outputs[0].open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    if args.report:
        with outputs[1].open('x', encoding='utf-8') as f:
            f.write(render_report(report))
    print(render_report(report))


if __name__ == "__main__":
    main()
