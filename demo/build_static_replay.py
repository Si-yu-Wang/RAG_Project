"""Export a query-switchable graph-to-score page. No model execution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "demo/static"
DEFAULT_ARCHIVE = Path(r"D:\WeChatData\xwechat_files\wxid_k495a29oaoxo12_9c35\msg\file\2026-11\webqsp_complete_no_index_20261109.tar.gz")
QUESTION_ID = "WebQTest-380"
QUESTION_IDS = (QUESTION_ID, "WebQTest-84", "WebQTest-124", "WebQTest-302", "WebQTest-237")
PREDICTIONS = "artifacts/webqsp_test_member2/predictions.jsonl"
RANKING = "rankings/webqsp_rearev_s1_test.jsonl"


def rows(stream):
    for line in stream:
        if line.strip():
            yield json.loads(line)


def load_examples(archive_path):
    with tarfile.open(archive_path, "r:gz") as archive:
        texts, questions = {}, {}
        for record in rows(archive.extractfile(PREDICTIONS)):
            for hit in record["retrieved_chunks"]:
                texts.setdefault(hit["chunk_id"], hit["text"])
            if record["question_id"] in QUESTION_IDS:
                questions[record["question_id"]] = record
        rankings = {r["question_id"]: r for r in rows(archive.extractfile(RANKING)) if r["question_id"] in QUESTION_IDS}
    examples = []
    for i, qid in enumerate(QUESTION_IDS):
        example = build_example(qid, questions[qid], rankings[qid], texts, archive_path.name)
        example["id"] = f"query-{i+1}"
        examples.append(example)
    return {"schema": "gnn-query-demo/v1", "queries": examples}


def build_example(question_id, record, ranking, texts, archive_name):
    question, hits = record["question"], record["retrieved_chunks"]
    score_map = {cid: {"score": score, "rank": i+1} for i, (cid, score) in enumerate(zip(ranking["ranked_chunk_ids"], ranking["scores"]))}
    top = ranking["ranked_chunk_ids"][:5]
    # Include three readable connected neighbors to show both relevance and topology.
    context_labels = {"Volcano", "Crater Glacier", "Volcano Adventure"} if question_id == QUESTION_ID else set()
    extra = [h["chunk_id"] for h in hits if h["text"].splitlines()[0].split(" | ")[-1] in context_labels]
    visible = list(dict.fromkeys(top+extra))
    nodes, edges, path_nodes, seen_edges = [], [], {}, set()
    candidate_ids = {cid: f"n{i+1}" for i, cid in enumerate(visible)}
    for cid in visible:
        text = texts[cid]
        for path in text.splitlines():
            path_nodes[path] = candidate_ids[cid]
        nodes.append({"id": candidate_ids[cid], "label": text.splitlines()[0].split(" | ")[-1], "kind": "candidate",
                      "text": text, "score": score_map[cid]["score"], "rank": score_map[cid]["rank"], "source_id": cid})
    all_ids = {n["id"] for n in nodes}
    for cid in visible:
        for text in texts[cid].splitlines():
            parts = text.split(" | ")
            if len(parts) < 3 or len(parts) % 2 != 1:
                raise ValueError("Expected a saved entity-relation path")
            previous = None
            for i in range(0, len(parts), 2):
                prefix = " | ".join(parts[:i+1])
                if prefix not in path_nodes:
                    node_id = f"s{len(path_nodes)}"
                    path_nodes[prefix] = node_id
                    nodes.append({"id": node_id, "label": parts[i], "kind": "source" if i==0 else "entity"})
                    all_ids.add(node_id)
                node_id = path_nodes[prefix]
                if previous is not None:
                    edge_key = previous, node_id, parts[i-1]
                    if edge_key not in seen_edges:
                        edges.append({"source": previous, "target": node_id, "relation": parts[i-1], "provenance": "saved_relation_path"})
                        seen_edges.add(edge_key)
                previous = node_id
    results = [{"node_id": candidate_ids[cid], "score": score_map[cid]["score"], "rank": i+1,
                "label": texts[cid].splitlines()[0].split(" | ")[-1], "source_id": cid} for i,cid in enumerate(top)]
    if not all(e["source"] in all_ids and e["target"] in all_ids for e in edges):
        raise ValueError("Graph contains an unknown node")
    return {"schema": "gnn-score-demo/v1", "question": question, "nodes": nodes, "edges": edges, "results": results,
            "provenance": {"question_id": question_id, "source_archive": archive_name,
                "predictions_member": PREDICTIONS, "ranking_member": RANKING,
                "ranking_sha256": hashlib.sha256(json.dumps(ranking, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "full_candidate_count": len(ranking["ranked_chunk_ids"]), "intermediate_states": False,
                "graph_scope": "Subset expanded from saved relation paths, not the complete original subgraph",
                "score_scope": "Original full-pool node probabilities; not renormalized over visible nodes"}}


def write_artifacts(example):
    serialized = json.dumps(example, ensure_ascii=False, separators=(",", ":"))
    for char, replacement in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026"), ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        serialized = serialized.replace(char, replacement)
    data = "globalThis.GNN_DEMO = " + serialized + ";\n"
    (STATIC / "replay-data.js").write_text(data, encoding="utf-8")
    html = (STATIC / "replay.html").read_text(encoding="utf-8")
    html = html.replace('<link rel="stylesheet" href="replay.css">', '<style>'+ (STATIC / "replay.css").read_text(encoding="utf-8")+'</style>')
    html = html.replace('<script src="replay-data.js"></script>', '<script>'+data+'</script>')
    html = html.replace('<script src="replay.js"></script>', '<script>'+ (STATIC / "replay.js").read_text(encoding="utf-8")+'</script>')
    destination = ROOT / "demo/rearev_replay.html"
    destination.write_text(html, encoding="utf-8")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--reuse-data", action="store_true", help="Rebuild the HTML using the already bundled queries")
    args = parser.parse_args()
    if args.reuse_data:
        raw = (STATIC / "replay-data.js").read_text(encoding="utf-8")
        example = json.loads(raw.split(" = ", 1)[1].rstrip(";\n"))
        if example.get("schema") != "gnn-query-demo/v1":
            raise ValueError("Rebuild from the source archive before reusing this data")
    else:
        example = load_examples(args.archive)
    destination = write_artifacts(example)
    print(json.dumps({"queries": len(example["queries"]),
                      "html": str(destination), "bytes": destination.stat().st_size}))


if __name__ == "__main__":
    main()
