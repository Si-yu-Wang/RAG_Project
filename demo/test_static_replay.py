"""Validate query graphs and score exports against saved model predictions."""
import json
from pathlib import Path
import sys
import tarfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_static_replay import DEFAULT_ARCHIVE, QUESTION_IDS, RANKING, STATIC, rows


class RecordedDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = (STATIC / "replay-data.js").read_text(encoding="utf-8")
        cls.bundle = json.loads(raw.split(" = ", 1)[1].rstrip(";\n"))
        cls.queries = cls.bundle["queries"]

    def test_connected_graph_and_scores_share_node_identity(self):
        self.assertEqual(self.bundle["schema"], "gnn-query-demo/v1")
        self.assertEqual(len(self.queries), 5)
        self.assertEqual(len({q["id"] for q in self.queries}), 5)
        for data in self.queries:
            with self.subTest(query=data["id"]):
                nodes = {n["id"]: n for n in data["nodes"]}
                self.assertEqual(len(nodes), len(data["nodes"]))
                sources = [n for n in nodes.values() if n["kind"] == "source"]
                self.assertEqual(len(sources), 1)
                reached = {sources[0]["id"]}
                for _ in nodes:
                    reached |= {e["target"] for e in data["edges"] if e["source"] in reached}
                self.assertEqual(reached, set(nodes))
                for edge in data["edges"]:
                    self.assertIn(edge["source"], nodes)
                    self.assertIn(edge["target"], nodes)
                    self.assertEqual(edge["provenance"], "saved_relation_path")
                for result in data["results"]:
                    node = nodes[result["node_id"]]
                    self.assertEqual(result["score"], node["score"])
                    self.assertEqual(result["label"], node["label"])
                self.assertFalse(data["provenance"]["intermediate_states"])

    def test_real_multi_edge_path_is_preserved(self):
        data = self.queries[0]
        nodes = {n["label"]: n["id"] for n in data["nodes"]}
        edges = {(e["source"], e["target"]) for e in data["edges"]}
        self.assertIn((nodes["Mount St. Helens"], nodes["Volcano"]), edges)
        self.assertIn((nodes["Volcano"], nodes["Volcano Adventure"]), edges)

    @unittest.skipUnless(DEFAULT_ARCHIVE.exists(), "Original archive unavailable; the HTML is self-contained")
    def test_every_visible_probability_matches_original_full_pool_output(self):
        with tarfile.open(DEFAULT_ARCHIVE) as archive:
            records = {r["question_id"]: r for r in rows(archive.extractfile(RANKING)) if r["question_id"] in QUESTION_IDS}
        for data in self.queries:
            record = records[data["provenance"]["question_id"]]
            source = {cid: score for cid, score in zip(record["ranked_chunk_ids"], record["scores"])}
            for node in data["nodes"]:
                if node["kind"] == "candidate":
                    self.assertEqual(node["score"], source[node["source_id"]])
            self.assertEqual([r["source_id"] for r in data["results"]], record["ranked_chunk_ids"][:5])
            self.assertEqual([r["score"] for r in data["results"]], record["scores"][:5])
            self.assertEqual(data["provenance"]["full_candidate_count"], len(source))


if __name__ == "__main__":
    unittest.main()
