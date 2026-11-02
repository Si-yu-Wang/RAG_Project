"""Generic contract tests without dense dependencies."""
import copy
import unittest
from dense_rag import Baseline, PreviewGenerator


class CustomRetriever:
    def __init__(self):
        self.config = {"method": "graph_stub", "parameters": {"max_hops": 2}}
        self.hits = [{"chunk_id": "c1", "paper_id": "p1", "text": "Evidence",
                      "score": 2.0, "rank": 1, "metadata": {"path": ["a", "b"]}}]
        self.calls = []

    def retrieve(self, question, top_k=5, paper_id=None):
        self.calls.append((question, top_k, paper_id))
        return self.hits


class PipelineTests(unittest.TestCase):
    def test_backend_without_manifest_or_encoder(self):
        retriever = CustomRetriever()
        out = Baseline(retriever, PreviewGenerator()).run({"question": "q", "paper_id": "p1"})
        self.assertEqual(out["schema_version"], "2.0")
        self.assertEqual(out["config"]["retriever"]["method"], "graph_stub")
        self.assertEqual(out["retrieved_chunks"][0]["metadata"]["path"], ["a", "b"])
        retriever.config["parameters"]["max_hops"] = 9
        retriever.hits[0]["text"] = "mutated"
        self.assertEqual(out["config"]["retriever"]["parameters"]["max_hops"], 2)
        self.assertEqual(out["retrieved_chunks"][0]["text"], "Evidence")

    def test_global_passes_no_paper_filter(self):
        retriever = CustomRetriever()
        Baseline(retriever, PreviewGenerator(), scope="global").run({"question": "q", "paper_id": "other"})
        self.assertEqual(retriever.calls, [("q", 5, None)])

    def test_bad_backend_results_rejected(self):
        retriever = CustomRetriever()
        original = copy.deepcopy(retriever.hits)
        for change in [{"paper_id": "other"}, {"rank": 0}, {"score": float("nan")},
                       {"text": ""}, {"metadata": []}]:
            with self.subTest(change=change):
                retriever.hits = [{**original[0], **change}]
                with self.assertRaises(ValueError):
                    Baseline(retriever, PreviewGenerator()).run({"question": "q", "paper_id": "p1"})
        for hits in [original * 2, tuple(original), original * 6]:
            retriever.hits = hits
            with self.assertRaises(ValueError):
                Baseline(retriever, PreviewGenerator()).run({"question": "q", "paper_id": "p1"})

    def test_generic_input_validation(self):
        for kwargs in [{"top_k": 0}, {"top_k": True}, {"context_chars": 1.2}, {"scope": "unknown"}]:
            with self.assertRaises(ValueError):
                Baseline(CustomRetriever(), PreviewGenerator(), **kwargs)
        for request in [{}, {"question": " "}, []]:
            with self.assertRaises(ValueError):
                Baseline(CustomRetriever(), PreviewGenerator()).run(request)

    def test_non_json_config_rejected(self):
        retriever = CustomRetriever()
        retriever.config = {"graph": object()}
        with self.assertRaisesRegex(ValueError, "JSON"):
            Baseline(retriever, PreviewGenerator())


if __name__ == "__main__":
    unittest.main()
