import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dense_rag.data import load_chunks, load_requests, public_path
from dense_rag.evaluation import export_requests, paragraph_recall
from dense_rag import Baseline, PreviewGenerator
from test_pipeline import CustomRetriever


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.row = {"chunk_id": "c1", "paper_id": "p1", "text": "clean text", "split": "dev",
                    "config_hash": "abc", "para_ids": ["p1#p0000"],
                    "para_spans": [["p1#p0000", 0, 10]], "section_path": "A > B",
                    "prev_chunk_id": None, "next_chunk_id": None,
                    "text_sha1": hashlib.sha1(b"clean text").hexdigest()}

    def write(self, name, rows):
        path = self.root / name
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding="utf-8")
        return path

    def test_parquet_jsonl_equivalence(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        path = self.root / "chunks.parquet"
        pq.write_table(pa.Table.from_pylist([{**self.row, "para_spans": json.dumps(self.row["para_spans"])}]), path)
        a, _ = load_chunks(path)
        b, _ = load_chunks(self.write("chunks.jsonl", [self.row]))
        self.assertEqual(a, b)
        self.assertEqual(a[0]["section_path"], "A > B")

    def test_manifest_mismatch_and_split(self):
        path = self.write("chunks.jsonl", [self.row])
        with self.assertRaises(ValueError):
            load_chunks(path, split="train")
        (self.root / "manifest.json").write_text(json.dumps({"config_hash": "wrong"}))
        with self.assertRaises(ValueError):
            load_chunks(path)
        (self.root / "manifest.json").write_text(json.dumps({"config_hash": "abc", "gold": {"secret": 1}}))
        _, info = load_chunks(path)
        self.assertNotIn("gold", info["upstream"])

    def test_public_labels_and_bad_spans_rejected(self):
        for extra in [{"gold_para_ids": []}, {"metadata": {"answer": "secret"}},
                      {"para_spans": [["wrong", 0, 10]]}, {"text": "changed"}]:
            with self.assertRaises(ValueError):
                load_chunks(self.write("chunks.jsonl", [{**self.row, **extra}]))

    def test_export_only_query_fields(self):
        qa = self.write("qa.jsonl", [{"question_id": "q", "paper_id": "p1", "question": "why",
                                     "split": "dev", "gold_para_ids": ["secret"], "decision": "yes"}])
        with self.assertRaises(ValueError):
            load_requests(qa)
        output = self.root / "requests.jsonl"
        export_requests(qa, output, "dev")
        rows = load_requests(output)
        self.assertEqual(set(rows[0]), {"question_id", "paper_id", "question", "split"})
        self.assertNotIn("secret", output.read_text())

    def test_macro_paragraph_recall_and_exclusions(self):
        qa = [{"question_id": "a", "paper_id": "p1", "zero_gold_reason": "", "gold_para_ids": ["a", "b"]},
              {"question_id": "b", "paper_id": "p1", "zero_gold_reason": "", "gold_para_ids": ["c"]},
              {"question_id": "c", "zero_gold_reason": "no_evidence", "gold_para_ids": []}]
        preds = [{"question_id": q, "paper_id": "p1", "config": {"top_k": 5},
                  "retrieved_chunks": [{"para_ids": [p, p]}]} for q, p in [("a", "a"), ("b", "c")]]
        result = paragraph_recall(qa, preds)
        self.assertEqual(result["macro_paragraph_recall"], .75)
        self.assertEqual(result["eligible_questions"], 2)
        self.assertIsNone(paragraph_recall(qa[2:], [], 5)["macro_paragraph_recall"])
        with self.assertRaises(KeyError):
            paragraph_recall(qa, preds[:1])

    def test_qasper_request_filename_and_label_protection(self):
        request = {"question_id": "q1", "paper_id": "p1", "question": "Why?"}
        path = self.write("qasper_first5_requests.jsonl", [request])
        self.assertEqual(load_requests(path), [request])
        for name in ("qa.parquet", "qa_dataset.jsonl", "qa_with_gold.jsonl", "QA-labels.jsonl"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                public_path(name)
        self.write(path.name, [{**request, "gold_para_ids": ["p1#p0000"]}])
        with self.assertRaises(ValueError):
            load_requests(path)

    def test_pubmed_prompt(self):
        generator = PreviewGenerator()
        with patch.object(generator, "generate", wraps=generator.generate) as mock:
            out = Baseline(CustomRetriever(), generator, task="pubmedqa").run({"question": "q", "paper_id": "p1"})
        self.assertIn("exactly one label", mock.call_args.args[0][0]["content"])
        self.assertEqual(out["config"]["task"], "pubmedqa")


if __name__ == "__main__":
    unittest.main()
