import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from dataclasses import replace

from dense_rag.core import (EncoderConfig, DenseRetriever, Baseline, PreviewGenerator,
                            HTTPGenerator, validate_chunks, unit_vectors, write_jsonl)


class FixedEncoder:
    config = EncoderConfig(model="test-vectors", revision="test-only")

    def encode(self, texts, query=False):
        return [[1, 0] if "alpha" in t else [0, 1] for t in texts]


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.source = root / "input.jsonl"
        self.folder = root / "index"
        self.chunks = [
            {"chunk_id": "a", "paper_id": "p1", "text": "alpha", "metadata": {"paragraph_index": 0}},
            {"chunk_id": "b", "paper_id": "p2", "text": "beta"},
            {"chunk_id": "c", "paper_id": "p1", "text": "beta"},
        ]
        write_jsonl(self.source, self.chunks)
        self.r = DenseRetriever.build(self.source, self.folder, FixedEncoder())

    def test_cosine_ranking_and_roundtrip(self):
        r = DenseRetriever.load(self.folder, FixedEncoder())
        hits = r.retrieve("alpha", 9)
        self.assertEqual(len(hits), 3)
        self.assertEqual(hits[0]["chunk_id"], "a")
        self.assertAlmostEqual(hits[0]["score"], 1)
        self.assertEqual(hits[0]["metadata"]["paragraph_index"], 0)

    def test_filter_before_top_k(self):
        hits = self.r.retrieve("alpha", 1, "p2")
        self.assertEqual(hits[0]["chunk_id"], "b")
        self.assertEqual(self.r.retrieve("alpha", 1, "missing"), [])

    def test_invalid_inputs(self):
        for q, k in [("", 1), ("a", 0), ("a", True)]:
            with self.assertRaises(ValueError):
                self.r.retrieve(q, k)
        for rows in [[], self.chunks + [self.chunks[0]], [{"chunk_id": "x"}]]:
            with self.assertRaises(ValueError):
                validate_chunks(rows)
        with self.assertRaises(ValueError):
            unit_vectors([[0, 0]])

    def test_corruption_detected(self):
        with (self.folder / "chunks.jsonl").open("a") as f:
            f.write("\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            DenseRetriever.load(self.folder, FixedEncoder())

    def test_encoder_mismatch(self):
        enc = FixedEncoder()
        enc.config = replace(enc.config, query_instruction="different")
        with self.assertRaisesRegex(ValueError, "configuration"):
            DenseRetriever.load(self.folder, enc)

    def test_context_budget_and_no_gold_leak(self):
        gen = PreviewGenerator()
        with patch.object(gen, "generate", wraps=gen.generate) as mocked:
            result = Baseline(self.r, gen, context_chars=2).run(
                {"question": "alpha", "paper_id": "p1", "gold_answer": "SECRET_GOLD"})
        self.assertEqual(result["contexts"][0]["text"], "al")
        self.assertTrue(result["contexts"][0]["truncated"])
        self.assertEqual(result["context_chunk_ids"], ["a"])
        self.assertNotIn("SECRET_GOLD", json.dumps(mocked.call_args.args))
        self.assertGreaterEqual(result["timing_ms"]["total"], result["timing_ms"]["retrieval"])

    def test_no_evidence_and_missing_scope(self):
        gen = PreviewGenerator()
        baseline = Baseline(self.r, gen)
        with patch.object(gen, "generate") as mocked:
            out = baseline.run({"question": "alpha", "paper_id": "missing"})
        mocked.assert_not_called()
        self.assertEqual(out["status"], "no_evidence")
        with self.assertRaises(ValueError):
            baseline.run({"question": "alpha"})

    def test_http_payload_and_parse(self):
        from io import BytesIO
        response = BytesIO(json.dumps({"choices": [{"message": {"content": "answer [a]"}}],
                                      "usage": {"total_tokens": 10}}).encode())
        with patch("urllib.request.urlopen", return_value=response) as mocked:
            result = HTTPGenerator("http://localhost:8000/v1/chat/completions", "team-model").generate(
                [{"role": "user", "content": "question"}])
        self.assertEqual(result["answer"], "answer [a]")
        payload = json.loads(mocked.call_args.args[0].data)
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["model"], "team-model")


if __name__ == "__main__":
    unittest.main()
