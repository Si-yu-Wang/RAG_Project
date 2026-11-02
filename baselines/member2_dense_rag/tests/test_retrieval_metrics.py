import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dense_rag.data import sha256
from dense_rag.evaluation import main
from dense_rag.retrieval_metrics import evaluate, scores


class RetrievalMetricsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.index = self.root/'index'
        self.index.mkdir()
        self.qa = self.root/'qa.jsonl'
        self.pred = self.root/'predictions.jsonl'
        self.chunks = [{'chunk_id': f'c{i}', 'paper_id': 'p', 'text': f'Text {i}', 'para_ids': [pid]}
                       for i, pid in enumerate(['x', 'a', 'b'])]
        self.write(self.index/'chunks.jsonl', self.chunks)
        self.manifest = {'count': 3, 'source_sha256': 'source',
                         'files': {'chunks.jsonl': sha256(self.index/'chunks.jsonl')}}
        (self.index/'manifest.json').write_text(json.dumps(self.manifest))
        self.questions = [
            {'question_id': 'q1', 'paper_id': 'p', 'question': 'One?', 'split': 'dev',
             'zero_gold_reason': '', 'gold_para_ids': ['a', 'b']},
            {'question_id': 'q2', 'paper_id': 'p', 'question': 'Two?', 'split': 'dev',
             'zero_gold_reason': '', 'gold_para_ids': ['a']},
            {'question_id': 'q3', 'paper_id': 'p', 'question': 'Three?', 'split': 'dev',
             'zero_gold_reason': 'no_evidence', 'gold_para_ids': []},
        ]
        self.predictions = [
            {'question_id': q['question_id'], 'paper_id': 'p', 'question': q['question'], 'split': 'dev',
             'status': 'ok', 'retrieved_chunks': [{**c, 'rank': i} for i, c in enumerate(self.chunks, 1)],
             'config': {'scope': 'paper', 'top_k': 3, 'retriever': {'corpus_sha256': 'source'}},
             'timing_ms': {'retrieval': 10}}
            for q in self.questions]
        self.save()

    def write(self, path, records):
        path.write_text(''.join(json.dumps(r)+'\n' for r in records), encoding='utf-8')

    def save(self):
        self.write(self.qa, self.questions)
        self.write(self.pred, self.predictions)

    def evaluate(self, ks=(1, 2, 3)):
        return evaluate(self.qa, self.pred, self.index, 'dev', ks)

    def test_hand_calculated_macro_recall_and_mrr(self):
        report = self.evaluate()
        self.assertEqual(report['eligible_questions'], 2)
        self.assertEqual(report['excluded_by_reason'], {'no_evidence': 1})
        self.assertEqual(report['metrics'][0]['macro_paragraph_recall'], 0)
        self.assertEqual(report['metrics'][1]['macro_paragraph_recall'], .75)
        self.assertEqual(report['metrics'][1]['mrr'], .5)
        self.assertEqual(report['metrics'][2]['macro_paragraph_recall'], 1)

    def test_paragraph_deduplication_and_first_rank(self):
        self.assertEqual(scores({'a', 'b'}, [{'a'}, {'a'}], 2), (.5, 1))
        self.assertEqual(scores({'a'}, [{'x'}, {'y'}], 2), (0, 0))

    def test_missing_extra_and_duplicate_predictions_fail(self):
        original = copy.deepcopy(self.predictions)
        for records in [original[:-1], original + [original[0]],
                        original + [{**original[0], 'question_id': 'extra'}]]:
            self.write(self.pred, records)
            with self.assertRaises(ValueError):
                self.evaluate()

    def test_tampered_evidence_and_rank_fail(self):
        original = copy.deepcopy(self.predictions)
        for change in [{'para_ids': ['fake']}, {'text': 'wrong'}, {'rank': 2}, {'paper_id': 'other'}]:
            self.predictions = copy.deepcopy(original)
            self.predictions[0]['retrieved_chunks'][0].update(change)
            self.save()
            with self.assertRaises(ValueError):
                self.evaluate()

    def test_invalid_ks_and_insufficient_depth_fail(self):
        for ks in [[], [0], [True], [4]]:
            with self.assertRaises(ValueError):
                self.evaluate(ks)

    def test_no_gold_returns_null_not_zero(self):
        for q in self.questions:
            q.update(zero_gold_reason='no_evidence', gold_para_ids=[])
        self.save()
        report = self.evaluate()
        self.assertEqual(report['eligible_questions'], 0)
        self.assertTrue(all(m['mrr'] is None and m['macro_paragraph_recall'] is None for m in report['metrics']))

    def test_mixed_config_and_corrupt_corpus_fail(self):
        self.predictions[0]['config']['top_k'] = 2
        self.save()
        with self.assertRaisesRegex(ValueError, 'Mixed'):
            self.evaluate()
        with (self.index/'chunks.jsonl').open('a') as f:
            f.write('\n')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.evaluate()

    def test_score_cli_creates_reports_without_overwrite(self):
        output, report = self.root/'metrics.json', self.root/'report.md'
        args = ['score', '--qa', str(self.qa), '--predictions', str(self.pred), '--index', str(self.index),
                '--ks', '1', '2', '3', '--output', str(output), '--report', str(report)]
        with patch('builtins.print'):
            main(args)
        self.assertEqual(json.loads(output.read_text())['eligible_questions'], 2)
        self.assertIn('75.00%', report.read_text(encoding='utf-8'))
        with self.assertRaises(SystemExit):
            main(args)

    def test_legacy_export_cli_remains_supported(self):
        output = self.root/'requests.jsonl'
        with patch('builtins.print'):
            main(['--qa', str(self.qa), '--output', str(output), '--split', 'dev'])
        exported = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(len(exported), 3)
        self.assertTrue(all('gold_para_ids' not in q for q in exported))
