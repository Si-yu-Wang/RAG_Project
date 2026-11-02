import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from dense_rag.cli import main


class CLITests(unittest.TestCase):
    def test_bad_requests_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('sys.argv', ['dense-rag', 'run', '--index', 'unused',
                                   '--questions', 'qa_with_gold.jsonl', '--output', str(Path(tmp)/'out.jsonl')]), \
                    patch('dense_rag.cli.DenseRetriever.load') as load:
                with self.assertRaises(ValueError):
                    main()
                load.assert_not_called()

    def test_existing_output_preserved_before_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'result.jsonl'
            output.write_text('existing result')
            with patch('sys.argv', ['dense-rag', 'run', '--index', 'unused',
                                   '--questions', 'unused', '--output', str(output)]), \
                    patch('dense_rag.cli.DenseRetriever.load') as load:
                with self.assertRaises(SystemExit):
                    main()
                load.assert_not_called()
            self.assertEqual(output.read_text(), 'existing result')
