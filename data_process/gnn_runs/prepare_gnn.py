"""Apply startup/cache/evaluation fixes to the pinned GNN-RAG source.

No packages, models, or datasets are downloaded. No training is started.
The original files are preserved as *.webqsp-original; unknown source is refused.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

COMMIT = "28d31bc0db3d54f0930e6125f1c98b5b1ace1ba4"
EXPECTED = {
    "parsing.py": "0cfb3426e6bbb929ff7bbfb6720ed232746d157d2e6bc8f28c810c4a95aff539",
    "train_model.py": "1faa7a95d4203b14d13dc5b30021c87f91f65c9951fe7472eb3bb33959d0a88f",
    "modules/question_encoding/bert_encoder.py": "fa5c0f29f9cfc2590dc7a7821e2132ac70a2ffe11ae4e98b4b14b22bf46821be",
}


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one patch target: {old!r}")
    return text.replace(old, new, 1)


def patched(rel, text):
    if rel == "parsing.py":
        text = replace_once(text, '    parser_nutrea = subparsers.add_parser("NuTrea")\n'
                            '    create_parser_nutrea(parser_nutrea)',
                            '    # WebQSP: omit unavailable NuTrea parser.')
    elif rel == "train_model.py":
        text = replace_once(text,
                            '                eval_f1, eval_h1, eval_em = self.evaluate(self.test_data, self.test_batch_size)\n'
                            '                self.logger.info("TEST F1: {:.4f}, H1: {:.4f}, EM {:.4f}".format(eval_f1, eval_h1, eval_em))',
                            '                # WebQSP: validation only during training.')
        text = replace_once(text, "        self.logger.info('Train Done! Evaluate on testset with saved model')",
                            "        self.logger.info('Train done; run explicit evaluation of the dev-selected checkpoint.')")
        text = replace_once(text, "        self.evaluate_best()",
                            "        # WebQSP: do not automatically compare checkpoints on test.")
    else:
        text = replace_once(text,
                            "os.environ['TRANSFORMERS_CACHE'] = '/export/scratch/costas/home/mavro016/.cache'",
                            "# WebQSP: respect the caller's HF_HOME/cache configuration.")
    ast.parse(text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gnn_dir", type=Path, help="GNN-RAG/gnn directory")
    args = ap.parse_args()
    changes, report = [], {"upstream_commit": COMMIT, "files": {}}
    # Validate ALL inputs before writing any source file.
    for rel, expected in EXPECTED.items():
        path = args.gnn_dir / rel
        backup = path.with_name(path.name + ".webqsp-original")
        current = path.read_text(encoding="utf-8")
        original = backup.read_text(encoding="utf-8") if backup.exists() else current
        if digest(original) != expected:
            raise ValueError(f"Unrecognized upstream file: {path}; expected commit {COMMIT}")
        result = patched(rel, original)
        if current not in (original, result):
            raise ValueError(f"Locally modified file; refusing to overwrite: {path}")
        changes.append((path, backup, original, result))
        report["files"][rel] = {"original_sha256_lf": expected, "patched_sha256_lf": digest(result)}
    for path, backup, original, result in changes:
        if not backup.exists():
            backup.write_text(original, encoding="utf-8", newline="\n")
        path.write_text(result, encoding="utf-8", newline="\n")
    (args.gnn_dir / "webqsp_patch_record.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("OK: startup fixed; cache override removed; training uses dev evaluation only.")
    print("The upstream loader still loads test data; no test metrics are computed during training.")
    print("This is source preparation, not a successful GPU training test.")


if __name__ == "__main__":
    main()
