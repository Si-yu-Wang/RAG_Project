"""Public chunk loading and gold-free request loading for upstream data-v1.0."""
import hashlib
import json
from pathlib import Path


FORBIDDEN = ("gold", "answer", "evidence", "extractive", "free_form", "writer_")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_records(path):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("Parquet requires pyarrow: pip install pyarrow") from exc
        return pq.read_table(path).to_pylist()
    if path.suffix.lower() != ".jsonl":
        raise ValueError("Expected .jsonl or .parquet")
    with path.open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def assert_public(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if any(part in key.lower() for part in FORBIDDEN) or key == "decision":
                raise ValueError(f"Public input contains forbidden label field: {key}")
            assert_public(item)
    elif isinstance(value, list):
        for item in value:
            assert_public(item)


def public_path(path):
    if Path(path).stem.lower().startswith("qa"):
        raise ValueError("qa* files belong to evaluation; export label-free requests first")


def load_chunks(path, split=None, manifest_path=None):
    public_path(path)
    rows = read_records(path)
    hashes = set()
    for row in rows:
        assert_public(row)
        if "config_hash" in row:
            hashes.add(row["config_hash"])
        if isinstance(row.get("para_spans"), str):
            row["para_spans"] = json.loads(row["para_spans"])
        if "para_ids" in row or "para_spans" in row:
            ids, spans = row.get("para_ids"), row.get("para_spans")
            if not isinstance(ids, list) or not all(isinstance(i, str) and i for i in ids):
                raise ValueError("para_ids must be a list of non-empty strings")
            if len(ids) != len(set(ids)) or not isinstance(spans, list) or len(ids) != len(spans):
                raise ValueError("para_ids and para_spans must match one-to-one")
            for pid, span in zip(ids, spans):
                if (not isinstance(span, list) or len(span) != 3 or span[0] != pid
                        or type(span[1]) is not int or type(span[2]) is not int
                        or not 0 <= span[1] < span[2]):
                    raise ValueError("Invalid paragraph-local span")
        if "text_sha1" in row:
            if hashlib.sha1(row["text"].encode("utf-8")).hexdigest() != row["text_sha1"]:
                raise ValueError("text_sha1 mismatch")
        if "question" in row:
            raise ValueError("Question rows cannot be indexed as chunks")
    if len(hashes) > 1 or (hashes and any(row.get("config_hash") not in hashes for row in rows)):
        raise ValueError("Mixed or missing chunk config_hash")
    explicit_manifest = manifest_path is not None
    manifest_path = Path(manifest_path) if explicit_manifest else Path(path).with_name("manifest.json")
    if explicit_manifest and not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    upstream = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if hashes and manifest.get("config_hash") not in hashes:
            raise ValueError("Upstream manifest config_hash mismatch")
        # Retain public provenance only: the upstream manifest also contains gold statistics.
        upstream = {key: manifest[key] for key in ("config_name", "config_hash", "config", "tokenizer") if key in manifest}
        upstream["manifest_sha256"] = sha256(manifest_path)
    elif manifest_path is not None and hashes:
        upstream["config_hash"] = next(iter(hashes))
    selected = [r for r in rows if split is None or r.get("split") == split]
    if not selected:
        raise ValueError("No chunks for the requested split")
    return selected, {"format": Path(path).suffix.lower()[1:], "split": split,
                      "text_field": "text", "upstream": upstream}


def load_requests(path, split=None):
    public_path(path)
    rows = read_records(path)
    result, seen = [], set()
    for row in rows:
        assert_public(row)
        if split is not None and row.get("split") != split:
            continue
        qid = row.get("question_id")
        if not isinstance(qid, str) or not qid or qid in seen:
            raise ValueError("Requests require unique non-empty question_id")
        seen.add(qid)
        for key in ("question", "paper_id"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Requests require {key}")
        result.append({k: row[k] for k in ("question_id", "paper_id", "question", "split") if k in row})
    if not result:
        raise ValueError("No requests for the requested split")
    return result
