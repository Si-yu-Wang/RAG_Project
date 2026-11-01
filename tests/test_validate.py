"""Tests for the executable contract.

Each test here pins a rule that, when violated, is *silent* in production: a leaked column
inflates recall without raising anything, a duplicated span makes evidence look twice as
plentiful, a NaN-vs-None comparison makes a healthy table look broken.
"""

from __future__ import annotations

from rag_data.validate import (
    CHUNK_REQUIRED,
    FAIL,
    FORBIDDEN_IN_PUBLIC,
    PASS,
    check_chunk_ids,
    check_columns,
    check_coverage,
    check_gold,
    check_neighbours,
    check_para_spans,
    check_token_budget,
)

PARAS = {"p#p0000": "Alpha beta. Gamma delta.", "p#p0001": "Epsilon zeta."}
LENS = {k: len(v) for k, v in PARAS.items()}


def _chunk(cid, idx, pids_spans, prev=None, nxt=None, tokens=5, text="Alpha beta.",
           pids=None):
    return {
        "chunk_id": cid,
        "paper_id": "p",
        "split": "dev",
        "section_idx": 0,
        "section_name": "S",
        "section_path": "S",
        "sec_key": "00",
        "chunk_idx_in_section": idx,
        "chunk_idx_in_paper": idx,
        "text": text,
        "n_tokens": tokens,
        "char_start": 0,
        "char_end": 10,
        "para_ids": list(pids) if pids is not None else [p for p, _, _ in pids_spans],
        "para_spans": list(pids_spans),
        "is_abstract": False,
        "is_truncated": False,
        "n_sentences": 1,
        "n_sections": 1,
        "prev_chunk_id": prev,
        "next_chunk_id": nxt,
        "text_sha1": "x",
        "config_hash": "h",
    }


def _healthy():
    rows = [
        _chunk("p#s00#c000", 0, [("p#p0000", 0, LENS["p#p0000"])]),
        _chunk("p#s00#c001", 1, [("p#p0001", 0, LENS["p#p0001"])]),
    ]
    rows[0]["next_chunk_id"] = rows[1]["chunk_id"]
    rows[1]["prev_chunk_id"] = rows[0]["chunk_id"]
    return rows


def test_clean_table_passes_everything():
    rows = _healthy()
    assert all(f.status == PASS for f in check_columns(rows, CHUNK_REQUIRED, FORBIDDEN_IN_PUBLIC))
    assert all(f.status == PASS for f in check_chunk_ids(rows))
    assert all(f.status == PASS for f in check_neighbours(rows))
    assert all(f.status == PASS for f in check_para_spans(rows, LENS))
    assert all(f.status == PASS for f in check_coverage(rows, LENS, PARAS))
    assert all(f.status == PASS for f in check_token_budget(rows, 512))


def test_leaked_answer_column_fails_loudly():
    rows = _healthy()
    rows[0]["gold_chunk_ids"] = ["p#s00#c000"]
    found = check_columns(rows, CHUNK_REQUIRED, FORBIDDEN_IN_PUBLIC)
    leak = [f for f in found if f.check == "columns.leakage"][0]
    assert leak.status == FAIL
    assert "gold_chunk_ids" in leak.detail


def test_duplicate_spans_are_rejected():
    """The real regression: spans were emitted once per sentence, not once per paragraph.

    ``para_ids`` is deduplicated by the chunker while ``para_spans`` was not, so the pairing
    is what breaks -- exactly the shape reproduced here.
    """
    rows = [
        _chunk("p#s00#c000", 0,
               [("p#p0000", 0, LENS["p#p0000"]), ("p#p0000", 0, LENS["p#p0000"]),
                ("p#p0001", 0, LENS["p#p0001"])],
               pids=["p#p0000", "p#p0001"]),
    ]
    found = {f.check: f for f in check_para_spans(rows, LENS)}
    assert found["spans.paired"].status == FAIL
    assert found["spans.unique"].status == FAIL


def test_nan_neighbour_is_treated_as_null():
    """A NULL column round-tripped through parquet becomes NaN, not None."""
    rows = _healthy()
    rows[0]["prev_chunk_id"] = float("nan")
    rows[1]["next_chunk_id"] = float("nan")
    (f,) = check_neighbours(rows)
    assert f.status == PASS


def test_broken_chain_is_caught():
    rows = _healthy()
    rows[0]["next_chunk_id"] = "p#s00#c999"   # points nowhere
    (f,) = check_neighbours(rows)
    assert f.status == FAIL


def test_content_gap_is_caught_but_whitespace_gap_is_not():
    # paragraph 0 needs 24 chars; cover only [0,6) -> real text lost
    rows = [_chunk("p#s00#c000", 0,
                   [("p#p0000", 0, 6), ("p#p0001", 0, LENS["p#p0001"])])]
    found = {f.check: f for f in check_coverage(rows, LENS, PARAS)}
    assert found["coverage.content"].status == FAIL

    # a gap that contains only whitespace is harmless
    paras = {"q#p0000": "Alpha.    Beta."}
    lens = {"q#p0000": len(paras["q#p0000"])}
    rows2 = [_chunk("q#s00#c000", 0, [("q#p0000", 0, 6), ("q#p0000", 10, 15)])]
    found2 = {f.check: f for f in check_coverage(rows2, lens, paras)}
    assert found2["coverage.content"].status == PASS


def test_gold_must_resolve_to_existing_ids():
    chunk_ids = {"p#s00#c000", "p#s00#c001"}
    para_ids = set(LENS)
    good = [{"gold_para_ids": ["p#p0000"], "gold_chunk_ids": ["p#s00#c000"]}]
    assert all(f.status == PASS for f in check_gold(good, chunk_ids, para_ids))

    bad = [{"gold_para_ids": ["p#p9999"], "gold_chunk_ids": ["p#s99#c000"]}]
    found = {f.check: f for f in check_gold(bad, chunk_ids, para_ids)}
    assert found["gold.para_refs"].status == FAIL
    assert found["gold.chunk_refs"].status == FAIL


def test_question_with_gold_but_no_chunk_is_flagged():
    found = check_gold([{"gold_para_ids": ["p#p0000"], "gold_chunk_ids": []}],
                       {"p#s00#c000"}, set(LENS))
    reach = [f for f in found if f.check == "gold.reachable"][0]
    assert reach.status == FAIL
