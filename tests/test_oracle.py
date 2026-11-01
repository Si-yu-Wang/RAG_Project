"""Semantics of the chunking-imposed recall ceiling.

These tests exist because two plausible-looking implementations were both wrong in ways that
silently flipped the conclusion about overlap:

* counting "paragraph appears in >1 chunk" as "paragraph is torn" (duplication != tearing);
* reading coverage only at the exact budget k, ignoring cheaper budgets already achieved.
"""

from rag_data.oracle import (
    evidence_recall_ceiling,
    paragraph_status,
)


def _run(gold_paras, spans, lens, ks=(1, 3)):
    recall, fits, n = evidence_recall_ceiling(gold_paras, spans, lens, ks)
    return recall, fits, n


def test_single_paragraph_inside_one_chunk_is_perfect():
    spans = {"p1": {"c1": (0, 100)}}
    recall, fits, n = _run([["p1"]], spans, {"p1": 100})
    assert n == 1
    assert recall["oracle_recall@1"] == 100.0
    assert fits[1] == 1


def test_torn_paragraph_needs_two_chunks():
    # paragraph 100 chars long, split as [0,60) and [60,100)
    spans = {"p1": {"c1": (0, 60), "c2": (60, 100)}}
    recall, fits, _ = _run([["p1"]], spans, {"p1": 100})
    assert recall["oracle_recall@1"] == 0.0   # one chunk can never show all of it
    assert recall["oracle_recall@3"] == 100.0  # two do
    assert fits[1] == 0 and fits[3] == 1


def test_duplicated_paragraph_is_not_torn():
    # overlap re-emits the whole paragraph in c2; it is duplicated but fully readable
    spans = {"p1": {"c1": (0, 100), "c2": (0, 100)}}
    torn, dup = paragraph_status(spans["p1"], 100)
    assert dup is True
    assert torn is False
    recall, _, _ = _run([["p1"]], spans, {"p1": 100})
    assert recall["oracle_recall@1"] == 100.0


def test_two_gold_paragraphs_in_different_chunks():
    spans = {"p1": {"c1": (0, 50)}, "p2": {"c2": (0, 50)}}
    recall, _, _ = _run([["p1", "p2"]], spans, {"p1": 50, "p2": 50})
    assert recall["oracle_recall@1"] == 50.0   # one chunk recovers half the evidence
    assert recall["oracle_recall@3"] == 100.0


def test_two_gold_paragraphs_sharing_a_chunk():
    spans = {"p1": {"c1": (0, 50)}, "p2": {"c1": (0, 50)}}
    recall, _, _ = _run([["p1", "p2"]], spans, {"p1": 50, "p2": 50})
    assert recall["oracle_recall@1"] == 100.0


def test_budget_carries_forward_across_k():
    # Both paragraphs live in c2/c3 respectively; coverage reached at step 2 must still be
    # reported at k=3 and k=5 rather than resetting to the value at the exact step.
    spans = {"p1": {"c2": (0, 50)}, "p2": {"c3": (0, 50)}}
    recall, fits, _ = _run([["p1", "p2"]], spans, {"p1": 50, "p2": 50}, ks=(1, 3, 5))
    assert recall["oracle_recall@1"] == 50.0
    assert recall["oracle_recall@3"] == 100.0
    assert recall["oracle_recall@5"] == 100.0
    assert fits[3] == 1


def test_missing_paragraph_is_skipped_not_zeroed():
    spans = {"p1": {"c1": (0, 50)}}
    recall, _, n = _run([["p1", "ghost"]], spans, {"p1": 50})
    assert n == 1
    assert recall["oracle_recall@1"] == 100.0


def test_paragraph_status_detects_true_tear():
    torn, dup = paragraph_status({"c1": (0, 40), "c2": (40, 90)}, 90)
    assert torn is True and dup is True
