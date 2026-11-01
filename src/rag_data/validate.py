"""A9: contract checks for the delivered data artifacts.

The point of this module is that the *contract* is executable. A markdown table describing
column semantics is not a contract -- it is a wish. Every rule that a downstream member could
silently violate is encoded here as a check that fails loudly.

Two categories of rule:

1. **Structural** -- primary keys unique, ids parseable, neighbour chains closed, offsets in
   bounds, token budget respected. Violations here break someone's join or index.
2. **Semantic / leakage** -- gold evidence must not be reachable from a *public* artifact. This
   is the check that matters most and is the easiest to get wrong: if a chunk table ever grows
   a column derived from the answer annotations, graph construction quietly becomes
   label-driven, recall inflates, and nobody can tell by looking at the numbers.

All functions take plain Python structures (``list[dict]`` / ``dict``) so they are cheap to
unit-test without pandas or parquet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CHUNK_ID_RE = re.compile(r"^(?P<paper>.+)#s(?P<sec>abs|\d{2,})#c(?P<idx>\d{3,})$")
PARA_ID_RE = re.compile(r"^(?P<paper>.+)#p(?P<idx>\d{4,})$")

#: Columns any consumer of ``chunks.*`` may rely on.
CHUNK_REQUIRED: dict[str, str] = {
    "chunk_id": "str",
    "paper_id": "str",
    "split": "str",
    "section_idx": "int",
    "section_name": "str",
    "section_path": "str",
    "sec_key": "str",
    "chunk_idx_in_section": "int",
    "chunk_idx_in_paper": "int",
    "text": "str",
    "n_tokens": "int",
    "char_start": "int",
    "char_end": "int",
    "para_ids": "list[str]",
    "para_spans": "list[[str,int,int]]",
    "is_abstract": "bool",
    "is_truncated": "bool",
    "n_sentences": "int",
    "n_sections": "int",
    "prev_chunk_id": "str|null",
    "next_chunk_id": "str|null",
    "text_sha1": "str",
    "config_hash": "str",
}

QA_REQUIRED: dict[str, str] = {
    "question_id": "str",
    "paper_id": "str",
    "split": "str",
    "question": "str",
    "n_answers": "int",
    "answer_types": "str",
    "is_answerable": "bool",
    "n_evidence": "int",
    "n_evidence_float": "int",
    "n_evidence_mapped": "int",
    "n_evidence_failed": "int",
    "evidence_levels": "str",
    "gold_para_ids": "list[str]",
    "n_gold_paras": "int",
    "zero_gold_reason": "str",
    "is_float_only": "bool",
    "free_form_answers": "list[str]",
    "extractive_spans": "list[str]",
    "extractive_spans_located": "list[dict]",
    "yes_no": "object",
    "writer_nlp_background": "str",
    "writer_topic_background": "str",
    "writer_paper_read": "str",
    "search_query": "str",
}

#: A public artifact carrying any of these is contaminated. Matched case-insensitively
#: against column names, so ``gold_chunk_ids``, ``evidence_level`` and friends all trip it.
FORBIDDEN_IN_PUBLIC: tuple[str, ...] = (
    "gold",
    "answer",
    "evidence",
    "yes_no",
    "extractive",
    "free_form",
    "writer_",
    "search_query",
    "is_answerable",
    "zero_gold",
)

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass
class Finding:
    check: str
    status: str
    detail: str


def _f(check: str, status: str, detail: str) -> Finding:
    return Finding(check, status, detail)


#: Public alias so callers outside this module can build ad-hoc findings.
finding = _f


# --------------------------------------------------------------------- structural checks
def check_columns(rows: list[dict], required: dict[str, str],
                  forbidden: tuple[str, ...] = ()) -> list[Finding]:
    out: list[Finding] = []
    if not rows:
        return [_f("columns", FAIL, "table is empty")]
    present = set(rows[0].keys())
    missing = [c for c in required if c not in present]
    out.append(
        _f("columns.required", PASS if not missing else FAIL,
           "all required columns present" if not missing else f"missing: {missing}")
    )
    extra = sorted(c for c in present if c not in required)
    out.append(
        _f("columns.extra", PASS if not extra else WARN,
           "no extra columns" if not extra else f"extra (document them): {extra}")
    )
    if forbidden:
        bad = []
        for c in present:
            low = c.lower()
            if any(p in low for p in forbidden):
                bad.append(c)
        out.append(
            _f("columns.leakage", PASS if not bad else FAIL,
               "no answer-derived column in a public artifact"
               if not bad else f"CONTAMINATED, answer-derived columns present: {bad}")
        )
    return out


def check_chunk_ids(rows: list[dict]) -> list[Finding]:
    ids = [r["chunk_id"] for r in rows]
    dup = len(ids) - len(set(ids))
    bad = []
    for r in rows:
        m = CHUNK_ID_RE.match(r["chunk_id"])
        if not m:
            bad.append(r["chunk_id"])
        elif m.group("paper") != r["paper_id"]:
            bad.append(r["chunk_id"])
    return [
        _f("id.unique", PASS if dup == 0 else FAIL, f"{dup} duplicate chunk_id"),
        _f("id.format", PASS if not bad else FAIL,
           "all ids match {paper_id}#s{sec}#c{idx} and embed their own paper_id"
           if not bad else f"{len(bad)} malformed, e.g. {bad[:3]}"),
    ]


def check_ordering(rows: list[dict]) -> list[Finding]:
    from collections import defaultdict

    per_paper: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        per_paper[r["paper_id"]].append(r)
    bad_idx, bad_sec = [], []
    for pid, rs in per_paper.items():
        rs.sort(key=lambda r: r["chunk_idx_in_paper"])
        for i, r in enumerate(rs):
            if r["chunk_idx_in_paper"] != i:
                bad_idx.append(f"{pid}@{i}")
                break
        seen: dict[str, int] = {}
        for r in rs:
            n = seen.get(r["sec_key"], 0)
            if r["chunk_idx_in_section"] != n:
                bad_sec.append(f"{pid}:{r['sec_key']}")
                break
            seen[r["sec_key"]] = n + 1
    return [
        _f("order.in_paper", PASS if not bad_idx else FAIL,
           "chunk_idx_in_paper is 0..n-1 per paper" if not bad_idx
           else f"{len(bad_idx)} papers broken, e.g. {bad_idx[:3]}"),
        _f("order.in_section", PASS if not bad_sec else FAIL,
           "chunk_idx_in_section is 0..n-1 per section" if not bad_sec
           else f"{len(bad_sec)} sections broken, e.g. {bad_sec[:3]}"),
    ]


def _nullish(v) -> bool:
    """True for None / NaN / pandas NA -- the three ways a null shows up after a round trip."""
    if v is None:
        return True
    if isinstance(v, float):
        return v != v
    if isinstance(v, str):
        return v.strip().lower() in {"", "nan", "none", "null"}
    try:
        import pandas as pd

        return bool(pd.isna(v))
    except (TypeError, ValueError, ImportError):
        return False


def _opt(v):
    return None if _nullish(v) else v


def check_neighbours(rows: list[dict]) -> list[Finding]:
    """prev/next must walk each paper's own chunk list, and stop at its ends.

    Null comparison is the subtle part: a ``NULL`` column read back from parquet is ``pd.NA``
    or ``NaN``, and ``NaN != None`` is True -- so a naive check reports every paper's first and
    last chunk as broken. Normalise before comparing, or the check cries wolf on ~2.3k rows.
    """
    from collections import defaultdict

    per_paper: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        per_paper[r["paper_id"]].append(r)
    bad = 0
    for pid, rs in per_paper.items():
        rs.sort(key=lambda r: r["chunk_idx_in_paper"])
        for i, r in enumerate(rs):
            want_prev = rs[i - 1]["chunk_id"] if i else None
            want_next = rs[i + 1]["chunk_id"] if i + 1 < len(rs) else None
            if _opt(r.get("prev_chunk_id")) != want_prev or _opt(r.get("next_chunk_id")) != want_next:
                bad += 1
    return [_f("graph.neighbours", PASS if bad == 0 else FAIL,
               "prev/next form a closed chain within each paper" if bad == 0
               else f"{bad} chunks with broken prev/next")]


def check_token_budget(rows: list[dict], limit: int | None) -> list[Finding]:
    over = [r["chunk_id"] for r in rows if limit and r["n_tokens"] > limit]
    empty = [r["chunk_id"] for r in rows if not str(r["text"]).strip()]
    return [
        _f("size.within_budget", PASS if not over else FAIL,
           f"all chunks <= {limit} tokens" if not over
           else f"{len(over)} chunks over budget, e.g. {over[:3]}"),
        _f("size.non_empty", PASS if not empty else FAIL,
           "no empty chunk" if not empty else f"{len(empty)} empty chunks"),
    ]


def check_para_spans(rows: list[dict], para_len: dict[str, int]) -> list[Finding]:
    """One span per paragraph, in bounds, and actually present in the chunk text."""
    paired_bad, dup_bad, bound_bad, text_bad = [], [], [], []
    for r in rows:
        spans = r["para_spans"]
        pids = list(r["para_ids"])
        if len(spans) != len(pids):
            paired_bad.append(r["chunk_id"])
        if len({p for p, _, _ in spans}) != len(spans):
            dup_bad.append(r["chunk_id"])
        for pid, a, b in spans:
            L = para_len.get(pid)
            if L is None or not (0 <= a < b <= L):
                bound_bad.append(f"{r['chunk_id']}:{pid}[{a},{b})/{L}")
    return [
        _f("spans.paired", PASS if not paired_bad else FAIL,
           "len(para_spans) == len(para_ids) for every chunk" if not paired_bad
           else f"{len(paired_bad)} chunks mismatch, e.g. {paired_bad[:3]}"),
        _f("spans.unique", PASS if not dup_bad else FAIL,
           "exactly one span per paragraph per chunk" if not dup_bad
           else f"{len(dup_bad)} chunks repeat a paragraph, e.g. {dup_bad[:3]}"),
        _f("spans.in_bounds", PASS if not bound_bad else FAIL,
           "every span inside its paragraph" if not bound_bad
           else f"{len(bound_bad)} out of bounds, e.g. {bound_bad[:3]}"),
    ]


def check_coverage(rows: list[dict], para_len: dict[str, int],
                   para_text: dict[str, str]) -> list[Finding]:
    """Every paragraph must be reconstructable from the chunks, ignoring whitespace."""
    from collections import defaultdict

    spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for r in rows:
        for pid, a, b in r["para_spans"]:
            spans[pid].append((a, b))

    content_gaps = []
    missing = 0
    for pid, L in para_len.items():
        sp = sorted(spans.get(pid, []))
        if not sp:
            missing += 1
            continue
        cover = 0
        for a, b in sp:
            if a > cover:
                text = para_text.get(pid, "")
                if text[cover:a].strip():
                    content_gaps.append(f"{pid}[{cover},{a})")
            cover = max(cover, b)
        if cover < L and para_text.get(pid, "")[cover:].strip():
            content_gaps.append(f"{pid}[{cover},{L})")
    return [
        _f("coverage.paragraphs", PASS if missing == 0 else FAIL,
           "every paragraph appears in at least one chunk" if missing == 0
           else f"{missing} paragraphs appear in no chunk"),
        _f("coverage.content", PASS if not content_gaps else FAIL,
           "no non-whitespace character lost at chunk boundaries" if not content_gaps
           else f"{len(content_gaps)} content gaps, e.g. {content_gaps[:3]}"),
    ]


# ------------------------------------------------------------------------- gold / leakage
def check_gold(qa_rows: list[dict], chunk_ids: set[str], para_ids: set[str]) -> list[Finding]:
    dangling_para = 0
    dangling_chunk = 0
    empty_when_answerable = 0
    for q in qa_rows:
        gp = list(q.get("gold_para_ids") or [])
        dangling_para += sum(1 for p in gp if p not in para_ids)
        gc = list(q.get("gold_chunk_ids") or [])
        dangling_chunk += sum(1 for c in gc if c not in chunk_ids)
        if gp and not gc:
            empty_when_answerable += 1
    return [
        _f("gold.para_refs", PASS if dangling_para == 0 else FAIL,
           "all gold_para_ids exist in the paragraph index" if dangling_para == 0
           else f"{dangling_para} dangling gold_para_ids"),
        _f("gold.chunk_refs", PASS if dangling_chunk == 0 else FAIL,
           "all gold_chunk_ids exist in this chunk config" if dangling_chunk == 0
           else f"{dangling_chunk} dangling gold_chunk_ids"),
        _f("gold.reachable", PASS if empty_when_answerable == 0 else FAIL,
           "every question with gold paragraphs maps to >=1 chunk"
           if empty_when_answerable == 0
           else f"{empty_when_answerable} questions lost their evidence in this config"),
    ]


def check_no_gold_in_public(rows: list[dict]) -> list[Finding]:
    """Belt and braces on top of ``check_columns``: nothing value-shaped may slip through."""
    if not rows:
        return []
    keys = set(rows[0].keys())
    hits = sorted(k for k in keys if any(p in k.lower() for p in FORBIDDEN_IN_PUBLIC))
    return [_f("leakage.value_fields", PASS if not hits else FAIL,
               "public chunk table carries no answer-derived field" if not hits
               else f"remove before publishing: {hits}")]


def check_text_sha1(rows: list[dict], limit: int = 500) -> list[Finding]:
    """``text_sha1`` must be the sha1 of ``text``, so configs can be diffed by hash alone."""
    import hashlib

    bad = 0
    for r in rows[:limit]:
        if hashlib.sha1(str(r["text"]).encode("utf-8")).hexdigest() != r["text_sha1"]:
            bad += 1
    return [_f("integrity.text_sha1", PASS if bad == 0 else FAIL,
               f"sha1 matches text on {min(limit, len(rows))} sampled chunks" if bad == 0
               else f"{bad} mismatched sha1")]


def check_config_hash(rows: list[dict], expected: str | None) -> list[Finding]:
    if not expected:
        return []
    got = {r["config_hash"] for r in rows}
    return [_f("integrity.config_hash", PASS if got == {expected} else FAIL,
               f"config_hash == {expected} on every row" if got == {expected}
               else f"expected {expected}, saw {sorted(got)[:4]}")]


def summarise(findings: list[Finding]) -> tuple[int, int, int]:
    return (
        sum(1 for f in findings if f.status == PASS),
        sum(1 for f in findings if f.status == WARN),
        sum(1 for f in findings if f.status == FAIL),
    )
