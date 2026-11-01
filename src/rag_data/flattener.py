"""P3: flatten a paper into a paragraph table with stable ids and char offsets.

``paragraphs.parquet`` is the project's **coordinate system**: every later artefact
(chunks, gold evidence, retrieval hits) points back to a ``para_id`` here.

Offset convention: all ``char_start`` / ``char_end`` are positions in the flattened
*cleaned* paper text, built as ``SEP.join(paragraphs)`` in reading order, with the
abstract promoted to a synthetic section (``section_idx = -1``).
"""

from __future__ import annotations

from dataclasses import dataclass

from .cleaner import Cleaner, encode_segments
from .loader import Paper

SEP = "\n\n"
ABSTRACT_SECTION_IDX = -1
ABSTRACT_SECTION_NAME = "Abstract"


@dataclass
class FlattenStats:
    paragraphs: int = 0
    empty_skipped: int = 0
    chars_raw: int = 0
    chars_clean: int = 0
    changed: int = 0
    rule_hits: dict[str, int] | None = None


def flatten_paper(
    paper: Paper,
    cleaner: Cleaner,
    counter=None,
    include_abstract: bool = True,
) -> tuple[list[dict], FlattenStats, str]:
    """Return (rows, stats, flattened_text)."""
    stats = FlattenStats(rule_hits={})

    seq: list[tuple[int, str, int, bool, str]] = []
    if include_abstract and paper.abstract.strip():
        seq.append((ABSTRACT_SECTION_IDX, ABSTRACT_SECTION_NAME, 0, True, paper.abstract))
    for si, sec in enumerate(paper.sections):
        for pi, para in enumerate(sec.paragraphs):
            seq.append((si, sec.name, pi, False, para.text))

    rows: list[dict] = []
    parts: list[str] = []
    cursor = 0
    gidx = 0
    for si, sname, pi, is_abs, raw in seq:
        stats.chars_raw += len(raw)
        res = cleaner.clean(raw)
        if not res.text.strip():
            stats.empty_skipped += 1
            continue
        if res.changed:
            stats.changed += 1
            for k, v in res.hits.items():
                stats.rule_hits[k] = stats.rule_hits.get(k, 0) + v
        if parts:
            cursor += len(SEP)
        start = cursor
        end = start + len(res.text)
        parts.append(res.text)
        cursor = end

        rows.append(
            {
                "para_id": f"{paper.paper_id}#p{gidx:04d}",
                "paper_id": paper.paper_id,
                "split": paper.split,
                "section_idx": si,
                "section_name": sname,
                "para_idx": pi,
                "is_abstract": is_abs,
                "text_raw": raw,
                "text_clean": res.text,
                "char_start": start,
                "char_end": end,
                "n_chars": len(res.text),
                "n_tokens": counter.count(res.text) if counter is not None else -1,
                "offset_map": encode_segments(res.segments),
            }
        )
        gidx += 1

    stats.paragraphs = len(rows)
    stats.chars_clean = len(SEP.join(parts))
    return rows, stats, SEP.join(parts)


def section_path(paper: Paper) -> dict[int, str]:
    """Best-effort hierarchical path for Member 3's graph edges.

    QASPER only gives flat ``section_name``, so the path equals the section name unless
    the name itself encodes numbering (``3.2 Training``), which we keep verbatim.
    """
    return {si: sec.name for si, sec in enumerate(paper.sections)}
