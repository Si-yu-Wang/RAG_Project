"""Model-free ceiling on evidence recall imposed purely by chunking.

Why this exists: every downstream method (dense, graph, fusion, rerank) is scored with
recall@k, and a chunking that scatters a question's gold evidence across six chunks caps
that recall no matter how good the ranker is. Measuring the ceiling separately keeps the
attribution honest -- "our retriever is weak" and "our chunking threw the answer away" are
completely different problems with completely different owners.

Two definitions matter and must never be conflated:

* **torn** -- no single chunk contains the paragraph in full, so the reader cannot see all of
  it no matter which one chunk is retrieved;
* **duplicated** -- the paragraph appears in two chunks because of overlap. That is the
  *design* of overlap, and a chunk holding the whole paragraph still satisfies coverage.

Conflating them makes overlap look uniformly harmful, which is the opposite of the truth:
overlap mostly *repairs* boundary tearing, because it pushes the next chunk's start back
before the paragraph begins.

Coverage of a paragraph may legitimately require two chunks when it is torn. The search below
is a greedy over a lexicographic key ``(newly fully covered paragraphs, newly covered chars)``;
the second term is what lets it assemble a torn paragraph out of two chunks. Greedy is exact
for the cases that dominate (1-2 gold paragraphs); otherwise it is a lower bound, so the
reported ceiling is never optimistic.
"""

from __future__ import annotations

DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)


def covers(intervals: list[tuple[int, int]], total: int) -> bool:
    """True when the union of ``intervals`` covers ``[0, total)`` with no hole."""
    cover = 0
    for a, b in sorted(intervals):
        if a > cover:
            return False
        cover = max(cover, b)
        if cover >= total:
            return True
    return cover >= total


def char_covered(intervals: list[tuple[int, int]], total: int) -> int:
    """Number of characters of ``[0, total)`` covered by ``intervals``."""
    cover = 0
    got = 0
    for a, b in sorted(intervals):
        if b <= cover:
            continue
        a2 = max(a, cover)
        got += b - a2
        cover = b
    return min(got, total)


def paragraph_status(spans: dict[str, tuple[int, int]], length: int) -> tuple[bool, bool]:
    """(torn, duplicated) for one paragraph, given ``{chunk_id: (start, end)}`` spans."""
    dup = len(spans) > 1
    whole = any(a == 0 and b >= length for (a, b) in spans.values())
    return (not whole, dup)


def evidence_recall_ceiling(
    gold_paras: list[list[str]],
    spans_by_para: dict[str, dict[str, tuple[int, int]]],
    para_len: dict[str, int],
    ks: tuple[int, ...] = DEFAULT_KS,
) -> tuple[dict[str, float], dict[str, int], int]:
    """Best achievable evidence recall@k for each question, averaged.

    Args:
        gold_paras: one entry per question, listing that question's gold paragraph ids.
        spans_by_para: ``para_id -> {chunk_id: (char_start, char_end)}`` for the chunking
            under test. Offsets are relative to the paragraph's own text.
        para_len: ``para_id -> length`` of the paragraph text.
        ks: cut-offs to report.

    Returns:
        ``(recall_pct, fits_counts, n_questions)`` where ``recall_pct[f"oracle_recall@{k}"]``
        is a percentage and ``fits_counts[k]`` counts questions whose entire gold evidence is
        readable within k chunks.
    """
    ks = tuple(sorted(ks))
    kmax = max(ks)
    acc = {k: 0.0 for k in ks}
    fits = {k: 0 for k in ks}
    n = 0

    for paras in gold_paras:
        pset = [p for p in paras if p in spans_by_para]
        if not pset:
            continue
        cands = {cid for p in pset for cid in spans_by_para[p]}
        selected: set[str] = set()
        covered: set[str] = set()
        step_covered: dict[int, int] = {}

        for step in range(1, kmax + 1):
            best_c, best_key = None, (0, 0)
            # ``sorted`` is not cosmetic: string hashing is randomised per process, so
            # iterating the raw set would make equal-key ties resolve differently between
            # runs and the ceiling would drift by ~0.2pp. Determinism matters here because
            # these numbers end up in a report.
            for c in sorted(cands - selected):
                newsel = selected | {c}
                newly = 0
                for p in pset:
                    if p in covered:
                        continue
                    iv = [spans_by_para[p][x] for x in newsel if x in spans_by_para[p]]
                    if iv and covers(iv, para_len[p]):
                        newly += 1
                if newly:
                    key = (newly, 1 << 30)  # completing a paragraph dominates
                else:
                    prog = 0
                    for p in pset:
                        if p in covered:
                            continue
                        old = [spans_by_para[p][x] for x in selected if x in spans_by_para[p]]
                        new = [spans_by_para[p][x] for x in newsel if x in spans_by_para[p]]
                        prog += char_covered(new, para_len[p]) - char_covered(old, para_len[p])
                    key = (0, prog)
                if key > best_key:
                    best_key, best_c = key, c
            if best_c is None:
                break
            selected.add(best_c)
            for p in pset:
                if p in covered:
                    continue
                iv = [spans_by_para[p][x] for x in selected if x in spans_by_para[p]]
                if iv and covers(iv, para_len[p]):
                    covered.add(p)
            step_covered[step] = len(covered)
            if len(covered) == len(pset):
                break

        # Coverage at budget k is the best over *all* budgets up to k, not the value that
        # happens to be recorded at step k exactly (early break leaves later steps empty).
        run = 0
        for k in ks:
            for s in range(1, k + 1):
                run = max(run, step_covered.get(s, 0))
            acc[k] += run / len(pset)
            if run == len(pset):
                fits[k] += 1
        n += 1

    recall = {f"oracle_recall@{k}": round(acc[k] / max(n, 1) * 100, 2) for k in ks}
    return recall, fits, n
