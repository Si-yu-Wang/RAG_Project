"""A5: sentence-aware, token-bounded chunking with paragraph provenance.

Design constraints come from the data, not from taste (see docs/dataset/dataset_profile.md):

* paragraphs are the gold-evidence unit, so every chunk records **which paragraphs and
  which character ranges** it covers -- otherwise Member 5 cannot map gold evidence to chunks;
* the longest paragraph is 3,636 tokens, so a sentence that alone exceeds the budget is
  hard-split **on token boundaries** (never inside a token) and flagged ``is_truncated``;
* ``section_name`` is a hierarchical S2ORC path (``"Experiments ::: Ablation"``), so section
  boundaries are real structure -- hence the ``section_boundary`` knob.

Chunk ids are ``{paper_id}#s{sec}#c{idx}`` where ``sec`` is ``abs`` for the abstract or the
zero-padded ``section_idx``. Ids are deterministic and independent of global ordering, so
running one paper alone yields the same ids as running the whole corpus.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass

SEP = "\n\n"
VERSION = "1.0.0"
ABS_KEY = "abs"

_ABBREV_GUARD = re.compile(
    r"\b(?:et al|e\.g|i\.e|Fig|Figs|Eq|Eqs|Sec|Secs|Tab|Ref|Refs|vs|cf|resp|approx)\.$"
)
_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


_SEGMENTER = None
_SEGMENTER_TRIED = False


def _pysbd_segment(text: str) -> list[str]:
    """Sentence-split via pysbd, reusing one Segmenter instance.

    pysbd builds a dozen regexes in ``__init__``; constructing one per paragraph was the
    single hottest call in the whole pipeline (61k paragraphs). The segmenter is stateless
    between ``segment()`` calls, so a module-level singleton is safe and output-identical.
    """
    global _SEGMENTER, _SEGMENTER_TRIED
    if not _SEGMENTER_TRIED:
        _SEGMENTER_TRIED = True
        try:
            import pysbd

            _SEGMENTER = pysbd.Segmenter(language="en", clean=False)
        except Exception:  # pragma: no cover - optional dependency
            _SEGMENTER = None
    if _SEGMENTER is None:  # pragma: no cover
        return []
    try:
        return _SEGMENTER.segment(text)
    except Exception:  # pragma: no cover
        return []


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Sentence (start, end) char offsets that always cover the whole text."""
    if not text:
        return []
    segs = _pysbd_segment(text)
    spans: list[tuple[int, int]] = []
    cursor = 0
    if segs:
        ok = True
        for seg in segs:
            if not seg:
                continue
            idx = text.find(seg, cursor)
            if idx < 0:
                ok = False
                break
            spans.append((idx, idx + len(seg)))
            cursor = idx + len(seg)
        if not ok:
            spans = []
    if not spans:
        last = 0
        for m in _SPLIT_RE.finditer(text):
            cut = m.start()
            if _ABBREV_GUARD.search(text[max(0, cut - 12) : cut + 1]):
                continue
            spans.append((last, cut))
            last = m.end()
        spans.append((last, len(text)))
    if spans[0][0] != 0:
        spans[0] = (0, spans[0][1])
    if spans[-1][1] != len(text):
        spans[-1] = (spans[-1][0], len(text))
    return spans


@dataclass
class ChunkConfig:
    chunk_size: int = 512
    overlap_ratio: float = 0.0
    overlap_tokens: int | None = None
    min_chunk_tokens: int = 64
    section_boundary: str = "hard"  # hard | soft
    include_abstract: bool = True
    merge_short_tail: bool = True

    def resolved_overlap(self) -> int:
        if self.overlap_tokens is not None:
            return int(self.overlap_tokens)
        return int(round(self.chunk_size * self.overlap_ratio))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["resolved_overlap_tokens"] = self.resolved_overlap()
        d["chunker_version"] = VERSION
        return d

    def hash(self) -> str:
        return hashlib.sha1(repr(sorted(self.to_dict().items())).encode()).hexdigest()[:12]

    def validate(self) -> None:
        if self.chunk_size < 4:
            raise ValueError("chunk_size too small")
        ov = self.resolved_overlap()
        if ov < 0:
            raise ValueError("overlap must be >= 0")
        if ov >= self.chunk_size:
            raise ValueError(f"overlap {ov} must be < chunk_size {self.chunk_size}")
        if self.section_boundary not in {"hard", "soft"}:
            raise ValueError("section_boundary must be hard|soft")
        if self.min_chunk_tokens >= self.chunk_size:
            raise ValueError("min_chunk_tokens must be < chunk_size")


@dataclass
class _Sent:
    para_idx: int
    abs_start: int
    abs_end: int
    n_tokens: int
    sec_idx: int
    sec_name: str
    is_abs: bool
    hard_split: bool = False


@dataclass
class ChunkRow:
    chunk_id: str
    paper_id: str
    split: str
    section_idx: int
    section_name: str
    section_path: str
    sec_key: str
    chunk_idx_in_section: int
    chunk_idx_in_paper: int
    text: str
    n_tokens: int
    char_start: int
    char_end: int
    para_ids: list[str]
    para_spans: list[tuple[str, int, int]]
    is_abstract: bool
    is_truncated: bool
    n_sentences: int
    n_sections: int = 1
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    text_sha1: str = ""
    config_hash: str = ""


def section_path_of(name: str) -> str:
    parts = [p.strip() for p in (name or "").split(":::") if p.strip()]
    return " > ".join(parts)


def _build_flat(para_rows: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    pieces: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for i, r in enumerate(para_rows):
        if i:
            cursor += len(SEP)
        s = cursor
        pieces.append(r["text_clean"])
        cursor = s + len(r["text_clean"])
        spans.append((s, cursor))
    return SEP.join(pieces), spans


def _sentence_stream(para_rows: list[dict], pspans: list[tuple[int, int]],
                     cfg: ChunkConfig, counter,
                     sent_cache: dict | None = None) -> list[_Sent]:
    """Flatten paragraphs into a sentence stream.

    ``sent_cache`` maps ``para_id -> [(char_a, char_b, n_tokens), ...]`` and lets a grid
    sweep reuse the (config-independent) sentence splits and their token counts across all
    configurations. It is keyed by ``para_id`` and validated against ``text_clean`` length,
    so a stale cache can never silently poison a run.
    """
    out: list[_Sent] = []
    for i, r in enumerate(para_rows):
        si = int(r["section_idx"])
        name = r["section_name"] or ""
        is_abs = bool(r["is_abstract"])
        if is_abs and not cfg.include_abstract:
            continue
        base = pspans[i][0]
        text = r["text_clean"]
        cached = None
        if sent_cache is not None:
            cached = sent_cache.get(r["para_id"])
            if cached is not None and cached[0] != len(text):
                cached = None
        if cached is None:
            pieces = []
            for (a, b) in split_sentences(text):
                piece = text[a:b]
                if not piece.strip():
                    continue
                pieces.append((a, b, counter.count(piece)))
            if sent_cache is not None:
                sent_cache[r["para_id"]] = (len(text), pieces)
            cached = (len(text), pieces)
        for (a, b, nt) in cached[1]:
            if nt <= cfg.chunk_size:
                out.append(_Sent(i, base + a, base + b, nt, si, name, is_abs))
                continue
            # A single sentence longer than the budget: hard-split on token boundaries.
            # This branch depends on chunk_size, so it is never served from the cache.
            piece = text[a:b]
            offs = counter.char_offsets(piece)
            if not offs:
                out.append(_Sent(i, base + a, base + b, nt, si, name, is_abs, True))
                continue
            k = 0
            while k < len(offs):
                nxt = min(k + cfg.chunk_size, len(offs))
                ca, cb = offs[k][0], offs[nxt - 1][1]
                sub = piece[ca:cb]
                if sub.strip():
                    out.append(_Sent(i, base + a + ca, base + a + cb, counter.count(sub),
                                     si, name, is_abs, True))
                k = nxt
    return out


def _normalize_spans(spans: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    """Collapse to exactly one span per paragraph, in first-appearance order.

    A chunk's sentences interleave, so a paragraph that contributes several sentences would
    otherwise be emitted once per sentence. A paragraph's intersection with one contiguous
    chunk is itself contiguous, so merging to ``[min(start), max(end))`` is lossless -- and it
    keeps the contract invariant ``len(para_spans) == len(para_ids)``.
    """
    order: list[str] = []
    limits: dict[str, list[int]] = {}
    for pid, a, b in spans:
        cur = limits.get(pid)
        if cur is None:
            order.append(pid)
            limits[pid] = [a, b]
        else:
            if a < cur[0]:
                cur[0] = a
            if b > cur[1]:
                cur[1] = b
    return [(pid, limits[pid][0], limits[pid][1]) for pid in order]


def chunk_paper(
    para_rows: list[dict],
    cfg: ChunkConfig,
    counter,
    paper_id: str,
    split: str,
    sent_cache: dict | None = None,
) -> list[ChunkRow]:
    """Chunk one paper. ``para_rows`` must be in reading order."""
    cfg.validate()
    rows_in = [r for r in para_rows if (r["text_clean"] or "").strip()]
    if not rows_in:
        return []
    flat, pspans = _build_flat(rows_in)
    sents = _sentence_stream(rows_in, pspans, cfg, counter, sent_cache)
    if not sents:
        return []

    overlap = cfg.resolved_overlap()
    chunks: list[ChunkRow] = []
    per_section_counter: dict[str, int] = {}

    def make(sl: list[_Sent]) -> ChunkRow:
        a, b = sl[0].abs_start, sl[-1].abs_end
        text = flat[a:b]
        base = sl[0]
        pids: list[str] = []
        pspans_out: list[tuple[str, int, int]] = []
        for s in sl:
            r = rows_in[s.para_idx]
            pid = r["para_id"]
            p0, p1 = pspans[s.para_idx]
            ca, cb = max(a, p0), min(b, p1)
            if pid not in pids:
                pids.append(pid)
            if ca < cb:
                pspans_out.append((pid, ca - p0, cb - p0))
        key = ABS_KEY if base.is_abs else f"{base.sec_idx:02d}"
        ci = per_section_counter.get(key, 0)
        per_section_counter[key] = ci + 1
        return ChunkRow(
            chunk_id=f"{paper_id}#s{key}#c{ci:03d}",
            paper_id=paper_id,
            split=split,
            section_idx=base.sec_idx,
            section_name=base.sec_name,
            section_path=section_path_of(base.sec_name),
            sec_key=key,
            chunk_idx_in_section=ci,
            chunk_idx_in_paper=len(chunks),
            text=text,
            n_tokens=counter.count(text),
            char_start=a,
            char_end=b,
            para_ids=pids,
            para_spans=_normalize_spans(pspans_out),
            is_abstract=base.is_abs,
            is_truncated=any(s.hard_split for s in sl),
            n_sentences=len(sl),
            n_sections=len({(s.sec_idx, s.is_abs) for s in sl}),
            text_sha1=hashlib.sha1(text.encode("utf-8")).hexdigest(),
            config_hash=cfg.hash(),
        )

    i, n = 0, len(sents)
    while i < n:
        acc = 0
        last_para = sents[i].para_idx
        j = i
        broke_on_section = False
        while j < n:
            s = sents[j]
            if j > i:
                same_sec = (s.sec_idx, s.is_abs) == (sents[i].sec_idx, sents[i].is_abs)
                if not same_sec and cfg.section_boundary == "hard":
                    broke_on_section = True
                    break
                if (
                    same_sec
                    and s.para_idx != last_para
                    and acc >= cfg.min_chunk_tokens
                    and acc + s.n_tokens > cfg.chunk_size
                ):
                    break  # prefer to end on a paragraph boundary
            if acc + s.n_tokens > cfg.chunk_size and j > i:
                break
            acc += s.n_tokens
            last_para = s.para_idx
            j += 1
            if acc >= cfg.chunk_size:
                break
        if j == i:
            j = i + 1
        chunks.append(make(sents[i:j]))
        if j >= n:
            break
        # Overlap exists to preserve continuity across a cut *inside* a section. Cutting at a
        # section boundary and then backtracking would re-emit the previous section's tail as a
        # spurious chunk, so in that case we start the next chunk cleanly at j.
        if overlap <= 0 or broke_on_section:
            i = j
            continue
        k = j
        acc_ov = 0
        while k > i + 1 and acc_ov < overlap:
            k -= 1
            acc_ov += sents[k].n_tokens
        i = k if k > i else i + 1

    # ---- merge a too-short tail into the previous chunk of the same section ----
    if cfg.merge_short_tail:
        merged: list[ChunkRow] = []
        for r in chunks:
            if (
                merged
                and merged[-1].sec_key == r.sec_key
                and r.n_tokens < cfg.min_chunk_tokens
            ):
                p = merged[-1]
                text = flat[p.char_start : r.char_end]
                nt = counter.count(text)
                if nt <= cfg.chunk_size:
                    p.text = text
                    p.n_tokens = nt
                    p.char_end = r.char_end
                    p.para_ids = list(dict.fromkeys(p.para_ids + r.para_ids))
                    p.para_spans = _normalize_spans(p.para_spans + r.para_spans)
                    p.n_sentences += r.n_sentences
                    p.is_truncated = p.is_truncated or r.is_truncated
                    p.text_sha1 = hashlib.sha1(text.encode("utf-8")).hexdigest()
                    continue
            merged.append(r)
        chunks = merged
        # re-index after merging
        per_section_counter.clear()
        for idx, r in enumerate(chunks):
            r.chunk_idx_in_paper = idx
            ci = per_section_counter.get(r.sec_key, 0)
            per_section_counter[r.sec_key] = ci + 1
            r.chunk_idx_in_section = ci
            r.chunk_id = f"{paper_id}#s{r.sec_key}#c{ci:03d}"

    for a, b in zip(chunks, chunks[1:]):
        a.next_chunk_id = b.chunk_id
        b.prev_chunk_id = a.chunk_id
    return chunks
