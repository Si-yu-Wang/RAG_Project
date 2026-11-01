"""P2: text cleaning with a provenance map.

Every rule is applied through :func:`_apply`, which rewrites the paragraph **and** keeps a
list of :class:`Segment` that maps cleaned-text offsets back to original-text offsets.
Without that map we could not later say "this chunk came from these characters of this
paragraph", and the whole evidence-provenance chain for Member 5 would break.

Rules are deliberately conservative: QASPER text comes from S2ORC, so most of the mess is
citation placeholders (``BIBREF12``), not LaTeX.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, asdict

# Zero-width / formatting characters that S2ORC and PDF extraction leave behind.
ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\u2060\ufeff\u00ad]")
# S2ORC inline placeholders: BIBREF0, TABREF12, FIGREF3 ...
# No leading \b: PDF extraction regularly glues them to the previous word
# ("event recommendationBIBREF6"), and those are exactly the ones we must fix.
# The replacement is prefixed with a space so a glued case becomes two tokens;
# the later space_collapse rule removes any duplicate space this introduces.
REF_RE = re.compile(r"(BIBREF|TABREF|FIGREF)\d+")
# "transla-\ntion" -> "translation"
HYPHEN_NL_RE = re.compile(r"(?<=[A-Za-z])-[ \t]*\n[ \t]*(?=[a-z])")
# A hard line break inside a paragraph is really a space.
NEWLINE_RE = re.compile(r"[ \t]*\n[ \t]*")
MULTISPACE_RE = re.compile(r"[ \t]{2,}")

REF_PLACEHOLDER = {"BIBREF": "[REF]", "TABREF": "[TABLE]", "FIGREF": "[FIGURE]"}


@dataclass(frozen=True)
class Segment:
    """Contiguous run of cleaned text and where it came from in the original."""

    clean_start: int
    clean_end: int
    orig_start: int
    orig_end: int
    rule: str


@dataclass
class CleanResult:
    text: str
    segments: list[Segment]
    hits: dict[str, int] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return bool(self.hits)

    def to_orig(self, pos: int) -> int:
        """Map a cleaned-text offset back to the original paragraph."""
        return _to_orig(self.segments, pos)


# ---------------------------------------------------------------------------
# low-level machinery
# ---------------------------------------------------------------------------
def _to_orig(segs: list[Segment], pos: int) -> int:
    if not segs:
        return pos
    lo, hi = 0, len(segs) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if segs[mid].clean_end < pos:
            lo = mid + 1
        else:
            hi = mid
    s = segs[lo]
    span = s.clean_end - s.clean_start
    if span <= 0:
        return s.orig_start
    if s.orig_end - s.orig_start == span:  # 1:1, the common case
        return s.orig_start + (pos - s.clean_start)
    ratio = (s.orig_end - s.orig_start) / span
    return s.orig_start + int(round((pos - s.clean_start) * ratio))


def _slice(segs: list[Segment], a: int, b: int, offset: int) -> list[Segment]:
    """Segments covering [a, b) of current text, shifted so that `a` lands at `offset`."""
    out: list[Segment] = []
    for s in segs:
        if s.clean_end <= a or s.clean_start >= b:
            continue
        ov_a = max(a, s.clean_start)
        ov_b = min(b, s.clean_end)
        o_a = _to_orig([s], ov_a)
        o_b = _to_orig([s], ov_b)
        out.append(Segment(offset + (ov_a - a), offset + (ov_b - a), o_a, o_b, s.rule))
    return out


def _merge(segs: list[Segment]) -> list[Segment]:
    """Coalesce adjacent segments that are contiguous in both coordinate systems."""
    if not segs:
        return segs
    out = [segs[0]]
    for s in segs[1:]:
        p = out[-1]
        if (
            p.rule == s.rule
            and p.clean_end == s.clean_start
            and p.orig_end == s.orig_start
            and p.clean_end > p.clean_start
        ):
            out[-1] = Segment(p.clean_start, s.clean_end, p.orig_start, s.orig_end, p.rule)
        else:
            out.append(s)
    return out


def _apply(
    text: str,
    segs: list[Segment],
    pattern: re.Pattern[str],
    repl,
    rule: str,
) -> tuple[str, list[Segment], int]:
    if not pattern.search(text):
        return text, segs, 0
    out: list[str] = []
    new_segs: list[Segment] = []
    cur_len = 0
    last = 0
    n = 0
    for m in pattern.finditer(text):
        a, b = m.span()
        if a > last:
            out.append(text[last:a])
            new_segs.extend(_slice(segs, last, a, cur_len))
            cur_len += a - last
        rep = repl(m) if callable(repl) else repl
        if rep:
            new_segs.append(
                Segment(cur_len, cur_len + len(rep), _to_orig(segs, a), _to_orig(segs, b), rule)
            )
            out.append(rep)
            cur_len += len(rep)
        n += 1
        last = b
    if last < len(text):
        out.append(text[last:])
        new_segs.extend(_slice(segs, last, len(text), cur_len))
    return "".join(out), _merge(new_segs), n


def _strip(text: str, segs: list[Segment]) -> tuple[str, list[Segment], int]:
    start = len(text) - len(text.lstrip())
    end = len(text.rstrip())
    if start == 0 and end == len(text):
        return text, segs, 0
    if end <= start:
        return "", [], 1
    new = text[start:end]
    return new, _merge(_slice(segs, start, end, 0)), 1


def _nfkc(text: str, segs: list[Segment]) -> tuple[str, list[Segment], int]:
    # Fast path: one C-level call tells us whether anything changes at all.
    if unicodedata.normalize("NFKC", text) == text:
        return text, segs, 0
    out: list[str] = []
    new_segs: list[Segment] = []
    cur = 0
    n = 0
    for i, ch in enumerate(text):
        rep = unicodedata.normalize("NFKC", ch)
        if rep != ch:
            n += 1
        out.append(rep)
        new_segs.append(
            Segment(cur, cur + len(rep), _to_orig(segs, i), _to_orig(segs, i + 1),
                    "nfkc" if rep != ch else "keep")
        )
        cur += len(rep)
    return "".join(out), _merge(new_segs), n


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
@dataclass
class CleanConfig:
    nfkc: bool = True
    strip_zero_width: bool = True
    ref_mode: str = "placeholder"  # "keep" | "placeholder" | "drop"
    join_hyphenated: bool = True
    join_newlines: bool = True
    collapse_spaces: bool = True
    strip: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class Cleaner:
    def __init__(self, config: CleanConfig | None = None) -> None:
        self.config = config or CleanConfig()
        if self.config.ref_mode not in {"keep", "placeholder", "drop"}:
            raise ValueError(f"bad ref_mode: {self.config.ref_mode}")

    def clean(self, text: str) -> CleanResult:
        cfg = self.config
        hits: dict[str, int] = {}
        segs = [Segment(0, len(text), 0, len(text), "keep")] if text else []

        if cfg.nfkc:
            text, segs, n = _nfkc(text, segs)
            if n:
                hits["nfkc"] = n
        if cfg.strip_zero_width:
            text, segs, n = _apply(text, segs, ZERO_WIDTH_RE, "", "zero_width")
            if n:
                hits["zero_width"] = n
        if cfg.ref_mode != "keep":
            # Only a *glued* placeholder needs a separating space; adding one everywhere
            # would fake thousands of "double space" hits and inflate the edit count.
            def _repl(m, _src=text):
                prev = _src[m.start() - 1] if m.start() > 0 else ""
                lead = " " if prev.isalnum() else ""
                if cfg.ref_mode == "drop":
                    return lead
                return lead + REF_PLACEHOLDER[m.group(1)]

            rule = "drop_ref" if cfg.ref_mode == "drop" else "ref_placeholder"
            text, segs, n = _apply(text, segs, REF_RE, _repl, rule)
            if n:
                hits["refs"] = n
        if cfg.join_hyphenated:
            text, segs, n = _apply(text, segs, HYPHEN_NL_RE, "", "hyphen_join")
            if n:
                hits["hyphen_join"] = n
        if cfg.join_newlines:
            text, segs, n = _apply(text, segs, NEWLINE_RE, " ", "newline_join")
            if n:
                hits["newline_join"] = n
        if cfg.collapse_spaces:
            text, segs, n = _apply(text, segs, MULTISPACE_RE, " ", "space_collapse")
            if n:
                hits["space_collapse"] = n
        if cfg.strip:
            text, segs, n = _strip(text, segs)
            if n:
                hits["strip"] = n

        return CleanResult(text=text, segments=segs, hits=hits)


def nfkc_is_charwise(text: str) -> bool:
    """NFKC applied per character must equal NFKC applied to the whole string.

    Our offset map assumes that; this helper lets a test assert it on the real corpus.
    """
    return "".join(unicodedata.normalize("NFKC", c) for c in text) == unicodedata.normalize(
        "NFKC", text
    )


# ---------------------------------------------------------------------------
# compact serialisation for the parquet column
# ---------------------------------------------------------------------------
def encode_segments(segs: list[Segment]) -> str:
    """``"0,12,0,12,keep;12,20,12,31,ref_placeholder"`` — only for paragraphs that changed."""
    if len(segs) == 1 and segs[0].rule == "keep":
        return ""
    return ";".join(f"{s.clean_start},{s.clean_end},{s.orig_start},{s.orig_end},{s.rule}" for s in segs)


def decode_segments(blob: str, text_len: int) -> list[Segment]:
    if not blob:
        return [Segment(0, text_len, 0, text_len, "keep")]
    out: list[Segment] = []
    for part in blob.split(";"):
        cs, ce, os_, oe, rule = part.split(",")
        out.append(Segment(int(cs), int(ce), int(os_), int(oe), rule))
    return out
