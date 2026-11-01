"""Raw QASPER loader.

Reads the *official* JSON release (``qasper-{train,dev}-v0.3.json``) rather than the
HuggingFace ``datasets`` mirror, because the mirror flattens ``full_text`` and loses the
paragraph-level ``section_name`` that our chunking coordinates depend on.

Official file layout (v0.3)::

    { "1909.00319": { "title": str,
                      "abstract": str,
                      "full_text": [ {"section_name": str,
                                      "paragraphs": [{"section_name": str,
                                                      "paragraph_text": str}, ...]}, ...],
                      "figures_and_tables": [{"file": str, "caption": str}, ...],
                      "qas": [ {...}, ... ] }, ... }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

PARAGRAPH_TEXT_KEYS = ("paragraph_text", "text")


@dataclass
class Paragraph:
    section_name: str
    text: str


@dataclass
class Section:
    name: str
    paragraphs: list[Paragraph] = field(default_factory=list)


@dataclass
class Paper:
    paper_id: str
    split: str
    title: str
    abstract: str
    sections: list[Section] = field(default_factory=list)
    qas: list[dict[str, Any]] = field(default_factory=list)
    figures_and_tables: list[dict[str, Any]] = field(default_factory=list)
    raw_keys: tuple[str, ...] = ()

    # -- derived helpers -------------------------------------------------
    @property
    def paragraphs(self) -> list[tuple[int, str, Paragraph]]:
        """(section_idx, section_name, paragraph) in reading order."""
        out: list[tuple[int, str, Paragraph]] = []
        for si, sec in enumerate(self.sections):
            for para in sec.paragraphs:
                out.append((si, sec.name, para))
        return out

    def body_text(self, sep: str = "\n\n") -> str:
        return sep.join(p.text for _, _, p in self.paragraphs)


def _paragraph_text(p: Any) -> str:
    """v0.3 stores paragraphs as **plain strings**; tolerate dicts for forward compat."""
    if isinstance(p, str):
        return p
    if isinstance(p, dict):
        for k in PARAGRAPH_TEXT_KEYS:
            if isinstance(p.get(k), str):
                return p[k]
    return ""


def normalize_paper(paper_id: str, raw: dict[str, Any], split: str) -> Paper:
    sections: list[Section] = []
    for sec in raw.get("full_text") or []:
        name = sec.get("section_name") or ""
        paras = []
        for p in sec.get("paragraphs") or []:
            sub_name = p.get("section_name", name) if isinstance(p, dict) else name
            paras.append(Paragraph(section_name=sub_name, text=_paragraph_text(p)))
        sections.append(Section(name=name, paragraphs=paras))
    return Paper(
        paper_id=paper_id,
        split=split,
        title=raw.get("title") or "",
        abstract=raw.get("abstract") or "",
        sections=sections,
        qas=raw.get("qas") or [],
        figures_and_tables=raw.get("figures_and_tables") or [],
        raw_keys=tuple(sorted(raw.keys())),
    )


def load_raw(path: str) -> dict[str, dict[str, Any]]:
    with open(path, "rb") as f:
        return json.load(f)


def load_split(path: str, split: str) -> list[Paper]:
    raw = load_raw(path)
    return [normalize_paper(pid, d, split) for pid, d in raw.items()]


def load_all(paths: dict[str, str]) -> list[Paper]:
    papers: list[Paper] = []
    for split, path in paths.items():
        papers.extend(load_split(path, split))
    return papers
