"""A4: align QASPER gold evidence onto our paragraph ids.

QASPER's ``answer.evidence`` is **verbatim text copied out of the paper**, not a coordinate.
So "save the annotations as an evaluation file" is not a copy job -- it needs alignment,
otherwise Member 5 gets strings and cannot compute retrieval recall.

Cascade, cheapest first:

1. ``FLOAT SELECTED`` prefix            -> evidence_level = "float"   (figure/table, not in body)
2. exact match on raw paragraph text    -> "paragraph"
3. exact match after normalisation      -> "paragraph_norm"
4. match on ``section_name``             -> "section"  (expands to every paragraph of that section)
5. fuzzy match (rapidfuzz)               -> "paragraph_fuzzy"
6. otherwise                             -> "fail"

Rule 4 exists because QASPER's ``section_name`` is a **hierarchical S2ORC path**
(``"Experiments ::: Ablation"``): 24.7% of paragraphs carry a ``:::`` level. A handful of
evidence items point at a whole subsection rather than a paragraph, and they show up as
exact section-name hits -- verified on the previously-unresolvable samples.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from rapidfuzz import fuzz, process

FLOAT_PREFIX = "FLOAT SELECTED"
WS_RE = re.compile(r"\s+")


def norm(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).replace("\u00ad", "")
    return WS_RE.sub(" ", s).strip()


def answer_type(answer: dict) -> str:
    if answer.get("unanswerable"):
        return "unanswerable"
    if any((x or "").strip() for x in (answer.get("extractive_spans") or [])):
        return "extractive"
    if answer.get("yes_no") is not None:
        return "yes_no"
    if (answer.get("free_form_answer") or "").strip():
        return "free_form"
    return "empty"


@dataclass
class AlignConfig:
    use_fuzzy: bool = True
    fuzzy_threshold: int = 88
    section_rule: bool = True

    def to_dict(self) -> dict:
        return {
            "use_fuzzy": self.use_fuzzy,
            "fuzzy_threshold": self.fuzzy_threshold,
            "section_rule": self.section_rule,
        }


@dataclass
class PaperIndex:
    raw: dict[str, str] = field(default_factory=dict)            # raw text -> para_id
    normed: dict[str, str] = field(default_factory=dict)         # norm text -> para_id
    section: dict[str, list[str]] = field(default_factory=dict)  # norm section -> para_ids
    norm_list: list[tuple[str, str]] = field(default_factory=list)  # (norm, para_id)

    @classmethod
    def build(cls, rows: list[dict]) -> PaperIndex:
        idx = cls()
        for r in rows:
            pid = r["para_id"]
            raw = r["text_raw"] or ""
            if raw and raw not in idx.raw:
                idx.raw[raw] = pid
            n = norm(raw)
            if n and n not in idx.normed:
                idx.normed[n] = pid
            idx.norm_list.append((n, pid))
            sec = norm(r["section_name"] or "")
            if sec:
                idx.section.setdefault(sec, []).append(pid)
        return idx


def align_evidence(ev: str, idx: PaperIndex, cfg: AlignConfig) -> tuple[str, list[str], str]:
    """Return (evidence_level, para_ids, note)."""
    if not isinstance(ev, str) or not ev.strip():
        return "empty", [], ""
    if ev.lstrip().startswith(FLOAT_PREFIX):
        return "float", [], ev[:80]
    if ev in idx.raw:
        return "paragraph", [idx.raw[ev]], ""
    n = norm(ev)
    if n in idx.normed:
        return "paragraph_norm", [idx.normed[n]], ""
    if cfg.section_rule and n in idx.section:
        return "section", idx.section[n], ""
    if cfg.use_fuzzy and idx.norm_list:
        best = process.extractOne(n, [x[0] for x in idx.norm_list], scorer=fuzz.token_set_ratio)
        if best and best[1] >= cfg.fuzzy_threshold:
            return "paragraph_fuzzy", [idx.norm_list[best[2]][1]], f"{best[1]}"
    return "fail", [], ev[:120]


def locate_spans(spans: list[str], rows: list[dict]) -> list[dict]:
    """Best-effort: which paragraph (and offset) does an extractive answer span live in?"""
    out: list[dict] = []
    for sp in spans or []:
        if not isinstance(sp, str) or not sp.strip():
            continue
        hit = {"text": sp, "para_id": None, "char_start": None}
        key = norm(sp)
        if key:
            for r in rows:
                pos = norm(r["text_raw"] or "").find(key)
                if pos >= 0:
                    hit["para_id"] = r["para_id"]
                    hit["char_start"] = pos
                    break
        out.append(hit)
    return out


def _zero_reason(n_gold: int, n_ev: int, n_float: int, n_mapped: int, n_fail: int) -> str:
    """Why does this question have no gold paragraph? M5 must not silently score these as 0."""
    if n_gold > 0:
        return ""
    if n_ev == 0:
        return "no_evidence"          # annotator answered without selecting any evidence (mostly yes/no)
    if n_mapped == 0 and n_float > 0 and n_float == n_ev:
        return "float_only"           # evidence is a figure/table caption, absent from body text
    if n_fail > 0:
        return "align_fail"
    return "other"


def align_paper(
    paper_qas: list[dict],
    para_rows: list[dict],
    split: str,
    cfg: AlignConfig,
) -> tuple[list[dict], Counter]:
    idx = PaperIndex.build(para_rows)
    levels: Counter[str] = Counter()
    records: list[dict] = []

    for qa in paper_qas:
        answers = qa.get("answers") or []
        types: list[str] = []
        gold: list[str] = []
        seen: set[str] = set()
        n_ev = n_float = n_mapped = n_fail = 0
        q_levels: set[str] = set()
        free_form: list[str] = []
        spans: list[str] = []
        yes_no = None
        for ann in answers:
            ans = ann.get("answer") or {}
            t = answer_type(ans)
            types.append(t)
            if (ans.get("free_form_answer") or "").strip():
                free_form.append(ans["free_form_answer"].strip())
            for sp in ans.get("extractive_spans") or []:
                if isinstance(sp, str) and sp.strip():
                    spans.append(sp.strip())
            if ans.get("yes_no") is not None and yes_no is None:
                yes_no = bool(ans["yes_no"])
            for ev in ans.get("evidence") or []:
                n_ev += 1
                level, pids, _note = align_evidence(ev, idx, cfg)
                levels[level] += 1
                q_levels.add(level)
                if level == "float":
                    n_float += 1
                elif level == "fail":
                    n_fail += 1
                else:
                    n_mapped += 1
                    for p in pids:
                        if p not in seen:
                            seen.add(p)
                            gold.append(p)

        records.append(
            {
                "question_id": qa.get("question_id"),
                "paper_id": None,  # filled by caller
                "split": split,
                "question": qa.get("question") or "",
                "n_answers": len(answers),
                "answer_types": ",".join(sorted(set(types))) if types else "no_answer",
                "is_answerable": any(t != "unanswerable" for t in types) if types else False,
                "n_evidence": n_ev,
                "n_evidence_float": n_float,
                "n_evidence_mapped": n_mapped,
                "n_evidence_failed": n_fail,
                "evidence_levels": ",".join(sorted(q_levels)),
                "gold_para_ids": gold,
                "n_gold_paras": len(gold),
                "zero_gold_reason": _zero_reason(len(gold), n_ev, n_float, n_mapped, n_fail),
                "is_float_only": n_float > 0 and n_mapped == 0,
                "free_form_answers": free_form,
                "extractive_spans": spans,
                "extractive_spans_located": locate_spans(spans, para_rows),
                "yes_no": yes_no,
                "writer_nlp_background": qa.get("nlp_background"),
                "writer_topic_background": qa.get("topic_background"),
                "writer_paper_read": qa.get("paper_read"),
                "search_query": qa.get("search_query"),
            }
        )
    return records, levels
