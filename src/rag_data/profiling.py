"""Dataset profiling for QASPER (Phase P1).

Everything here is *measurement*, not transformation: the point is to replace assumptions
about the corpus with numbers, especially about how ``evidence`` strings relate to the
paragraphs they are supposed to be copied from.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd

from .loader import Paper
from .tokenizer import TokenCounter

WS_RE = re.compile(r"\s+")
FLOAT_PREFIX = "FLOAT SELECTED"


# --------------------------------------------------------------------------
# text helpers
# --------------------------------------------------------------------------
def norm_text(s: str) -> str:
    """Aggressive-but-lossless normalization used only for *matching*."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00ad", "")  # soft hyphen
    return WS_RE.sub(" ", s).strip()


def answer_type(answer: dict[str, Any]) -> str:
    """Classify one annotation into one of the four QASPER answer types."""
    if answer.get("unanswerable"):
        return "unanswerable"
    spans = answer.get("extractive_spans") or []
    if any(isinstance(x, str) and x.strip() for x in spans):
        return "extractive"
    if answer.get("yes_no") is not None:
        return "yes_no"
    if (answer.get("free_form_answer") or "").strip():
        return "free_form"
    return "empty"


def non_empty_type_count(answer: dict[str, Any]) -> int:
    """v0.3 promises 'exactly one non-empty' field; count violations to verify it."""
    n = 0
    n += 1 if answer.get("unanswerable") else 0
    n += 1 if any((x or "").strip() for x in (answer.get("extractive_spans") or [])) else 0
    n += 1 if answer.get("yes_no") is not None else 0
    n += 1 if (answer.get("free_form_answer") or "").strip() else 0
    return n


def dist(series: pd.Series) -> pd.DataFrame:
    vc = series.value_counts(dropna=False).sort_index()
    out = pd.DataFrame({series.name or "value": vc.index, "count": vc.to_numpy()})
    out["pct"] = (out["count"] / out["count"].sum() * 100).round(2)
    return out


def quantiles(values: list[float], name: str) -> dict[str, float]:
    if not values:
        return {}
    a = np.asarray(values, dtype=float)
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 2),
        "p10": round(float(np.percentile(a, 10)), 2),
        "p50": round(float(np.percentile(a, 50)), 2),
        "p90": round(float(np.percentile(a, 90)), 2),
        "p99": round(float(np.percentile(a, 99)), 2),
        "max": round(float(a.max()), 2),
    }


# --------------------------------------------------------------------------
# evidence <-> paragraph alignment probe
# --------------------------------------------------------------------------
def probe_evidence_alignment(paper: Paper) -> dict[str, Any]:
    """How well do ``evidence`` strings map back onto paragraphs?

    QASPER guarantees that textual ``evidence`` is *entire paragraphs*. If that holds,
    alignment for ``evidence`` is essentially exact-match and P6 becomes cheap; only
    ``highlighted_evidence`` (sentence level) needs substring search.
    """
    paras = paper.paragraphs
    raw_map: dict[str, list[int]] = defaultdict(list)
    norm_map: dict[str, list[int]] = defaultdict(list)
    for gi, (_, _, p) in enumerate(paras):
        if p.text.strip():
            raw_map[p.text].append(gi)
            norm_map[norm_text(p.text)].append(gi)
    abs_norm = norm_text(paper.abstract)

    res: Counter[str] = Counter()
    float_items = 0
    empty_items = 0
    failures: list[tuple[str, str]] = []
    n_evidence = 0
    n_highlight = 0
    hl_in_para = 0

    for qa in paper.qas:
        for ann in qa.get("answers") or []:
            ans = ann.get("answer") or {}
            for ev in ans.get("evidence") or []:
                n_evidence += 1
                if not isinstance(ev, str) or not ev.strip():
                    empty_items += 1
                    res["empty"] += 1
                    continue
                if ev.lstrip().startswith(FLOAT_PREFIX):
                    float_items += 1
                    res["float"] += 1
                    continue
                if ev in raw_map:
                    res["exact_raw"] += 1
                    continue
                ne = norm_text(ev)
                if ne in norm_map:
                    res["exact_norm"] += 1
                    continue
                if ne and abs_norm and ne == abs_norm:
                    res["abstract"] += 1
                    continue
                hit = False
                for np_ in norm_map:
                    if ne and np_ and (ne in np_ or np_ in ne):
                        hit = True
                        break
                if hit:
                    res["substring"] += 1
                else:
                    res["fail"] += 1
                    if len(failures) < 5:
                        failures.append((paper.paper_id, ev[:160]))

            for hl in ans.get("highlighted_evidence") or []:
                if not isinstance(hl, str) or not hl.strip():
                    continue
                n_highlight += 1
                nh = norm_text(hl)
                if any(nh and nh in k for k in norm_map):
                    hl_in_para += 1

    return {
        "n_evidence": n_evidence,
        "n_float": float_items,
        "n_empty_evidence": empty_items,
        "n_highlight": n_highlight,
        "n_highlight_in_para": hl_in_para,
        "buckets": dict(res),
        "failures": failures,
    }


# --------------------------------------------------------------------------
# main profile
# --------------------------------------------------------------------------
def build_profile(papers: list[Paper], counter: TokenCounter) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    stats: dict[str, Any] = {}
    tables: dict[str, pd.DataFrame] = {}

    # ---- 1. schema ------------------------------------------------------
    key_shapes = Counter(p.raw_keys for p in papers)
    tables["top_level_keys"] = pd.DataFrame(
        [{"keys": " | ".join(k), "papers": v, "pct": round(v / len(papers) * 100, 2)}
         for k, v in key_shapes.most_common()]
    )

    # ---- 2. corpus scale ------------------------------------------------
    body_tokens, body_chars, abs_tokens = [], [], []
    for p in papers:
        body_tokens.append(counter.count(p.body_text()))
        body_chars.append(len(p.body_text()))
        abs_tokens.append(counter.count(p.abstract))
    stats["scale"] = {
        "papers": len(papers),
        "body_tokens": quantiles(body_tokens, "body_tokens"),
        "body_chars": quantiles(body_chars, "body_chars"),
        "abstract_tokens": quantiles(abs_tokens, "abstract_tokens"),
        "total_body_tokens": int(sum(body_tokens)),
        "total_abstract_tokens": int(sum(abs_tokens)),
    }

    # ---- 3. structure ---------------------------------------------------
    n_sections = [len(p.sections) for p in papers]
    paras_per_section = [len(s.paragraphs) for p in papers for s in p.sections]
    empty_paras = sum(1 for p in papers for _, _, para in p.paragraphs if not para.text.strip())
    total_paras = sum(len(p.paragraphs) for p in papers)
    dup_name_papers = 0
    dup_name_total = 0
    for p in papers:
        c = Counter(s.name for s in p.sections)
        dups = {k: v for k, v in c.items() if v > 1}
        if dups:
            dup_name_papers += 1
            dup_name_total += sum(v - 1 for v in dups.values())
    para_tokens = [counter.count(para.text) for p in papers for _, _, para in p.paragraphs]
    stats["structure"] = {
        "sections_per_paper": quantiles(n_sections, "sections"),
        "paras_per_section": quantiles(paras_per_section, "paras"),
        "paragraph_tokens": quantiles(para_tokens, "tokens"),
        "total_paragraphs": total_paras,
        "empty_paragraphs": empty_paras,
        "empty_paragraph_pct": round(empty_paras / max(total_paras, 1) * 100, 3),
        "papers_with_dup_section_name": dup_name_papers,
        "papers_with_dup_section_name_pct": round(dup_name_papers / max(len(papers), 1) * 100, 2),
        "extra_dup_occurrences": dup_name_total,
        "papers_with_zero_sections": sum(1 for p in papers if not p.sections),
        "papers_with_empty_abstract": sum(1 for p in papers if not p.abstract.strip()),
    }

    # ---- 4. questions & answers ----------------------------------------
    n_qas = [len(p.qas) for p in papers]
    ans_per_q, ev_per_q, hl_per_q = [], [], []
    type_counter: Counter[str] = Counter()
    q_type_sets: Counter[str] = Counter()
    violations = 0
    n_answers = 0
    for p in papers:
        for qa in p.qas:
            answers = qa.get("answers") or []
            ans_per_q.append(len(answers))
            types = set()
            q_ev = 0
            q_hl = 0
            for ann in answers:
                ans = ann.get("answer") or {}
                t = answer_type(ans)
                type_counter[t] += 1
                types.add(t)
                n_answers += 1
                if non_empty_type_count(ans) != 1:
                    violations += 1
                q_ev += len(ans.get("evidence") or [])
                q_hl += len(ans.get("highlighted_evidence") or [])
            q_type_sets["+".join(sorted(types)) if types else "no_answer"] += 1
            ev_per_q.append(q_ev)
            hl_per_q.append(q_hl)

    stats["qa"] = {
        "n_papers": len(papers),
        "n_questions": sum(n_qas),
        "n_answers": n_answers,
        "qas_per_paper": quantiles(n_qas, "qas"),
        "answers_per_question": quantiles(ans_per_q, "ans"),
        "evidence_per_question": quantiles(ev_per_q, "ev"),
        "highlight_per_question": quantiles(hl_per_q, "hl"),
        "answers_violating_single_type_rule": violations,
        "answers_violating_pct": round(violations / max(n_answers, 1) * 100, 3),
        "questions_without_answers": sum(1 for x in ans_per_q if x == 0),
    }
    tables["answer_type"] = dist(pd.Series([t for t in type_counter.elements()], name="answer_type"))
    tables["question_type_combo"] = dist(pd.Series(list(q_type_sets.elements()), name="type_combo"))

    # ---- 5. writer metadata --------------------------------------------
    for field_name in ("nlp_background", "topic_background", "paper_read"):
        vals = []
        for p in papers:
            for qa in p.qas:
                v = qa.get(field_name)
                vals.append("(empty)" if v in (None, "") else str(v))
        tables[f"meta_{field_name}"] = dist(pd.Series(vals, name=field_name))
    sq = [bool((qa.get("search_query") or "").strip()) for p in papers for qa in p.qas]
    stats["search_query_filled_pct"] = round(sum(sq) / max(len(sq), 1) * 100, 2)

    # ---- 6. evidence alignment probe -----------------------------------
    buckets: Counter[str] = Counter()
    fails: list[tuple[str, str]] = []
    tot_ev = tot_hl = tot_hl_in = tot_float = tot_empty = 0
    for p in papers:
        r = probe_evidence_alignment(p)
        buckets.update(r["buckets"])
        tot_ev += r["n_evidence"]
        tot_float += r["n_float"]
        tot_empty += r["n_empty_evidence"]
        tot_hl += r["n_highlight"]
        tot_hl_in += r["n_highlight_in_para"]
        fails.extend(r["failures"])
    textual = tot_ev - tot_float - tot_empty
    aligned = buckets["exact_raw"] + buckets["exact_norm"] + buckets["substring"] + buckets["abstract"]
    stats["evidence"] = {
        "total_evidence_items": tot_ev,
        "float_items": tot_float,
        "float_pct_of_all": round(tot_float / max(tot_ev, 1) * 100, 2),
        "empty_items": tot_empty,
        "textual_items": textual,
        "exact_raw": buckets["exact_raw"],
        "exact_norm": buckets["exact_norm"],
        "substring": buckets["substring"],
        "abstract": buckets["abstract"],
        "fail": buckets["fail"],
        "alignable_pct_of_textual": round(aligned / max(textual, 1) * 100, 2),
        "highlight_items": tot_hl,
        "highlight_in_paragraph_pct": round(tot_hl_in / max(tot_hl, 1) * 100, 2),
        "sample_failures": fails[:8],
    }
    tables["evidence_buckets"] = pd.DataFrame(
        [{"bucket": k, "count": v, "pct_of_all": round(v / max(tot_ev, 1) * 100, 2)}
         for k, v in buckets.most_common()]
    )

    # ---- 7. figures -----------------------------------------------------
    fig_counts = [len(p.figures_and_tables) for p in papers]
    stats["figures"] = quantiles(fig_counts, "figs")

    # ---- 8. S2ORC artifact tokens (BIBREF / TABREF / FIGREF) --------------
    art_re = re.compile(r"\b(BIBREF|TABREF|FIGREF)\d+\b")
    art_counts = Counter()
    paras_with_art = 0
    total_paras_all = 0
    per_paper_art = []
    sample_para = None
    for p in papers:
        c = Counter()
        for _, _, para in p.paragraphs:
            total_paras_all += 1
            hits = art_re.findall(para.text)
            if hits:
                paras_with_art += 1
                c.update(hits)
                if sample_para is None and len(para.text) < 400:
                    sample_para = para.text
        art_counts.update(c)
        per_paper_art.append(sum(c.values()))
    stats["artifacts"] = {
        "total_occurrences": dict(art_counts),
        "paragraphs_with_artifacts": paras_with_art,
        "paragraphs_with_artifacts_pct": round(paras_with_art / max(total_paras_all, 1) * 100, 2),
        "per_paper": quantiles(per_paper_art, "art"),
        "sample_paragraph": sample_para,
    }

    # ---- 9. per-split summary ------------------------------------------
    rows = []
    for split in sorted({p.split for p in papers}):
        sub = [p for p in papers if p.split == split]
        rows.append({
            "split": split,
            "papers": len(sub),
            "questions": sum(len(p.qas) for p in sub),
            "answers": sum(len(qa.get("answers") or []) for p in sub for qa in p.qas),
            "body_tokens": sum(counter.count(p.body_text()) for p in sub),
            "avg_body_tokens": round(np.mean([counter.count(p.body_text()) for p in sub]), 1),
        })
    tables["split_summary"] = pd.DataFrame(rows)
    stats["tokenizer_backend"] = counter.backend

    return stats, tables
