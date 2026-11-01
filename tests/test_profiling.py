from __future__ import annotations

from rag_data.loader import normalize_paper
from rag_data.profiling import answer_type, non_empty_type_count, norm_text

RAW = {
    "title": "A Paper",
    "abstract": "We study  things.",
    "full_text": [
        {
            "section_name": "Introduction",
            "paragraphs": [
                {"section_name": "Introduction", "paragraph_text": "First para."},
                {"section_name": "Introduction", "paragraph_text": "Second  para."},
            ],
        },
        {"section_name": "Introduction", "paragraphs": [
            {"section_name": "Introduction", "paragraph_text": "Dup name section."}
        ]},
        {"section_name": "Method", "paragraphs": [
            {"section_name": "Method", "paragraph_text": ""}
        ]},
    ],
    "figures_and_tables": [{"file": "1-Table1-1.png", "caption": "Table 1: ..."}],
    "qas": [{"question_id": "q1", "question": "What?", "answers": []}],
}


def test_normalize_paper_structure():
    p = normalize_paper("1234.5678", RAW, "train")
    assert p.paper_id == "1234.5678"
    assert len(p.sections) == 3
    assert len(p.paragraphs) == 4
    assert p.paragraphs[0][0] == 0 and p.paragraphs[2][0] == 1
    assert p.sections[1].name == "Introduction"  # duplicate name preserved as-is
    assert p.figures_and_tables[0]["file"].endswith(".png")
    assert set(p.raw_keys) == {"title", "abstract", "full_text", "figures_and_tables", "qas"}


def test_body_text_excludes_nothing_but_keeps_empty_para():
    p = normalize_paper("1234.5678", RAW, "train")
    body = p.body_text()
    assert "First para." in body and "Second  para." in body


def test_missing_fields_tolerated():
    p = normalize_paper("0000.0000", {"title": "T"}, "dev")
    assert p.sections == [] and p.qas == [] and p.abstract == ""


def test_answer_type_classification():
    assert answer_type({"unanswerable": True}) == "unanswerable"
    assert answer_type({"extractive_spans": ["span"], "free_form_answer": "x"}) == "extractive"
    assert answer_type({"extractive_spans": [], "yes_no": False}) == "yes_no"
    assert answer_type({"yes_no": None, "free_form_answer": " text "}) == "free_form"
    assert answer_type({}) == "empty"


def test_single_type_violation_detected():
    assert non_empty_type_count({"yes_no": True}) == 1
    assert non_empty_type_count({"yes_no": True, "unanswerable": True}) == 2


def test_norm_text_collapses_whitespace_and_nfkc():
    assert norm_text("  a\u00a0 b\n\n c ") == "a b c"
    assert norm_text("") == ""
