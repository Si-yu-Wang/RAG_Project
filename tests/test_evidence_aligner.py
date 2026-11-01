from __future__ import annotations

from rag_data.evidence_aligner import AlignConfig, PaperIndex, align_evidence, align_paper, norm


def _rows():
    return [
        {"para_id": "p0", "text_raw": "Intro para about retrieval.", "section_name": "Introduction"},
        {"para_id": "p1", "text_raw": "Method para one.", "section_name": "Experiments ::: Ablation"},
        {"para_id": "p2", "text_raw": "Method para two.", "section_name": "Experiments ::: Ablation"},
        {"para_id": "p3", "text_raw": "Other  spaced   text.", "section_name": "Conclusion"},
    ]


def test_exact_paragraph_hit():
    idx = PaperIndex.build(_rows())
    lvl, pids, _ = align_evidence("Method para one.", idx, AlignConfig())
    assert lvl == "paragraph" and pids == ["p1"]


def test_normalised_hit():
    idx = PaperIndex.build(_rows())
    lvl, pids, _ = align_evidence("Other spaced text.", idx, AlignConfig())
    assert lvl == "paragraph_norm" and pids == ["p3"]


def test_section_rule_expands_to_all_paragraphs():
    idx = PaperIndex.build(_rows())
    lvl, pids, _ = align_evidence("Experiments ::: Ablation", idx, AlignConfig())
    assert lvl == "section"
    assert pids == ["p1", "p2"]


def test_section_rule_can_be_disabled():
    idx = PaperIndex.build(_rows())
    lvl, pids, _ = align_evidence("Experiments ::: Ablation", idx, AlignConfig(section_rule=False))
    assert lvl == "fail" and pids == []


def test_float_detected():
    idx = PaperIndex.build(_rows())
    lvl, pids, _ = align_evidence("FLOAT SELECTED: Table 2 results", idx, AlignConfig())
    assert lvl == "float" and pids == []


def test_empty_evidence():
    idx = PaperIndex.build(_rows())
    assert align_evidence("", idx, AlignConfig())[0] == "empty"
    assert align_evidence(None, idx, AlignConfig())[0] == "empty"  # type: ignore[arg-type]


def test_norm_is_stable():
    assert norm("  A \u00a0 b\n\n c ") == "A b c"


def test_align_paper_builds_gold_and_reasons():
    qas = [
        {  # two annotators, overlapping evidence -> union without duplicates
            "question_id": "q1",
            "question": "What?",
            "answers": [
                {"answer": {"evidence": ["Method para one."], "extractive_spans": ["one"],
                            "free_form_answer": "", "yes_no": None, "unanswerable": False}},
                {"answer": {"evidence": ["Method para one.", "Method para two."],
                            "extractive_spans": [], "free_form_answer": "x",
                            "yes_no": None, "unanswerable": False}},
            ],
        },
        {  # no evidence at all
            "question_id": "q2",
            "question": "Do they?",
            "answers": [{"answer": {"evidence": [], "yes_no": True, "unanswerable": False}}],
        },
        {  # float only
            "question_id": "q3",
            "question": "Table?",
            "answers": [{"answer": {"evidence": ["FLOAT SELECTED: Table 1"],
                                    "free_form_answer": "y", "unanswerable": False}}],
        },
    ]
    recs, levels = align_paper(qas, _rows(), "dev", AlignConfig())
    by_id = {r["question_id"]: r for r in recs}
    assert by_id["q1"]["gold_para_ids"] == ["p1", "p2"]
    assert by_id["q1"]["n_gold_paras"] == 2
    assert by_id["q1"]["answer_types"] == "extractive,free_form"
    assert by_id["q2"]["zero_gold_reason"] == "no_evidence"
    assert by_id["q3"]["zero_gold_reason"] == "float_only"
    assert by_id["q3"]["is_float_only"] is True
    assert levels["float"] == 1
