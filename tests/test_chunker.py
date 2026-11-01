from __future__ import annotations

from rag_data.chunker import ChunkConfig, chunk_paper, section_path_of, split_sentences


class FakeCounter:
    """Whitespace token counter: deterministic, no tokenizer download needed in CI."""

    backend = "fake:whitespace"

    def count(self, text: str) -> int:
        return len(text.split())

    def char_offsets(self, text: str):
        out, pos = [], 0
        for w in text.split(" "):
            out.append((pos, pos + len(w)))
            pos += len(w) + 1
        return out


C = FakeCounter()


def _para(pid, sec_idx, name, text, is_abs=False):
    return {
        "para_id": pid,
        "section_idx": sec_idx,
        "section_name": name,
        "text_clean": text,
        "is_abstract": is_abs,
    }


def _paper():
    return [
        _para("p#p0000", -1, "Abstract", "This is the abstract. It has two sentences.", True),
        _para("p#p0001", 0, "Introduction", "Alpha beta gamma. Delta epsilon.", False),
        _para("p#p0002", 0, "Introduction", "Zeta eta theta. Iota kappa.", False),
        _para("p#p0003", 1, "Experiments ::: Ablation", "Lambda mu. Nu xi omicron.", False),
    ]


def test_basic_chunking_and_ids():
    rows = chunk_paper(_paper(), ChunkConfig(chunk_size=12, min_chunk_tokens=1), C, "p", "dev")
    assert rows, "should produce chunks"
    assert rows[0].chunk_id == "p#sabs#c000"
    assert all(r.chunk_id.startswith("p#s") for r in rows)
    assert len({r.chunk_id for r in rows}) == len(rows)


def test_token_budget_respected():
    cfg = ChunkConfig(chunk_size=10, min_chunk_tokens=1)
    rows = chunk_paper(_paper(), cfg, C, "p", "dev")
    for r in rows:
        assert r.n_tokens <= cfg.chunk_size, (r.chunk_id, r.n_tokens)


def test_hard_section_boundary_never_crosses():
    cfg = ChunkConfig(chunk_size=100, section_boundary="hard", min_chunk_tokens=1)
    rows = chunk_paper(_paper(), cfg, C, "p", "dev")
    assert all(r.n_sections == 1 for r in rows)


def test_soft_boundary_can_merge_tiny_sections():
    rows = [
        _para("p#p0000", 0, "Short", "Only one short sentence.", False),
        _para("p#p0001", 1, "Other", "Second tiny section here.", False),
    ]
    cfg = ChunkConfig(chunk_size=100, section_boundary="soft", min_chunk_tokens=10,
                      include_abstract=False)
    out = chunk_paper(rows, cfg, C, "p", "dev")
    assert len(out) == 1 and out[0].n_sections == 2


def test_paragraph_provenance_is_complete():
    cfg = ChunkConfig(chunk_size=12, min_chunk_tokens=1)
    rows = chunk_paper(_paper(), cfg, C, "p", "dev")
    covered: dict[str, list[tuple[int, int]]] = {}
    for r in rows:
        for pid, a, b in r.para_spans:
            covered.setdefault(pid, []).append((a, b))
    for pid, spans in covered.items():
        spans.sort()
        assert spans[0][0] == 0, pid
        for (a1, b1), (a2, b2) in zip(spans, spans[1:]):
            assert a2 <= b1 + 1, (pid, a1, b1, a2, b2)  # contiguous (overlap allowed)


def test_overlap_shares_text_between_neighbours():
    cfg = ChunkConfig(chunk_size=8, overlap_ratio=0.5, min_chunk_tokens=1)
    rows = chunk_paper(_paper(), cfg, C, "p", "dev")
    same_sec = [r for r in rows if r.sec_key == "00"]
    assert len(same_sec) >= 2
    a, b = same_sec[0], same_sec[1]
    assert a.char_end > b.char_start, "expected overlapping character ranges"


def test_determinism():
    cfg = ChunkConfig(chunk_size=12, overlap_ratio=0.2, min_chunk_tokens=1)
    a = chunk_paper(_paper(), cfg, C, "p", "dev")
    b = chunk_paper(_paper(), cfg, C, "p", "dev")
    assert [r.chunk_id for r in a] == [r.chunk_id for r in b]
    assert [r.text for r in a] == [r.text for r in b]


def test_neighbour_links_consistent():
    cfg = ChunkConfig(chunk_size=12, min_chunk_tokens=1)
    rows = chunk_paper(_paper(), cfg, C, "p", "dev")
    for x, y in zip(rows, rows[1:]):
        assert x.next_chunk_id == y.chunk_id
        assert y.prev_chunk_id == x.chunk_id
    assert rows[0].prev_chunk_id is None and rows[-1].next_chunk_id is None


def test_oversized_sentence_is_hard_split():
    long_text = " ".join(f"w{i}" for i in range(40)) + "."
    rows = [_para("p#p0000", 0, "S", long_text, False)]
    cfg = ChunkConfig(chunk_size=10, min_chunk_tokens=1, include_abstract=False)
    out = chunk_paper(rows, cfg, C, "p", "dev")
    assert all(r.n_tokens <= cfg.chunk_size for r in out)
    assert any(r.is_truncated for r in out)


def test_empty_paper():
    assert chunk_paper([], ChunkConfig(), C, "p", "dev") == []
    blank = [_para("p#p0000", 0, "S", "   ", False)]
    assert chunk_paper(blank, ChunkConfig(), C, "p", "dev") == []


def test_config_validation():
    import pytest

    with pytest.raises(ValueError):
        ChunkConfig(chunk_size=100, overlap_ratio=1.0).validate()
    with pytest.raises(ValueError):
        ChunkConfig(chunk_size=100, min_chunk_tokens=100).validate()
    with pytest.raises(ValueError):
        ChunkConfig(section_boundary="nope").validate()


def test_sentence_split_covers_text():
    t = "First sentence. Second one! Third? Done."
    spans = split_sentences(t)
    assert spans[0][0] == 0 and spans[-1][1] == len(t)
    assert len(spans) >= 3


def test_section_path():
    assert section_path_of("Experiments ::: Ablation") == "Experiments > Ablation"
    assert section_path_of("Introduction") == "Introduction"


def test_para_spans_one_entry_per_paragraph():
    """Contract invariant: ``len(para_spans) == len(para_ids)``, one span each.

    Regression for a real defect: spans were appended per *sentence*, so a four-sentence
    paragraph produced four identical tuples and every downstream consumer that trusted the
    pairing saw duplicated evidence.
    """
    long_para = " ".join(["Alpha beta gamma.", "Delta epsilon zeta.", "Eta theta iota.",
                          "Kappa lambda mu."])
    rows = [_para("p#p0000", 0, "S", long_para, False)]
    cfg = ChunkConfig(chunk_size=8, overlap_ratio=0.0, min_chunk_tokens=1,
                      include_abstract=False)
    for c in chunk_paper(rows, cfg, C, "p", "dev"):
        assert len(c.para_spans) == len(c.para_ids)
        assert [p for p, _, _ in c.para_spans] == c.para_ids


def test_para_spans_offsets_are_relative_and_within_bounds():
    rows = _paper()
    cfg = ChunkConfig(chunk_size=12, overlap_ratio=0.2, min_chunk_tokens=1)
    by_id = {r["para_id"]: r["text_clean"] for r in rows}
    for c in chunk_paper(rows, cfg, C, "p", "dev"):
        for pid, a, b in c.para_spans:
            assert 0 <= a < b <= len(by_id[pid])
            # the span must equal the chunk's slice of that paragraph's text
            assert c.text.find(by_id[pid][a:b]) >= 0

