from __future__ import annotations

import unicodedata

from rag_data.cleaner import (
    Cleaner,
    CleanConfig,
    decode_segments,
    encode_segments,
    nfkc_is_charwise,
)


def test_clean_removes_refs_and_keeps_offsets():
    c = Cleaner(CleanConfig())
    raw = "BIBREF0 built a model. Results improve on TABREF12 clearly."
    res = c.clean(raw)
    assert "[REF]" in res.text and "[TABLE]" in res.text
    assert "BIBREF" not in res.text
    # every cleaned position must map into the original range
    for pos in range(len(res.text)):
        o = res.to_orig(pos)
        assert 0 <= o <= len(raw)
    # monotonic
    origs = [res.to_orig(p) for p in range(len(res.text) + 1)]
    assert origs == sorted(origs)


def test_identity_when_nothing_to_clean():
    c = Cleaner()
    raw = "A plain sentence about retrieval."
    res = c.clean(raw)
    assert res.text == raw
    assert res.segments == [type(res.segments[0])(0, len(raw), 0, len(raw), "keep")]
    assert encode_segments(res.segments) == ""


def test_nfkc_and_zero_width():
    c = Cleaner()
    raw = "e\ufb03cient\u200b model"
    res = c.clean(raw)
    assert res.text == "efficient model"
    assert "nfkc" in res.hits and "zero_width" in res.hits


def test_hyphen_and_newline_join():
    c = Cleaner()
    raw = "transla-\ntion quality\nis better"
    res = c.clean(raw)
    assert res.text == "translation quality is better"


def test_drop_mode():
    c = Cleaner(CleanConfig(ref_mode="drop"))
    res = c.clean("as shown in BIBREF3 and BIBREF4 .")
    assert "BIBREF" not in res.text
    assert "as shown in and" in res.text.replace("  ", " ")


def test_strip_and_empty():
    c = Cleaner()
    assert c.clean("   ").text == ""
    assert c.clean("  hi  ").text == "hi"


def test_decode_roundtrip():
    c = Cleaner()
    res = c.clean("BIBREF1 says hi.\nNext line.")
    blob = encode_segments(res.segments)
    segs = decode_segments(blob, len(res.text))
    assert segs[0].clean_start == 0
    assert segs[-1].clean_end == len(res.text)
    cover = 0
    for s in segs:
        assert s.clean_start == cover
        cover = s.clean_end


def test_nfkc_charwise_on_sample():
    for s in ["e\ufb03cient", "ＡＢＣ", "ﬁne", "plain text 123"]:
        assert nfkc_is_charwise(s)
    assert nfkc_is_charwise("x" + unicodedata.normalize("NFKC", "\ufb01"))
