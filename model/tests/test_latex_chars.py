import pytest
from conftest import FakeTokenizer, UNK

import latex_chars as lc


class AllProducible(FakeTokenizer):
    def _ids(self, text):
        return [ord(c) for c in text]


class NoQuestionMark(FakeTokenizer):
    def _ids(self, text):
        return [UNK if c == "?" else ord(c) for c in text]


class NoX(FakeTokenizer):
    def _ids(self, text):
        return [UNK if c == "x" else ord(c) for c in text]


class DropsSpaces(FakeTokenizer):
    def decode(self, ids, **kw):
        return super().decode(ids, **kw).replace(" ", "")


def test_normalize_strips_dollars_and_collapses_whitespace():
    assert lc.normalize("$ a  b $") == "a b"
    assert lc.normalize("a\n b") == "a b"


def test_normalize_spoken_keeps_case_and_drops_trailing_punctuation():
    assert lc.normalize_spoken("  Xi   star.  ") == "Xi star"


@pytest.mark.parametrize(
    "latex, encoded",
    [
        (r"\frac{a}{b}^2", '@frac"a?"b?!2'),
        ("x < y", "x @lt y"),
        ("a~b", "a@nbspb"),
        ("a`b", "a@bqb"),
    ],
)
def test_encode_and_decode_round_trip(latex, encoded):
    assert lc.encode(latex) == encoded
    assert lc.decode(encoded) == lc.normalize(latex)


@pytest.mark.parametrize("bad", ["   ", "a@b", 'a"b', "a?b", "a!b", "é"])
def test_encode_rejects_unrepresentable(bad):
    with pytest.raises(ValueError):
        lc.encode(bad)


def test_check_tokenizer_accepts_t5_like_vocab(capsys):
    lc.check_tokenizer(FakeTokenizer())
    assert capsys.readouterr().out == ""


def test_check_tokenizer_notes_when_vocab_can_produce_unproducible(capsys):
    lc.check_tokenizer(AllProducible())
    assert "can now produce" in capsys.readouterr().out


def test_check_tokenizer_raises_when_stand_in_is_unproducible():
    with pytest.raises(AssertionError, match="stand-in"):
        lc.check_tokenizer(NoQuestionMark())


def test_roundtrip_ok_true_for_normal_equation():
    assert lc.roundtrip_ok(r"\frac{a}{b}", FakeTokenizer())


def test_roundtrip_ok_false_when_encode_fails():
    assert not lc.roundtrip_ok("a@b", FakeTokenizer())


def test_roundtrip_ok_false_on_unknown_token():
    assert not lc.roundtrip_ok("x + 1", NoX())


def test_roundtrip_ok_false_when_decode_differs():
    assert not lc.roundtrip_ok("a b", DropsSpaces())
