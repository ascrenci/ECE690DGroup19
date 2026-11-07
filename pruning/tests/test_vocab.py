import pytest
import torch

import surgery
import vocab
from conftest import logits


def test_kept_ids_are_sorted_specials_plus_used_tokens(tok, rows):
    ids = vocab.kept_ids(tok, rows, batch=2)  # batch < len(rows) exercises the chunk loop
    assert ids == sorted(set(ids)) and ids[:3] == [0, 1, 2]
    for word in ("Convert", "squared", "equals", "!"):
        assert tok.convert_tokens_to_ids(word) in ids
    assert tok.convert_tokens_to_ids("r") in ids and tok.convert_tokens_to_ids("$") not in ids


def test_remapped_maps_in_and_out(tok):
    kept = [0, 1, 2, tok.convert_tokens_to_ids("x"), tok.convert_tokens_to_ids("squared")]
    r = vocab.Remapped(tok, kept, 64)
    assert r("x squared").input_ids == [3, 4]
    assert r("x plus").input_ids == [3, 2]  # "plus" is not kept -> <unk>
    assert r(["x", "squared x"]).input_ids == [[3], [4, 3]]
    pt = r(["x squared"], return_tensors="pt").input_ids
    assert pt.tolist() == [[3, 4]]
    assert r.decode([3, 4]) == "x squared" and r.batch_decode(torch.tensor([[3, 4], [4, 3]])) == ["x squared", "squared x"]
    assert r.pad_token_id == 0  # other attributes pass through


def test_remapped_requires_specials_first(tok):
    with pytest.raises(AssertionError, match="specials"):
        vocab.Remapped(tok, [5, 6, 7], 64)


def test_apply_prunes_model_and_returns_matching_tokenizer(model, tok):
    kept = [0, 1, 2, tok.convert_tokens_to_ids("x"), tok.convert_tokens_to_ids("squared")]
    wrapped = vocab.apply(model, tok, kept)
    assert model.config.vocab_size == 5 and isinstance(wrapped, vocab.Remapped)
    out = logits(model, (3, 4, 1), (0, 3))[0, -1]
    assert out.shape == (5,) and surgery.spec_of(model)["vocab"] == 5
