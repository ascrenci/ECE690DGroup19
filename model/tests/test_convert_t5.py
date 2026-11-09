import json
import runpy
import sys
import types

import numpy as np
import pytest
from conftest import FakeModel, FakeTokenizer, _arr

import convert_t5

V, D = 8, 4


def test_encoder_wrapper_returns_hidden_states():
    out = convert_t5.EncoderWrapper(FakeModel(d_model=D)).forward(np.zeros((1, 3)), np.ones((1, 3)))
    assert out.shape == (1, 3, D)


@pytest.mark.parametrize("tied, scale", [(False, 1.0), (True, D**-0.5)])
def test_decoder_wrapper_selects_position_and_rescales_when_tied(tied, scale):
    model = FakeModel(d_model=D, tied=tied)
    model.lm_head = lambda h: h  # identity so the scale is visible
    out = convert_t5.DecoderWrapper(model).forward(np.zeros((1, 5)), np.zeros((1, 3, D)), np.ones((1, 3)), np.array([1]))
    assert out.shape == (1, D)
    assert np.allclose(out, scale)


def test_convert_saves_encoder_and_decoder(tmp_path, coreml):
    _, saved = coreml
    convert_t5.convert(FakeModel(d_model=D), 3, 5, tmp_path)
    assert [p.rsplit("T5", 1)[1] for p in saved] == ["Encoder.mlpackage", "Decoder.mlpackage"]


def make_predictor(coreml, peak, tokens):
    """Encoder returns ones; decoder returns `peak` at the next token in `tokens` each call."""
    ct, _ = coreml
    calls = iter(tokens)

    def predict(path, feed):
        if "Encoder" in path:
            return {"last_hidden_state": np.ones((1, 3, D))}
        logits = np.zeros((1, V))
        logits[0, next(calls)] = peak
        return {"logits": logits}

    ct.models.MLModel.predictor = staticmethod(predict)


def run_validate(tmp_path, generated, dec_len=6, tied=False):
    model = FakeModel(vocab=V, d_model=D, tied=tied, generated=generated)
    model.logits = _arr(np.zeros((1, 1, V)))
    convert_t5.validate(model, convert_t5.AutoTokenizer.from_pretrained(), 128, dec_len, tmp_path)


def test_validate_skips_off_macos(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(convert_t5.platform, "system", lambda: "Linux")
    run_validate(tmp_path, (0, 5, 1))
    assert "skipping validation" in capsys.readouterr().out


def test_validate_passes_when_core_ml_matches_pytorch(tmp_path, monkeypatch, coreml, capsys):
    monkeypatch.setattr(convert_t5.platform, "system", lambda: "Darwin")
    make_predictor(coreml, 0.01, [5, 1])  # emits 5, then EOS -> stops early
    run_validate(tmp_path, (0, 5, 1))
    out = capsys.readouterr().out
    assert "Core ML :" in out and "WARNING" not in out


def test_validate_warns_when_greedy_outputs_differ(tmp_path, monkeypatch, coreml, capsys):
    monkeypatch.setattr(convert_t5.platform, "system", lambda: "Darwin")
    make_predictor(coreml, 0.01, [5, 5])  # never emits EOS: runs out of steps
    run_validate(tmp_path, (0, 7, 1), dec_len=3)
    assert "WARNING: greedy outputs differ" in capsys.readouterr().out


def test_validate_fails_when_logits_diverge(tmp_path, monkeypatch, coreml):
    monkeypatch.setattr(convert_t5.platform, "system", lambda: "Darwin")
    make_predictor(coreml, 1.0, [5, 1])
    with pytest.raises(SystemExit, match="validation failed"):
        run_validate(tmp_path, (0, 5, 1))


@pytest.fixture
def recorded(monkeypatch):
    calls = []
    monkeypatch.setattr(convert_t5, "convert", lambda *a: calls.append("convert"))
    monkeypatch.setattr(convert_t5, "validate", lambda *a: calls.append("validate"))
    return calls


@pytest.mark.parametrize("extra, expected", [([], ["convert", "validate"]), (["--no-validate"], ["convert"])])
def test_main_writes_tokenizer_meta_and_converts(tmp_path, monkeypatch, recorded, extra, expected):
    monkeypatch.setattr(sys, "argv", ["convert_t5.py", "--out", str(tmp_path), *extra])
    convert_t5.main()
    assert recorded == expected
    meta = json.loads((tmp_path / "t5_tokenizer" / "t5_meta.json").read_text())
    assert meta["prompt"] == convert_t5.PROMPT
    assert meta["latexDecode"][0] == ["@lt", "<"]
    assert (tmp_path / "t5_tokenizer" / "config.json").exists()
    assert meta["keptIds"] is None


def test_load_plain_checkpoint(tmp_path):
    model, tok, model_tok, kept = convert_t5.load(tmp_path)
    assert isinstance(model, FakeModel) and model_tok is tok and kept is None


@pytest.fixture
def pruned(tmp_path, monkeypatch):
    """A folder with spec.json; fake surgery/vocab modules record what load() asks of them."""
    (tmp_path / "spec.json").write_text("{}")
    model = FakeModel(vocab=V)
    monkeypatch.setattr(convert_t5, "T5ForConditionalGeneration", lambda config: model)
    surgery = types.SimpleNamespace(kept=None)
    surgery.load = lambda m, path: surgery.kept
    vocab = types.SimpleNamespace(Remapped=lambda tok, kept, old: ("remapped", tok, kept, old))
    monkeypatch.setitem(sys.modules, "surgery", surgery)
    monkeypatch.setitem(sys.modules, "vocab", vocab)
    return tmp_path, model, surgery


def test_load_pruned_remaps_tokenizer_over_its_full_vocab(pruned):
    path, model, surgery = pruned
    surgery.kept = [0, 1, 2]
    got, tok, model_tok, kept = convert_t5.load(path)
    assert got is model and kept == [0, 1, 2]
    assert model_tok == ("remapped", tok, [0, 1, 2], 127)


def test_load_pruned_without_vocab_pruning_keeps_tokenizer(pruned):
    path, _, _ = pruned
    _, tok, model_tok, kept = convert_t5.load(path)
    assert model_tok is tok and kept is None


def test_main_writes_kept_ids_and_validates_in_model_id_space(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(convert_t5, "load", lambda path: (FakeModel(), FakeTokenizer(), "model_tok", [0, 1, 2, 9]))
    monkeypatch.setattr(convert_t5, "convert", lambda *a: None)
    monkeypatch.setattr(convert_t5, "validate", lambda model, tok, *a: calls.append(tok))
    monkeypatch.setattr(sys, "argv", ["convert_t5.py", "--out", str(tmp_path)])
    convert_t5.main()
    assert calls == ["model_tok"]
    assert json.loads((tmp_path / "t5_tokenizer" / "t5_meta.json").read_text())["keptIds"] == [0, 1, 2, 9]


def test_runs_as_script(tmp_path, monkeypatch, coreml):
    monkeypatch.setattr(sys, "argv", ["convert_t5.py", "--out", str(tmp_path), "--no-validate"])
    runpy.run_module("convert_t5", run_name="__main__")
    assert len(coreml[1]) == 2
