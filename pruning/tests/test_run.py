import json
import runpy
import sys

import pytest
from transformers import AutoTokenizer, T5ForConditionalGeneration

import run
import surgery
import train_t5


@pytest.fixture
def env(tmp_path, model, tok, rows, monkeypatch, make_model):
    """Tiny data on disk, tiny model/tokenizer instead of from_pretrained, and fit() recorded instead of run."""
    data = tmp_path / "data"
    data.mkdir()
    for name in ("train", "eval"):
        (data / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: tok)
    monkeypatch.setattr(T5ForConditionalGeneration, "from_pretrained", lambda *a, **k: make_model())
    fits = []
    monkeypatch.setattr(train_t5, "fit", lambda *a, **k: fits.append((a, k)))
    monkeypatch.setattr(run, "PROXY_ROWS", 2)
    flags = ["--data", str(data), "--metrics", str(tmp_path / "m"), "--models", str(tmp_path / "models"),
             "--recover-rows", "2"]

    def go(method, *extra):
        run.main([method, *flags, *extra])
        return json.loads((tmp_path / "m" / f"{run.run_id(method, float(extra[1]) if extra else 0.5)}.json").read_text())

    go.fits, go.tmp, go.flags = fits, tmp_path, flags
    return go


def test_read_rows_and_run_id(tmp_path):
    (tmp_path / "r.jsonl").write_text('{"a": 1}\n{"a": 2}\n')
    assert run.read_rows(tmp_path / "r.jsonl") == [{"a": 1}, {"a": 2}]
    assert [run.run_id(m, 0.3) for m in ("baseline", "vocab", "channel", "nas", "combined")] == [
        "baseline", "vocab", "channel-30", "nas-30", "combined-30"]


def test_baseline_measures_without_recovery_or_saving(env):
    record = env("baseline")
    assert record["id"] == "baseline" and record["recovery"] is None and record["level"] is None
    assert env.fits == [] and not (env.tmp / "models").exists()


def test_channel_prunes_recovers_measures_and_saves(env):
    record = env("channel", "--level", "0.5")
    assert record["spec"]["heads"] == {"encoder": 2, "decoder": 2} and record["spec"]["ffn"]["encoder"] == [16, 16]
    assert record["recovery"]["rows"] == 2 and len(env.fits) == 1
    (a, kw) = env.fits[0]
    assert kw == {"save": False} and len(a[2]) == 2  # recovery dataset has --recover-rows rows
    assert (env.tmp / "models" / "channel-50" / "spec.json").exists()


def test_nas_records_search_stats(env):
    record = env("nas", "--level", "0.5")
    assert record["nas"]["evaluations"] > 0 and record["nas"]["proxy_loss"] > 0
    assert record["params"]["prunable"] > 0


def test_vocab_prunes_embeddings_and_saves_kept_ids(env):
    record = env("vocab")
    assert record["spec"]["vocab"] == record["kept_tokens"] < 64
    saved = json.loads((env.tmp / "models" / "vocab" / "spec.json").read_text())
    assert saved["kept_ids"][:3] == [0, 1, 2] and len(saved["kept_ids"]) == record["kept_tokens"]


def test_combined_uses_the_better_channel_or_nas_result(env):
    channel, nas = env("channel", "--level", "0.5"), env("nas", "--level", "0.5")
    best = max((channel, nas), key=lambda r: r["eval"]["exact_match"])
    record = env("combined", "--level", "0.5")
    assert record["channel_from"] == best["id"] and record["id"] == "combined-50"
    assert record["spec"]["ffn"] == best["spec"]["ffn"] and record["spec"]["vocab"] == record["kept_tokens"]


def test_runs_as_script(env, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run.py", "baseline", *env.flags])
    runpy.run_module("run", run_name="__main__")
    assert (env.tmp / "m" / "baseline.json").exists()
