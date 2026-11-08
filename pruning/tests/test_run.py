import json
import runpy
import sys
from types import SimpleNamespace

import pytest
from transformers import AutoTokenizer, T5ForConditionalGeneration

import distill
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
        return run.main([method, *flags, *extra])

    go.fits, go.tmp, go.flags = fits, tmp_path, flags
    return go


def args(method, level=0.5, only=None, tag=None, keep=4):
    return SimpleNamespace(method=method, level=level, only=only, tag=tag, keep_decoder=keep)


def test_read_rows_and_load_eval(tmp_path):
    (tmp_path / "eval.jsonl").write_text('{"a": 1}\n')
    assert run.read_rows(tmp_path / "eval.jsonl") == [{"a": 1}]
    assert run.load_eval(tmp_path) == [{"a": 1}]
    (tmp_path / "eval_extra.jsonl").write_text('{"a": 2}\n{"a": 3}\n')
    assert run.load_eval(tmp_path) == [{"a": 1}, {"a": 2}, {"a": 3}]


def test_run_id_covers_every_method_variant():
    assert run.run_id(args("baseline")) == "baseline" and run.run_id(args("vocab")) == "vocab"
    assert run.run_id(args("channel", 0.3)) == "channel-30" and run.run_id(args("nas", 0.5)) == "nas-50"
    assert run.run_id(args("channel", 0.3, only="ffn")) == "channel-ffn-30"
    assert run.run_id(args("layers", keep=6)) == "layers-dec6"
    assert run.run_id(args("combined", 0.5, tag="distill")) == "combined-50-distill"


def test_baseline_measures_without_recovery_or_saving(env):
    record = env("baseline")
    assert record["id"] == "baseline" and record["recovery"] is None and record["level"] is None
    assert env.fits == [] and not (env.tmp / "models").exists()


def test_channel_prunes_recovers_measures_and_saves(env):
    record = env("channel", "--level", "0.5")
    assert record["spec"]["heads"] == {"encoder": 2, "decoder": 2} and record["spec"]["ffn"]["encoder"] == [16, 16]
    assert record["recovery"]["rows"] == 2 and record["recovery"]["distill"] is None and len(env.fits) == 1
    (a, kw) = env.fits[0]
    assert kw == {"save": False, "grad_accum": 1} and len(a[2]) == 2 and a[4] == 1.0  # rows, epochs
    assert (env.tmp / "models" / "channel-50" / "spec.json").exists()


@pytest.mark.parametrize("only, heads, ffn", [("ffn", 4, 16), ("heads", 2, 32)])
def test_channel_only_prunes_just_one_part(env, only, heads, ffn):
    record = env("channel", "--level", "0.5", "--only", only)
    assert record["id"] == f"channel-{only}-50" and record["only"] == only
    assert record["spec"]["heads"]["encoder"] == heads and record["spec"]["ffn"]["encoder"] == [ffn, ffn]


def test_nas_records_search_stats(env):
    record = env("nas", "--level", "0.5")
    assert record["nas"]["evaluations"] > 0 and record["nas"]["proxy_loss"] > 0
    assert record["params"]["prunable"] > 0


def test_vocab_prunes_embeddings_and_saves_kept_ids(env):
    record = env("vocab")
    assert record["spec"]["vocab"] == record["kept_tokens"] < 64
    saved = json.loads((env.tmp / "models" / "vocab" / "spec.json").read_text())
    assert saved["kept_ids"][:3] == [0, 1, 2] and len(saved["kept_ids"]) == record["kept_tokens"]


def test_layers_drops_decoder_blocks_by_influence(env):
    record = env("layers", "--keep-decoder", "1")
    assert record["id"] == "layers-dec1" and record["level"] is None
    assert record["layers"]["decoder_kept"] == [0] and len(record["layers"]["influence"]) == 2
    assert record["spec"]["ffn"] == {"encoder": [32, 32], "decoder": [32]}


def test_combined_uses_the_better_channel_or_nas_result(env):
    channel, nas = env("channel", "--level", "0.5"), env("nas", "--level", "0.5")
    best = max((channel, nas), key=lambda r: r["eval"]["exact_match"])
    record = env("combined", "--level", "0.5")
    assert record["channel_from"] == best["id"] and record["id"] == "combined-50"
    assert record["spec"]["ffn"] == best["spec"]["ffn"] and record["spec"]["vocab"] == record["kept_tokens"]


def test_long_distilled_recovery_passes_teacher_epochs_and_accumulation(env):
    record = env("vocab", "--epochs", "2", "--distill", "--grad-accum", "2", "--tag", "distill")
    (a, kw) = env.fits[0]
    assert record["id"] == "vocab-distill" and a[4] == 2.0
    assert kw["trainer_cls"] is distill.DistillTrainer and kw["grad_accum"] == 2
    assert kw["kept"][:3] == [0, 1, 2] and kw["alpha"] == 0.5 and kw["temp"] == 2.0
    assert next(kw["teacher"].parameters()).shape[0] == 64  # the unpruned model: full vocab
    assert record["recovery"]["distill"] == {"alpha": 0.5, "temp": 2.0} and record["recovery"]["grad_accum"] == 2


@pytest.mark.parametrize("method, extra", [("channel", ["--level", "0.5"]), ("vocab", []), ("layers", ["--keep-decoder", "1"])])
def test_reeval_reproduces_a_saved_model_and_uses_extra_eval_rows(env, method, extra):
    first = env(method, *extra)
    path = env.tmp / "m" / f"{first['id']}.json"

    run.main(["reeval", "--id", first["id"], *env.flags])  # same weights reloaded from disk, same rows
    same = json.loads(path.read_text())
    assert same["eval"] == pytest.approx(first["eval"])

    (env.tmp / "data" / "eval_extra.jsonl").write_text(
        json.dumps({"spoken": "x plus y", "latex": "x + y", "target": "x + y"}) + "\n")
    more = run.main(["reeval", "--id", first["id"], *env.flags])
    assert more["eval"]["rows"] == first["eval"]["rows"] + 1
    assert (more["id"], more["spec"], more["recovery"]) == (first["id"], first["spec"], first["recovery"])


def test_runs_as_script(env, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run.py", "baseline", *env.flags])
    runpy.run_module("run", run_name="__main__")
    assert (env.tmp / "m" / "baseline.json").exists()
    assert surgery.STACKS == ("encoder", "decoder")
