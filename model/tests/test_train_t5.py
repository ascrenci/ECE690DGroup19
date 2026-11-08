import json
import runpy
import sys

import pytest
from conftest import FakeDataset, FakeModel, FakeTrainer

import train_t5

TRAIN = [{"spoken": "x squared", "latex": "x ^ { 2 }", "target": 'x ! " 2 ?'}] * 3
EVAL = [{"spoken": "x", "latex": "x"}, {"spoken": "y", "latex": "y"}]


@pytest.fixture
def data_dir(tmp_path):
    for name, rows in (("train", TRAIN), ("eval", EVAL)):
        (tmp_path / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return tmp_path


@pytest.fixture
def model(monkeypatch):
    m = FakeModel(generated=(ord("x"), 1))
    monkeypatch.setattr(sys.modules["transformers"].T5ForConditionalGeneration, "from_pretrained", lambda *a, **k: m)
    return m


@pytest.fixture
def run(monkeypatch, data_dir, model):
    monkeypatch.setattr(train_t5, "load_dataset", lambda *a, **k: FakeDataset(TRAIN))

    def _run(*extra):
        monkeypatch.setattr(sys, "argv", ["train_t5.py", "--data", str(data_dir), "--out", str(data_dir / "out"), *extra])
        train_t5.main()

    return _run


def test_evaluate_scores_exact_match_and_similarity(model, tok, capsys):
    result = train_t5.evaluate(model, tok, EVAL, max_source=8, max_target=8, batch_size=1)
    assert result["exact_match"] == 0.5
    assert 0.5 <= result["char_similarity"] < 1.0
    out = capsys.readouterr().out
    assert "OK " in out and "BAD" in out


def test_eval_only_does_not_train(run, capsys):
    FakeTrainer.last = None
    run("--eval-only")
    assert FakeTrainer.last is None
    assert "eval on 2 rows" in capsys.readouterr().out


def test_train_uses_all_rows_without_cuda(run, model):
    run()
    args = FakeTrainer.last.kw["args"]
    assert args.bf16 is False and args.fp16 is False
    assert len(FakeTrainer.last.kw["train_dataset"]) == 3
    assert model.cuda_device is None
    assert FakeTrainer.last.resume is None
    assert FakeTrainer.last.saved.endswith("final")


def test_train_with_row_limit_resume_and_cuda(run, model, monkeypatch):
    cuda = sys.modules["torch"].cuda
    monkeypatch.setattr(cuda, "is_available", lambda: True)
    monkeypatch.setattr(cuda, "is_bf16_supported", lambda: True)
    run("--max-train-rows", "2", "--resume")
    assert len(FakeTrainer.last.kw["train_dataset"]) == 2
    assert FakeTrainer.last.kw["args"].bf16 is True
    assert FakeTrainer.last.resume is True
    assert model.cuda_device == "cuda"


def test_runs_as_script(monkeypatch, data_dir, model):
    monkeypatch.setattr(sys, "argv", ["train_t5.py", "--eval-only", "--data", str(data_dir)])
    runpy.run_module("train_t5", run_name="__main__")


def test_fit_passes_accumulation_and_extra_kwargs_to_a_custom_trainer(model, tok, tmp_path):
    class Custom(FakeTrainer):
        pass

    trainer = train_t5.fit(model, tok, FakeDataset(TRAIN), tmp_path, 1, 2, 1e-4, 8, 8, grad_accum=3,
                           trainer_cls=Custom, teacher="T")
    assert isinstance(trainer, Custom) and trainer.kw["teacher"] == "T"
    assert trainer.kw["args"].gradient_accumulation_steps == 3
