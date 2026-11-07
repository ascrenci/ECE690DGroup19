import json

import pytest
import torch

import channel
import metrics
import surgery


def test_param_counts_split_embeddings_from_layers(model):
    counts = metrics.param_counts(model)
    emb = 64 * 16 * 2  # embedding + lm_head (untied)
    assert counts["total"] == sum(p.numel() for p in model.parameters())
    assert counts["embeddings"] == emb and counts["layers"] == counts["total"] - emb
    assert counts["prunable"] == surgery.prunable_params(model)


def test_eval_loss_is_token_weighted_and_batch_size_independent(model, tok, rows):
    one = metrics.eval_loss(model, tok, rows, 16, 16, batch_size=1)
    three = metrics.eval_loss(model, tok, rows, 16, 16, batch_size=3)
    assert one > 0 and one == pytest.approx(three, rel=1e-4)


def test_eval_loss_changes_when_model_is_pruned(model, tok, rows):
    before = metrics.eval_loss(model, tok, rows, 16, 16)
    channel.apply(model, {"encoder": [2, 2], "decoder": [2, 2]}, {"encoder": 1, "decoder": 1})
    assert metrics.eval_loss(model, tok, rows, 16, 16) != before


def test_latency_is_positive_and_leaves_model_alone(model, tok, rows):
    ms = metrics.latency_ms(model, tok, rows, "cpu", max_target=4)
    assert ms > 0 and next(model.parameters()).device.type == "cpu"


def test_measure_records_everything(model, tok, rows):
    record = metrics.measure(model, tok, rows, 16, 4, latency_rows=2)
    assert set(record) == {"params", "size_mb", "eval", "latency_ms", "spec"}
    assert record["size_mb"]["fp32"] == record["params"]["total"] * 4 / 1e6
    assert set(record["eval"]) == {"exact_match", "char_similarity", "loss"}
    assert list(record["latency_ms"]) == ["cpu"]  # conftest hides the GPU


def test_measure_adds_cuda_latency_when_available(model, tok, rows, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(metrics, "latency_ms", lambda m, t, r, device, max_target: 1.0)
    assert list(metrics.measure(model, tok, rows, 16, 4)["latency_ms"]) == ["cpu", "cuda"]


def test_write_creates_directories(tmp_path):
    metrics.write(tmp_path / "a" / "b.json", {"x": 1})
    assert json.loads((tmp_path / "a" / "b.json").read_text()) == {"x": 1}
