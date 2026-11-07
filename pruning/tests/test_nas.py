import copy
import random

import pytest

import channel
import nas
import surgery


def test_cost_matches_actual_prunable_params(model):
    cand = {"ffn": {"encoder": [8, 32], "decoder": [16, 24]}, "heads": {"encoder": 2, "decoder": 3}}
    cfg = model.config
    expected = nas.cost(cfg, **cand)
    channel.apply(model, cand["ffn"], cand["heads"])
    assert surgery.prunable_params(model) == expected


def test_random_candidate_and_mutate_stay_in_search_space(model):
    cfg, rng = model.config, random.Random(0)
    g = nas.grid(cfg)
    cand = nas.random_candidate(cfg, rng)
    child = nas.mutate(cfg, cand, rng)
    for c in (cand, child):
        assert all(w in g for s in surgery.STACKS for w in c["ffn"][s])
        assert all(1 <= c["heads"][s] <= cfg.num_heads for s in surgery.STACKS)
    assert child is not cand


def test_fit_budget_shrinks_then_fills(model):
    cfg, rng = model.config, random.Random(1)
    full = {"ffn": {s: [cfg.d_ff] * 2 for s in surgery.STACKS}, "heads": {s: cfg.num_heads for s in surgery.STACKS}}
    budget = nas.cost(cfg, **full) // 2
    fitted = nas.fit_budget(cfg, copy.deepcopy(full), budget, rng)
    cost = nas.cost(cfg, **fitted)
    assert cost <= budget
    # filled: no single knob can still grow within the budget
    for knob in nas._knobs(cfg, fitted, +1):
        trial = copy.deepcopy(fitted)
        nas._move(cfg, trial, knob, +1)
        assert nas.cost(cfg, **trial) > budget


def test_fit_budget_rejects_impossible_budget(model):
    cfg = model.config
    cand = nas.random_candidate(cfg, random.Random(0))
    with pytest.raises(ValueError, match="smallest"):
        nas.fit_budget(cfg, cand, 1, random.Random(0))


def test_search_finds_candidate_within_budget_and_caches(model):
    cfg = model.config
    budget = nas.cost(cfg, *channel.uniform(model, 0.5))
    calls = []

    def score(cand):
        calls.append(1)
        return -cand["heads"]["encoder"] - sum(cand["ffn"]["decoder"]) / 100  # prefers heads + decoder width

    best, best_score, evals = nas.search(cfg, budget, score, random.Random(0), population=6, generations=3, offspring=3)
    assert nas.cost(cfg, **best) <= budget
    assert len(calls) == evals and evals <= 6 + 3 * 3
    assert best_score == score(best)


def test_make_score_returns_loss_of_pruned_copy(model, tok, rows):
    score = nas.make_score(model, tok, rows, 16, 16)
    cand = {"ffn": {s: [16, 16] for s in surgery.STACKS}, "heads": {s: 2 for s in surgery.STACKS}}
    first = score(cand)
    assert first > 0 and score(cand) == first
    assert surgery.spec_of(model)["heads"] == {"encoder": 4, "decoder": 4}  # original untouched
