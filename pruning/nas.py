"""Method 2: NAS pruning. One-shot search over per-layer FFN widths and per-stack head counts.

Channels and heads are ordered by magnitude (channel.apply), so every sub-network is "top-k" of the
trained model and can be scored with no training. Candidates are scored by loss on proxy rows that
are NOT the eval set. Search is evolutionary: mutate good candidates, repair/fill to the budget.
"""
import copy
import json

import channel
import metrics
from surgery import STACKS

RATIOS = (0.25, 0.5, 0.75, 1.0)


def grid(cfg):
    return [int(r * cfg.d_ff) for r in RATIOS]


def cost(cfg, ffn, heads):
    """Attention + FFN weight count of a candidate (matches surgery.prunable_params)."""
    total = 0
    for stack, attns in (("encoder", 1), ("decoder", 2)):
        attn = attns * 4 * cfg.d_model * heads[stack] * cfg.d_kv
        total += sum(3 * cfg.d_model * w + attn for w in ffn[stack])
    return total


def _knobs(cfg, cand, step):
    """Layer/head settings that can move one step (step=-1 shrink, +1 grow): (stack, layer or None)."""
    g = grid(cfg)
    out = []
    for stack in STACKS:
        for i, w in enumerate(cand["ffn"][stack]):
            if 0 <= g.index(w) + step < len(g):
                out.append((stack, i))
        if 1 <= cand["heads"][stack] + step <= cfg.num_heads:
            out.append((stack, None))
    return out


def _move(cfg, cand, knob, step):
    stack, i = knob
    if i is None:
        cand["heads"][stack] += step
    else:
        g = grid(cfg)
        cand["ffn"][stack][i] = g[g.index(cand["ffn"][stack][i]) + step]


def fit_budget(cfg, cand, budget, rng):
    """Shrink random knobs until within budget, then grow random knobs while it still fits."""
    while cost(cfg, **cand) > budget:
        knobs = _knobs(cfg, cand, -1)
        if not knobs:
            raise ValueError("budget is below the smallest architecture")
        _move(cfg, cand, rng.choice(knobs), -1)
    while True:
        fits = []
        for knob in _knobs(cfg, cand, +1):
            trial = copy.deepcopy(cand)
            _move(cfg, trial, knob, +1)
            if cost(cfg, **trial) <= budget:
                fits.append(knob)
        if not fits:
            return cand
        _move(cfg, cand, rng.choice(fits), +1)


def random_candidate(cfg, rng):
    g = grid(cfg)
    return {
        "ffn": {"encoder": [rng.choice(g) for _ in range(cfg.num_layers)],
                "decoder": [rng.choice(g) for _ in range(cfg.num_decoder_layers)]},
        "heads": {s: rng.randint(1, cfg.num_heads) for s in STACKS},
    }


def mutate(cfg, cand, rng):
    child = copy.deepcopy(cand)
    g = grid(cfg)
    for _ in range(rng.randint(1, 3)):
        stack = rng.choice(STACKS)
        if rng.random() < 0.25:
            child["heads"][stack] = rng.randint(1, cfg.num_heads)
        else:
            child["ffn"][stack][rng.randrange(len(child["ffn"][stack]))] = rng.choice(g)
    return child


def search(cfg, budget, score_fn, rng, population=16, generations=12, offspring=8):
    """Lowest-score candidate within budget: (candidate, score, number of distinct evaluations)."""
    seen = {}

    def score(cand):
        key = json.dumps(cand, sort_keys=True)
        if key not in seen:
            seen[key] = (score_fn(cand), cand)
        return seen[key][0]

    pop = [fit_budget(cfg, random_candidate(cfg, rng), budget, rng) for _ in range(population)]
    for _ in range(generations):
        pop.sort(key=score)
        parents = pop[: population // 2]
        pop = pop[:population] + [fit_budget(cfg, mutate(cfg, rng.choice(parents), rng), budget, rng)
                                  for _ in range(offspring)]
    best = min(pop, key=score)
    return best, score(best), len(seen)


def make_score(model, tok, rows, max_source=127, max_target=127):
    """Score a candidate: prune a copy of `model` and return its loss on `rows`."""
    def score(cand):
        trial = copy.deepcopy(model)
        channel.apply(trial, cand["ffn"], cand["heads"])
        return metrics.eval_loss(trial, tok, rows, max_source, max_target)

    return score
