# Handover: pruning results

Read `AGENTS.md` first (rules for tests, asking before design decisions, concise commits).

## Get the results (tracked in git)

```bash
git pull
```

- `data/comparison.md` (and `.csv`): the table, baseline vs every pruned model.
- `data/metrics/<id>.json`: one file per model (`baseline`, `channel-30`, `channel-50`, `nas-30`, `nas-50`, `vocab`, `combined-50`). Keys: `params` (total, embeddings, layers, prunable), `size_mb` (fp32, fp16), `eval` (exact_match, char_similarity, loss), `latency_ms` (cpu, cuda), `spec` (per-layer FFN widths, heads per stack, vocab size), `recovery` (rows, epochs, lr, seed), plus method extras (`nas`, `kept_tokens`, `channel_from`).
- More runs may add files here later; check `git log -- data`.

## Get the images (not in git)

`images/` is gitignored. The PNGs are attached to a GitHub release:

```bash
mkdir -p images && gh release download results-images-v1 --dir images
```

- `comparison.png`: the results table.
- `project-status.png`: where the project stands against the proposal.

If a newer `results-images-vN` release exists, use the highest N.

## Reproduce a row

1. Model: `gh release download t5-math-v2 --dir model/checkpoints/t5-math-v2` then unzip into `model/checkpoints/t5-math-v2/final`.
2. Data (a few minutes, deterministic with the default seed): `python model/prepare_data.py`. It should print `train 300000  eval 1000`.
3. Run: `python pruning/run.py <baseline|channel|nas|vocab|combined> [--level 0.3|0.5]`, then `python pruning/report.py` to rebuild the table. Needs a CUDA GPU for a sane runtime; `combined` needs the `channel-<level>` and `nas-<level>` metrics first.
4. Tests: `pytest` (model) and `cd pruning && pytest` (pruning, real CPU torch); both must stay at 100%.

## What the results say

- Vocabulary pruning is free: -23% weights, same exact match (41.3%), by dropping tokens the data never uses.
- Channel and NAS pruning cost accuracy (about -3 points at 30%, -7 at 50%). NAS vs channel is within noise: 1000 eval rows and one seed resolve only about 3 points.
- Combined (vocab + channel at 50%) removes 52% of the weights (about 74 MB at fp16) for -7 points; the proposal wants under 5% relative drop and under 50 MB.
- Caveats: sizes are PyTorch parameter counts (Core ML files duplicate the embedding); GPU latency at batch size 1 is launch-bound and not meaningful, so read CPU latency; vocabulary pruning needs an id remap in `T5Translator.swift` that does not exist yet.
