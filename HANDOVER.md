# Handover: pruning results

Read `AGENTS.md` first (rules for tests, asking before design decisions, concise commits).

## Get the results (tracked in git)

```bash
git pull
```

- `data/comparison.md` (and `.csv`): the table, baseline vs every pruned model.
- `data/metrics/<id>.json`: one file per model (`baseline`, `channel-{30,50}`, `channel-ffn-{30,50}`, `channel-heads-{30,50}`, `nas-{30,50}`, `layers-dec{6,4}`, `vocab`, `combined-50`, `combined-50-long`, `combined-50-distill`). Keys: `params` (total, embeddings, layers, prunable), `size_mb` (fp32, fp16), `eval` (exact_match, char_similarity, loss), `latency_ms` (cpu, cuda), `spec` (per-layer FFN widths, heads per stack, vocab size), `recovery` (rows, epochs, lr, batch_size, grad_accum, seed, distill), `eval.rows` (5000), plus method extras (`nas`, `kept_tokens`, `channel_from`, `layers`).
- More runs may add files here later; check `git log -- data`.

## Get the images (not in git)

`images/` is gitignored. The PNGs are attached to a GitHub release:

```bash
mkdir -p images && gh release download results-images-v2 --dir images
```

- `comparison.png`: the results table.
- `project-status.png`: where the project stands against the proposal.

(v2 is only published once approved; until then regenerate or ask the owner.) If a newer `results-images-vN` release exists, use the highest N.

## Reproduce a row

1. Model: `gh release download t5-math-v2 --dir model/checkpoints/t5-math-v2` then unzip into `model/checkpoints/t5-math-v2/final`.
2. Data (a few minutes, deterministic with the default seed): `python model/prepare_data.py --extra-eval-rows 4000`. It should print `train 300000  eval 1000` and write `eval_extra.jsonl`; eval = 5000 rows total.
3. Run: `python pruning/run.py <baseline|channel|nas|vocab|combined|layers|reeval> [--level 0.3|0.5] [--only ffn|heads] [--keep-decoder 6|4] [--epochs 2 --recover-rows 300000 --distill --batch-size 32 --grad-accum 2 --tag long|distill]`, then `python pruning/report.py` to rebuild the table. Needs a CUDA GPU for a sane runtime; `combined` needs the `channel-<level>` and `nas-<level>` metrics first.
4. Tests: `pytest` (model) and `cd pruning && pytest` (pruning, real CPU torch); both must stay at 100%.

## What the results say

All on 5000 held-out rows (about +/-0.7 pt, one seed). Baseline 40.2% exact.

- Vocabulary pruning is free: -23% weights, 39.6% (-0.6, within noise).
- Channel/NAS: about -3 to -5 pt at 30%, -8 at 50%; NAS vs channel is within noise.
- Ablation: FFN-only is much cheaper in accuracy than heads-only (ffn-50 -2.7 pt for 16% fewer weights; heads-50 -5.0 pt for 12%).
- Decoder layer dropping: keep 6 = -1.0 pt; keep 4 = -3.6 pt and the only cut that cuts CPU decode time clearly (444 to 288 ms). Width pruning barely changes latency.
- Combined (vocab + channel 50%, 74 MB fp16): 31.6% with the 1-epoch recovery; 37.4% (-2.8) with 2 epochs on 300k rows, so most of the loss was short recovery. Adding distillation (36.3%) did not help.
- Still short of the proposal: under 5% relative drop and under 50 MB. Untried: iterative prune/fine-tune, int8 quantization, combining layer dropping with the rest.
- Caveats: sizes are PyTorch parameter counts (Core ML duplicates the embedding); GPU batch-1 latency is launch-bound, read CPU latency; vocab pruning needs an id remap in `T5Translator.swift` that does not exist yet; the pruned models are saved as `model.pt` + `spec.json`, not HF format.
