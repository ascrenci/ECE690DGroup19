#!/usr/bin/env python3
"""Run one pruning experiment: prune the trained model, recover with a fine-tune, measure, record.

  python pruning/run.py baseline
  python pruning/run.py channel --level 0.3                  # also: nas, vocab, combined
  python pruning/run.py channel --level 0.5 --only ffn       # ablation: FFN channels only (or --only heads)
  python pruning/run.py layers --keep-decoder 4              # drop decoder blocks by influence
  python pruning/run.py combined --epochs 2 --recover-rows 300000 --distill --batch-size 32 --grad-accum 2 --tag distill
  python pruning/run.py reeval --id channel-30               # re-measure a saved model on the current eval rows
  python pruning/report.py                                   # comparison table from data/metrics/*.json

Default recovery is the same for every model: first --recover-rows training rows, 1 epoch, same lr and seed.
Eval rows are eval.jsonl plus eval_extra.jsonl when present. `combined` = vocab pruning + the better of
channel/nas at --level, re-pruned from the base weights (needs data/metrics/{channel,nas}-<level>.json).
--distill uses the unpruned base model as a frozen teacher during recovery.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))  # train_t5, latex_chars

import channel  # noqa: E402
import distill  # noqa: E402
import layers  # noqa: E402
import metrics  # noqa: E402
import nas  # noqa: E402
import surgery  # noqa: E402
import train_t5  # noqa: E402
import vocab  # noqa: E402

PROXY_ROWS = 512  # tail of train.jsonl: scores NAS candidates and layer influence, never overlaps eval or recovery rows
DISTILL = {"alpha": 0.5, "temp": 2.0}


def read_rows(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def load_eval(data):
    """eval.jsonl plus eval_extra.jsonl when present (extra rows come after the train rows, never trained on)."""
    rows = read_rows(data / "eval.jsonl")
    extra = data / "eval_extra.jsonl"
    return rows + read_rows(extra) if extra.exists() else rows


def reeval(args, model, tok, eval_rows):
    """Re-measure a saved pruned model on the current eval rows; keeps its recorded method/level/recovery."""
    old = json.loads((args.metrics / f"{args.id}.json").read_text())
    old_vocab = model.config.vocab_size
    kept = surgery.load(model, args.models / args.id)
    tok = tok if kept is None else vocab.Remapped(tok, kept, old_vocab)
    record = {**old, **metrics.measure(model, tok, eval_rows)}
    metrics.write(args.metrics / f"{args.id}.json", record)
    return record


def run_id(args):
    method, pct = args.method, round(args.level * 100)
    if method in ("baseline", "vocab"):
        rid = method
    elif method == "layers":
        rid = f"layers-dec{args.keep_decoder}"
    else:
        rid = f"{method}-{args.only}-{pct}" if args.only else f"{method}-{pct}"
    return f"{rid}-{args.tag}" if args.tag else rid


def prune(method, model, tok, rows, args):
    """Apply `method` to `model` in place; returns (tokenizer to use afterwards, kept_ids or None, extra record)."""
    if method == "channel":
        channel.apply(model, *channel.uniform(model, args.level, args.only))
        return tok, None, {"only": args.only}
    if method == "layers":
        scores = layers.influence(model, tok, rows[-PROXY_ROWS:])
        keep = layers.choose(scores, args.keep_decoder)
        surgery.drop_layers(model, {"decoder": keep})
        return tok, None, {"layers": {"decoder_kept": keep, "influence": scores}}
    if method == "nas":
        budget = nas.cost(model.config, *channel.uniform(model, args.level))
        score_fn = nas.make_score(model, tok, rows[-PROXY_ROWS:])
        best, score, evals = nas.search(model.config, budget, score_fn, random.Random(args.seed))
        channel.apply(model, best["ffn"], best["heads"])
        return tok, None, {"nas": {"proxy_loss": score, "evaluations": evals}}
    extra = {}
    if method == "combined":
        found = [json.loads((args.metrics / f"{m}-{round(args.level * 100)}.json").read_text()) for m in ("channel", "nas")]
        best = max(found, key=lambda r: r["eval"]["exact_match"])
        channel.apply(model, best["spec"]["ffn"], best["spec"]["heads"])
        extra = {"channel_from": best["id"]}
    kept = vocab.kept_ids(tok, rows)
    return vocab.apply(model, tok, kept), kept, {**extra, "kept_tokens": len(kept)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("method", choices=["baseline", "channel", "nas", "vocab", "combined", "layers", "reeval"])
    ap.add_argument("--id", help="reeval: id of a saved model, e.g. channel-30")
    ap.add_argument("--level", type=float, default=0.5, help="fraction of attention+FFN params removed")
    ap.add_argument("--only", choices=["ffn", "heads"], help="channel: prune only FFN channels or only heads")
    ap.add_argument("--keep-decoder", type=int, default=4, help="layers: decoder blocks to keep (of 8)")
    ap.add_argument("--base", type=Path, default=Path("model/checkpoints/t5-math-v2/final"))
    ap.add_argument("--data", type=Path, default=Path("model/data"))
    ap.add_argument("--metrics", type=Path, default=Path("data/metrics"))
    ap.add_argument("--models", type=Path, default=Path("model/checkpoints/pruned"))
    ap.add_argument("--recover-rows", type=int, default=100_000)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--grad-accum", type=int, default=1, help="e.g. --batch-size 32 --grad-accum 2 keeps effective 64")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--distill", action="store_true", help="recover with the unpruned base model as teacher")
    ap.add_argument("--tag", help="suffix for the model id, e.g. long or distill")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base, clean_up_tokenization_spaces=False)
    model = T5ForConditionalGeneration.from_pretrained(args.base, torch_dtype=torch.float32).to(device)
    eval_rows, rows = load_eval(args.data), read_rows(args.data / "train.jsonl")
    if args.method == "reeval":
        return reeval(args, model, tok, eval_rows)

    kept, extra, recovery = None, {}, None
    if args.method != "baseline":
        tok, kept, extra = prune(args.method, model, tok, rows, args)
        teacher = {}
        if args.distill:
            base = T5ForConditionalGeneration.from_pretrained(args.base, torch_dtype=torch.float32).to(device)
            teacher = {"trainer_cls": distill.DistillTrainer, "teacher": base, "kept": kept, **DISTILL}
        recovery = {"rows": args.recover_rows, "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
                    "grad_accum": args.grad_accum, "seed": args.seed, "distill": DISTILL if args.distill else None}
        train_t5.fit(model, tok, Dataset.from_list(rows[: args.recover_rows]), args.models / "tmp", args.epochs,
                     args.batch_size, args.lr, 127, 127, save=False, grad_accum=args.grad_accum, **teacher)

    rid = run_id(args)
    record = {"id": rid, "method": args.method,
              "level": None if args.method in ("baseline", "vocab", "layers") else args.level,
              "recovery": recovery, **extra, **metrics.measure(model, tok, eval_rows)}
    metrics.write(args.metrics / f"{rid}.json", record)
    if args.method != "baseline":
        surgery.save(model, args.models / rid, kept)
    print(f"wrote {args.metrics / (rid + '.json')}")
    return record


if __name__ == "__main__":
    main()
