#!/usr/bin/env python3
"""Run one pruning experiment: prune the trained model, recover with a short fine-tune, measure, record.

  python pruning/run.py baseline
  python pruning/run.py channel --level 0.3      # also: nas, vocab, combined
  python pruning/report.py                       # comparison table from data/metrics/*.json

Every model gets the same recovery (first --recover-rows training rows, 1 epoch, same lr and seed) and
the same 1000 held-out eval rows. `combined` = vocab pruning + the better of channel/nas at --level,
re-pruned from the base weights (needs data/metrics/{channel,nas}-<level>.json to exist).
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
import metrics  # noqa: E402
import nas  # noqa: E402
import surgery  # noqa: E402
import train_t5  # noqa: E402
import vocab  # noqa: E402

PROXY_ROWS = 512  # tail of train.jsonl: scores NAS candidates, never overlaps the eval rows or the recovery rows


def read_rows(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def run_id(method, level):
    return method if method in ("baseline", "vocab") else f"{method}-{round(level * 100)}"


def prune(method, model, tok, rows, args):
    """Apply `method` to `model` in place; returns (tokenizer to use afterwards, kept_ids or None, extra record)."""
    if method == "channel":
        channel.apply(model, *channel.uniform(model, args.level))
        return tok, None, {}
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
    ap.add_argument("method", choices=["baseline", "channel", "nas", "vocab", "combined"])
    ap.add_argument("--level", type=float, default=0.5, help="fraction of attention+FFN params removed")
    ap.add_argument("--base", type=Path, default=Path("model/checkpoints/t5-math-v2/final"))
    ap.add_argument("--data", type=Path, default=Path("model/data"))
    ap.add_argument("--metrics", type=Path, default=Path("data/metrics"))
    ap.add_argument("--models", type=Path, default=Path("model/checkpoints/pruned"))
    ap.add_argument("--recover-rows", type=int, default=100_000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.base, clean_up_tokenization_spaces=False)
    model = T5ForConditionalGeneration.from_pretrained(args.base, torch_dtype=torch.float32)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    eval_rows, rows = read_rows(args.data / "eval.jsonl"), read_rows(args.data / "train.jsonl")

    kept, extra, recovery = None, {}, None
    if args.method != "baseline":
        tok, kept, extra = prune(args.method, model, tok, rows, args)
        recovery = {"rows": args.recover_rows, "epochs": 1, "lr": args.lr, "batch_size": args.batch_size, "seed": args.seed}
        train_t5.fit(model, tok, Dataset.from_list(rows[: args.recover_rows]), args.models / "tmp", 1,
                     args.batch_size, args.lr, 127, 127, save=False)

    rid = run_id(args.method, args.level)
    record = {"id": rid, "method": args.method, "level": None if args.method in ("baseline", "vocab") else args.level,
              "recovery": recovery, **extra, **metrics.measure(model, tok, eval_rows)}
    metrics.write(args.metrics / f"{rid}.json", record)
    if args.method != "baseline":
        surgery.save(model, args.models / rid, kept)
    print(f"wrote {args.metrics / (rid + '.json')}")


if __name__ == "__main__":
    main()
