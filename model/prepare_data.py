#!/usr/bin/env python3
"""Build a clean train/eval subset of MathBridge (Kyudan/MathBridge, MIT) for fine-tuning T5.

Writes <out>/train.jsonl and <out>/eval.jsonl, one object per line:
  spoken : context_before + spoken equation + context_after, normalized (model input, after the task prompt)
  latex  : the same sentence as a note, "before $equation$ after" (what we score against)
  target : latex_chars.encode(latex) (what the model is trained to emit)

Filters (all counted and printed):
  * empty spoken text, or single-character equations (the paper drops ~7M of these too),
  * rows with no context, or with "$" in the context,
  * notes that can't be represented with T5's vocabulary (see latex_chars.roundtrip_ok),
  * exact duplicate (spoken, latex) pairs,
  * rows longer than MAX_TOKENS on either side (training would see them truncated).
The eval rows are disjoint from the train rows (same shuffled stream, taken first).
"""
import argparse
import collections
import itertools
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

import latex_chars

MAX_TOKENS = 127  # convert_t5.py's default 128 encoder/decoder lengths, minus </s>
BATCH = 1_000  # rows read and tokenized per call; one call per row spent most of the time in Python overhead


def screen(row):
    """The note built from one MathBridge row, with the reason to drop it if a cheap check already fails."""
    said = latex_chars.normalize_spoken(row["spoken_English"] or "")
    equation = latex_chars.normalize(row["equation"] or "")
    before = " ".join((row["context_before"] or "").split())
    after = " ".join((row["context_after"] or "").split())
    drop = None
    if not said:
        drop = "empty spoken"
    elif len(equation) < 2:
        drop = "single-char equation"
    elif not (before or after):
        # a lone "$...$" would be stripped back to bare LaTeX by latex_chars.normalize
        drop = "no context"
    elif "$" in before + after:
        drop = "math in context"
    return {
        "spoken": latex_chars.normalize_spoken(f"{before} {said} {after}"),
        "latex": " ".join(p for p in (before, f"${equation}$", after) if p),
        "drop": drop,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", default="google/flan-t5-small")
    ap.add_argument("--out", type=Path, default=Path("model/data"))
    ap.add_argument("--train-rows", type=int, default=300_000)
    ap.add_argument("--eval-rows", type=int, default=1_000)
    ap.add_argument("--extra-eval-rows", type=int, default=0,
                    help="more held-out rows (eval_extra.jsonl) taken AFTER the train rows, so they stay unseen by any model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--buffer", type=int, default=100_000, help="shuffle buffer for the streamed dataset")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, clean_up_tokenization_spaces=False)
    latex_chars.check_tokenizer(tok)

    stream = load_dataset("Kyudan/MathBridge", split="train", streaming=True)
    stream = stream.shuffle(seed=args.seed, buffer_size=args.buffer)

    want = args.train_rows + args.eval_rows + args.extra_eval_rows
    max_batches = -(-(want * 3 + 10_000) // BATCH)  # filters drop <10%; the cap just bounds a bad run
    seen, kept, dropped = set(), [], collections.Counter()
    for batch in itertools.islice(stream.iter(batch_size=BATCH), max_batches):
        rows = [screen(dict(zip(batch, values))) for values in zip(*batch.values())]
        fine = [r for r in rows if not r["drop"]]
        for r, ok in zip(fine, latex_chars.roundtrip_ok_many([r["latex"] for r in fine], tok)):
            r["drop"] = None if ok else "not representable in T5 vocab"
        fine = [r for r in fine if not r["drop"]]
        for r in fine:
            r["target"] = latex_chars.encode(r["latex"])
        sources = latex_chars.token_ids(tok, [latex_chars.PROMPT + r["spoken"] for r in fine])
        targets = latex_chars.token_ids(tok, [r["target"] for r in fine])
        for r, s, t in zip(fine, sources, targets):
            r["drop"] = "too long" if max(len(s), len(t)) > MAX_TOKENS else None

        # in stream order, so the kept rows, the duplicate check and the counts match a row-at-a-time pass
        for r in rows:
            if r["drop"]:
                dropped[r["drop"]] += 1
            elif (r["spoken"], r["latex"]) in seen:
                dropped["duplicate"] += 1
            else:
                seen.add((r["spoken"], r["latex"]))
                kept.append({k: r[k] for k in ("spoken", "latex", "target")})
                if len(kept) == want:
                    break
        if len(kept) == want:
            break

    if len(kept) < want:
        print(f"warning: only {len(kept)} usable rows, wanted {want}")
    eval_rows = kept[: args.eval_rows]
    train_rows = kept[args.eval_rows : args.eval_rows + args.train_rows]
    splits = [("eval", eval_rows), ("train", train_rows)]
    if args.extra_eval_rows:
        splits.append(("eval_extra", kept[args.eval_rows + args.train_rows :]))

    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits:
        with open(args.out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train {len(train_rows)}  eval {len(eval_rows)}  -> {args.out}")
    print("dropped:", dict(dropped))


if __name__ == "__main__":
    main()
