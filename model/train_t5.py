#!/usr/bin/env python3
"""Fine-tune (or just evaluate) T5 on spoken math -> LaTeX, using data from prepare_data.py.

  python model/prepare_data.py                       # once
  python model/train_t5.py                           # fine-tune flan-t5-small
  python model/train_t5.py --eval-only --model model/checkpoints/t5-math   # score a checkpoint
  python model/convert_t5.py --model model/checkpoints/t5-math             # then convert to Core ML

Source text is `latex_chars.PROMPT + spoken`; target is the T5-safe encoding of the LaTeX
(see latex_chars.py). Scoring decodes the model output back to real LaTeX first.

Notes:
  * fp16 is never used (T5 overflows); bf16 is used when the GPU supports it, otherwise fp32.
  * Sequence lengths here must stay <= convert_t5.py's --enc-len/--dec-len (default 128).
"""
import argparse
import difflib
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
)

import latex_chars


def evaluate(model, tok, rows, max_source, max_target, batch_size=64, show=10):
    """Exact match + character similarity on real (decoded) LaTeX, greedy decoding."""
    model.eval()
    device = next(model.parameters()).device
    preds = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        enc = tok(
            [latex_chars.PROMPT + r["spoken"] for r in chunk],
            return_tensors="pt", padding=True, truncation=True, max_length=max_source,
        ).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_target, num_beams=1, do_sample=False)
        preds += [latex_chars.decode(t) for t in tok.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)]

    exact = sum(p == r["latex"] for p, r in zip(preds, rows)) / len(rows)
    sim = sum(difflib.SequenceMatcher(None, p, r["latex"]).ratio() for p, r in zip(preds, rows)) / len(rows)
    print(f"eval on {len(rows)} rows: exact match {exact:.3f} | char similarity {sim:.3f}")
    for p, r in list(zip(preds, rows))[:show]:
        mark = "OK " if p == r["latex"] else "BAD"
        print(f"  {mark} {r['spoken']!r}\n      want {r['latex']!r}\n      got  {p!r}")
    return {"exact_match": exact, "char_similarity": sim}


def fit(model, tok, train, out, epochs, batch_size, lr, max_source, max_target, resume=False, save=True):
    """Fine-tune `model` in place on `train` (rows with `spoken` and `target`); returns the trainer.

    Takes a model object so pruning/ can reuse it. save=False skips intermediate checkpoints.
    """
    def tokenize(batch):
        enc = tok([latex_chars.PROMPT + s for s in batch["spoken"]], max_length=max_source, truncation=True)
        enc["labels"] = tok(text_target=batch["target"], max_length=max_target, truncation=True)["input_ids"]
        return enc

    train = train.map(tokenize, batched=True, remove_columns=train.column_names)

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    targs = Seq2SeqTrainingArguments(
        output_dir=str(out),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
        weight_decay=0.01,
        bf16=bf16,
        fp16=False,
        logging_steps=100,
        save_strategy="steps" if save else "no",
        save_steps=2000,
        save_total_limit=2,
        report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train,
        data_collator=DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100),
        tokenizer=tok,
    )
    trainer.train(resume_from_checkpoint=True if resume else None)
    return trainer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/flan-t5-small", help="base model, or a checkpoint dir with --eval-only")
    ap.add_argument("--data", type=Path, default=Path("model/data"))
    ap.add_argument("--out", type=Path, default=Path("model/checkpoints/t5-math"))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-source", type=int, default=127)
    ap.add_argument("--max-target", type=int, default=127)
    ap.add_argument("--max-train-rows", type=int, default=0, help="0 = all rows in train.jsonl")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--resume", action="store_true", help="resume from the latest checkpoint in --out")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, clean_up_tokenization_spaces=False)
    latex_chars.check_tokenizer(tok)
    model = T5ForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float32)
    if torch.cuda.is_available():
        model.to("cuda")

    eval_rows = [json.loads(line) for line in open(args.data / "eval.jsonl", encoding="utf-8")]
    if args.eval_only:
        evaluate(model, tok, eval_rows, args.max_source, args.max_target)
        return

    files = {"train": str(args.data / "train.jsonl")}
    train = load_dataset("json", data_files=files, split="train")
    if args.max_train_rows:
        train = train.select(range(min(args.max_train_rows, len(train))))

    trainer = fit(model, tok, train, args.out, args.epochs, args.batch_size, args.lr,
                  args.max_source, args.max_target, resume=args.resume)

    final = args.out / "final"
    trainer.save_model(str(final))
    tok.save_pretrained(final)
    print(f"saved {final}")
    evaluate(model, tok, eval_rows, args.max_source, args.max_target)


if __name__ == "__main__":
    main()
