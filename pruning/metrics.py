"""Metrics shared by every model in the study (baseline and pruned): size, accuracy, loss, latency."""
import copy
import json
import time

import torch

import latex_chars
import surgery
import train_t5


def param_counts(model):
    """total / embeddings (input embedding + lm_head) / layers (everything else)."""
    total = sum(p.numel() for p in model.parameters())
    emb = model.get_input_embeddings().weight.numel() + model.lm_head.weight.numel()
    return {"total": total, "embeddings": emb, "layers": total - emb, "prunable": surgery.prunable_params(model)}


def batches(tok, rows, max_source=127, max_target=127, batch_size=32):
    """(encoded inputs, labels with padding as -100) per batch, tokenized exactly as in training."""
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        enc = tok([latex_chars.PROMPT + r["spoken"] for r in chunk], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_source)
        labels = tok(text_target=[r["target"] for r in chunk], return_tensors="pt", padding=True,
                     truncation=True, max_length=max_target)["input_ids"]
        labels[labels == tok.pad_token_id] = -100
        yield enc, labels


def eval_loss(model, tok, rows, max_source=127, max_target=127, batch_size=32):
    """Teacher-forced token cross-entropy on `rows`."""
    model.eval()
    device = next(model.parameters()).device
    loss_sum, tokens = 0.0, 0
    for enc, labels in batches(tok, rows, max_source, max_target, batch_size):
        with torch.no_grad():
            loss = model(**enc.to(device), labels=labels.to(device)).loss
        n = int((labels != -100).sum())
        loss_sum, tokens = loss_sum + float(loss) * n, tokens + n
    return loss_sum / tokens


def latency_ms(model, tok, rows, device, max_target=127):
    """Mean greedy-decode time per sentence at batch size 1 on `device` (a PyTorch proxy, not Core ML)."""
    model = copy.deepcopy(model).to(device).eval()
    times = []
    for row in rows:
        enc = tok(latex_chars.PROMPT + row["spoken"], return_tensors="pt").to(device)
        start = time.perf_counter()
        with torch.no_grad():
            model.generate(**enc, max_new_tokens=max_target, do_sample=False)
        (torch.cuda.synchronize() if device == "cuda" else None)
        times.append(time.perf_counter() - start)
    return 1000 * sum(times[1:]) / (len(times) - 1)  # first call is warm-up


def measure(model, tok, eval_rows, max_source=127, max_target=127, latency_rows=30):
    """Everything we record per model, as a JSON-ready dict."""
    scores = train_t5.evaluate(model, tok, eval_rows, max_source, max_target, show=0)
    params = param_counts(model)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    return {
        "params": params,
        "size_mb": {"fp32": params["total"] * 4 / 1e6, "fp16": params["total"] * 2 / 1e6},
        "eval": {**scores, "loss": eval_loss(model, tok, eval_rows, max_source, max_target), "rows": len(eval_rows)},
        "latency_ms": {d: latency_ms(model, tok, eval_rows[:latency_rows], d, max_target) for d in devices},
        "spec": surgery.spec_of(model),
    }


def write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
