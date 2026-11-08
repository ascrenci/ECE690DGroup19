#!/usr/bin/env python3
"""Comparison table: baseline vs every pruned model, from data/metrics/*.json -> data/comparison.{md,csv}."""
import argparse
import csv
import io
import json
from pathlib import Path

ORDER = ["baseline", "channel-30", "channel-50", "channel-ffn-30", "channel-ffn-50", "channel-heads-30", "channel-heads-50",
         "nas-30", "nas-50", "layers-dec6", "layers-dec4", "vocab", "combined-50", "combined-50-long", "combined-50-distill"]
COLUMNS = ["Model", "Params (M)", "Param change", "fp16 MB", "Exact match", "Exact change", "Char sim", "Loss",
           "CPU ms", "GPU ms"]


def load(metrics_dir):
    """Records sorted baseline first, then ORDER, then anything else by name."""
    records = [json.loads(p.read_text()) for p in Path(metrics_dir).glob("*.json")]
    rank = {name: i for i, name in enumerate(ORDER)}
    return sorted(records, key=lambda r: (rank.get(r["id"], len(ORDER)), r["id"]))


def rows(records):
    base = next(r for r in records if r["id"] == "baseline")
    out = []
    for r in records:
        params, exact = r["params"]["total"], r["eval"]["exact_match"]
        lat = r["latency_ms"]
        out.append([
            r["id"],
            f"{params / 1e6:.1f}",
            f"{100 * (params / base['params']['total'] - 1):+.0f}%",
            f"{r['size_mb']['fp16']:.0f}",
            f"{100 * exact:.1f}%",
            f"{100 * (exact - base['eval']['exact_match']):+.1f} pt",
            f"{r['eval']['char_similarity']:.3f}",
            f"{r['eval']['loss']:.3f}",
            f"{lat['cpu']:.0f}",
            f"{lat['cuda']:.0f}" if "cuda" in lat else "n/a",
        ])
    return out


def markdown(table):
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "|".join("---" for _ in COLUMNS) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in table]
    return "\n".join(lines) + "\n"


def to_csv(table):
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows([COLUMNS, *table])
    return buf.getvalue()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", type=Path, default=Path("data/metrics"))
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args(argv)
    table = rows(load(args.metrics))
    (args.out / "comparison.md").write_text(markdown(table), encoding="utf-8")
    (args.out / "comparison.csv").write_text(to_csv(table), encoding="utf-8")
    print(markdown(table))


if __name__ == "__main__":
    main()
