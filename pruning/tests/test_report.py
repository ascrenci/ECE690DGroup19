import json
import runpy
import sys

import report


def record(rid, total, exact, cuda=True):
    lat = {"cpu": 100.0, **({"cuda": 10.0} if cuda else {})}
    return {"id": rid, "params": {"total": total}, "size_mb": {"fp16": total * 2 / 1e6},
            "eval": {"exact_match": exact, "char_similarity": 0.9, "loss": 0.5}, "latency_ms": lat}


def write(tmp_path, *records):
    for r in records:
        (tmp_path / f"{r['id']}.json").write_text(json.dumps(r))


def test_load_orders_baseline_first_then_known_then_other(tmp_path):
    write(tmp_path, record("zzz", 1, 0.1), record("vocab", 1, 0.1), record("baseline", 2, 0.2), record("channel-30", 1, 0.1))
    assert [r["id"] for r in report.load(tmp_path)] == ["baseline", "channel-30", "vocab", "zzz"]


def test_rows_show_deltas_against_baseline_and_na_without_gpu():
    base, small = record("baseline", 77_000_000, 0.40), record("vocab", 38_500_000, 0.38, cuda=False)
    first, second = report.rows([base, small])
    assert first == ["baseline", "77.0", "+0%", "154", "40.0%", "+0.0 pt", "0.900", "0.500", "100", "10"]
    assert second[2] == "-50%" and second[5] == "-2.0 pt" and second[9] == "n/a"


def test_markdown_and_csv_have_a_header_and_one_line_per_model():
    table = report.rows([record("baseline", 10, 0.5)])
    md, csv_text = report.markdown(table), report.to_csv(table)
    assert md.splitlines()[0].startswith("| Model |") and md.splitlines()[1].startswith("|---") and len(md.splitlines()) == 3
    assert csv_text.splitlines()[0].startswith("Model,") and len(csv_text.splitlines()) == 2


def test_main_writes_md_and_csv(tmp_path, capsys):
    write(tmp_path, record("baseline", 10, 0.5), record("nas-50", 6, 0.4))
    out = tmp_path / "out"
    out.mkdir()
    report.main(["--metrics", str(tmp_path), "--out", str(out)])
    assert "nas-50" in (out / "comparison.md").read_text() and "nas-50" in (out / "comparison.csv").read_text()
    assert "baseline" in capsys.readouterr().out


def test_runs_as_script(tmp_path, monkeypatch):
    write(tmp_path, record("baseline", 10, 0.5))
    monkeypatch.setattr(sys, "argv", ["report.py", "--metrics", str(tmp_path), "--out", str(tmp_path)])
    runpy.run_module("report", run_name="__main__")
    assert (tmp_path / "comparison.md").exists()
