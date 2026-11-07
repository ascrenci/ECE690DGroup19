import json
import runpy
import sys

import prepare_data


class Stream:
    def __init__(self, rows):
        self.rows = rows

    def shuffle(self, seed, buffer_size):
        return iter(self.rows)


def row(spoken, equation):
    return {"spoken_English": spoken, "equation": equation}


ROWS = [
    row("x squared", "$ x ^ { 2 } $"),  # kept (becomes eval)
    row("x squared", "x ^ { 2 }"),  # duplicate
    row("", "z = 1"),  # empty spoken
    row(None, "z = 2"),  # missing spoken
    row("x", "x"),  # single-character equation
    row("at", "a @ b"),  # reserved character
    row("y plus one", "y + 1"),  # kept
    row("z minus one", "z - 1"),  # kept -> reaches the requested total
    row("never read", "q = 1"),
]


def args(out, train, eval_):
    return ["prepare_data.py", "--out", str(out), "--train-rows", str(train), "--eval-rows", str(eval_)]


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_main_filters_splits_and_writes_jsonl(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(prepare_data, "load_dataset", lambda *a, **k: Stream(ROWS))
    monkeypatch.setattr(sys, "argv", args(tmp_path, 2, 1))
    prepare_data.main()

    eval_rows, train_rows = read(tmp_path / "eval.jsonl"), read(tmp_path / "train.jsonl")
    assert [r["latex"] for r in eval_rows] == ["x ^ { 2 }"]
    assert eval_rows[0]["target"] == 'x ! " 2 ?'
    assert [r["latex"] for r in train_rows] == ["y + 1", "z - 1"]
    out = capsys.readouterr().out
    assert "train 2  eval 1" in out
    for reason in ("duplicate", "empty spoken", "single-char equation", "not representable"):
        assert reason in out
    assert "warning" not in out


def test_main_warns_when_not_enough_usable_rows(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(prepare_data, "load_dataset", lambda *a, **k: Stream(ROWS[:1]))
    monkeypatch.setattr(sys, "argv", args(tmp_path, 5, 1))
    prepare_data.main()
    assert "warning: only 1 usable rows" in capsys.readouterr().out


def test_runs_as_script(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules["datasets"], "load_dataset", lambda *a, **k: Stream(ROWS))
    monkeypatch.setattr(sys, "argv", args(tmp_path, 2, 1))
    runpy.run_module("prepare_data", run_name="__main__")
    assert (tmp_path / "train.jsonl").exists()
