import json
import runpy
import sys

import prepare_data


class Stream:
    def __init__(self, rows):
        self.rows = rows

    def shuffle(self, seed, buffer_size):
        return iter(self.rows)


def row(spoken, equation, before="so", after=""):
    return {"spoken_English": spoken, "equation": equation, "context_before": before, "context_after": after}


ROWS = [
    row("x squared.", "$ x ^ { 2 } $", before="we  get", after="here"),  # kept (becomes eval)
    row("x squared", "x ^ { 2 }", before="we get", after="here"),  # duplicate
    row("", "z = 1"),  # empty spoken
    row(None, "z = 2"),  # missing spoken
    row("x", "x"),  # single-character equation
    row("w plus one", "w + 1", before=None, after=None),  # no context
    row("w plus two", "w + 2", after="and $ v $"),  # math in context
    row("at", "a @ b"),  # reserved character
    row("long", "y = 1", after="word " * 200),  # too long
    row("y plus one", "y + 1", before="", after="holds"),  # kept, context after only
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
    assert eval_rows == [{"spoken": "we get x squared here", "latex": "we get $x ^ { 2 }$ here", "target": 'we get $x ! " 2 ?$ here'}]
    assert [r["latex"] for r in train_rows] == ["$y + 1$ holds", "so $z - 1$"]
    assert train_rows[0]["spoken"] == "y plus one holds"
    out = capsys.readouterr().out
    assert "train 2  eval 1" in out
    for reason in ("duplicate", "empty spoken", "single-char equation", "no context", "math in context",
                   "not representable", "too long"):
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
