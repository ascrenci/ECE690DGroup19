import json
import runpy
import sys

import prepare_data


class Stream:
    def __init__(self, rows):
        self.rows = rows

    def shuffle(self, seed, buffer_size):
        return self

    def iter(self, batch_size):
        for i in range(0, len(self.rows), batch_size):
            chunk = self.rows[i : i + batch_size]
            yield {k: [r[k] for r in chunk] for k in chunk[0]}


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


def test_extra_eval_rows_come_after_train_and_leave_it_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_data, "load_dataset", lambda *a, **k: Stream(ROWS))
    monkeypatch.setattr(sys, "argv", args(tmp_path, 2, 1) + ["--extra-eval-rows", "1"])
    prepare_data.main()
    assert [r["latex"] for r in read(tmp_path / "train.jsonl")] == ["$y + 1$ holds", "so $z - 1$"]
    assert [r["latex"] for r in read(tmp_path / "eval_extra.jsonl")] == ["so $q = 1$"]


def test_no_extra_file_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_data, "load_dataset", lambda *a, **k: Stream(ROWS))
    monkeypatch.setattr(sys, "argv", args(tmp_path, 2, 1))
    prepare_data.main()
    assert not (tmp_path / "eval_extra.jsonl").exists()


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


def run(tmp_path, monkeypatch, batch, rows=ROWS):
    monkeypatch.setattr(prepare_data, "BATCH", batch)
    monkeypatch.setattr(prepare_data, "load_dataset", lambda *a, **k: Stream(rows))
    monkeypatch.setattr(sys, "argv", args(tmp_path, 2, 1) + ["--extra-eval-rows", "1"])
    prepare_data.main()
    return [(tmp_path / f).read_text() for f in ("eval.jsonl", "train.jsonl", "eval_extra.jsonl")]


def test_any_batch_size_gives_the_same_rows_and_counts(tmp_path, monkeypatch, capsys):
    # size 2 also gives batches where every row fails a cheap check, so nothing is tokenized
    whole = run(tmp_path / "a", monkeypatch, len(ROWS)), capsys.readouterr().out
    for batch in (1, 2, 3):
        assert (run(tmp_path / str(batch), monkeypatch, batch), capsys.readouterr().out.replace(f"/{batch}", "/a")) == whole


def test_duplicates_are_caught_across_batches(tmp_path, monkeypatch, capsys):
    run(tmp_path, monkeypatch, 1, rows=ROWS[:1] + ROWS)
    assert "'duplicate': 2" in capsys.readouterr().out


def test_cap_stops_reading_a_stream_of_unusable_rows(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(prepare_data, "BATCH", 10_000)
    monkeypatch.setattr(prepare_data, "load_dataset", lambda *a, **k: Stream([row("", "z = 1")] * 30_000))
    monkeypatch.setattr(sys, "argv", args(tmp_path, 1, 1))
    prepare_data.main()
    assert "'empty spoken': 20000" in capsys.readouterr().out  # cap 2*3 + 10_000 rounds up to 2 batches
