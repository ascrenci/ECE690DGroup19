# Working agreements

For everyone who changes this repo: teammates and AI agents alike. `CLAUDE.md` imports this file, so there is one copy to maintain.

## Project in one paragraph

iOS app that turns spoken math from lectures into LaTeX, live and fully on-device. Whisper (WhisperKit) transcribes; a fine-tuned T5 (Core ML) converts the transcript to LaTeX.

- `HelloWorld/` – Swift app (`Transcriber`, `T5Translator`, `ContentView`)
- `model/` – Python: data prep, training, Core ML conversion, and `latex_chars.py` (the LaTeX ↔ T5-safe character mapping; T5 cannot emit `\ ^ { } ~ < ` `)
- `pruning/` – pruning study on the trained T5 (channel, NAS, vocabulary); `python pruning/run.py <method>`, `python pruning/report.py`
- `data/` – tracked results: `data/metrics/*.json` per model, `data/comparison.{md,csv}` (not the training data, which is `model/data/`)
- `scripts/` – bash only (`build-and-download.sh`, `download-latest.sh`); no Windows duplicates
- `.github/workflows/build.yml` – converts the model on macOS and builds the unsigned IPA
- Generated and never committed: `HelloWorld/Models/`, `model/data/`, `model/checkpoints/`, `model/coreml/`, `dist/`

## 0. Pull first

Before anything else, check whether teammates have pushed changes:

```bash
git fetch && git status -sb
```

If the branch is `behind`, commit or stash local work, then `git pull`. Pulled changes can make the graph stale, so run `graphify update .` afterwards.

## 1. Understand before changing: use the graph

Before planning or editing, and again whenever a request touches code you haven't read:

`graphify-out/` is gitignored: every clone builds its own graph (first time: `/graphify .` from the repo root).

1. Read `graphify-out/GRAPH_REPORT.md` and run `graphify query "<question>"` (or `graphify path` / `graphify explain`) for what the change touches.
2. Find what already exists that solves the problem, and reuse it. New code is a last resort. Prefer the smallest diff that fully solves the task.
3. Keep the graph current: after changes, run `graphify update .`. If `graphify-out/` is missing or stale, rebuild with `/graphify .` from the repo root.

## 2. Ask before deciding

Do not make a design decision on your own. If a choice is open (new dependency, new file or module, data format, model or hyperparameter change, public API, workflow or CI change, anything with more than one reasonable answer), stop and ask, with your recommendation, and wait for the answer. Agents: use the question tool. Do not infer approval from silence or from an earlier, different approval.

## 3. Tests: every line, 100% coverage

- All code ships with unit tests. No exceptions for "small" or "script" code.
- Coverage must be **100%** (lines and branches). No `# pragma: no cover` / coverage excludes without asking first.
- Python (live): `pip install -r model/requirements-test.txt`, then run `pytest` from the repo root. Tests are in `model/tests/`; `pytest.ini` enforces 100% line + branch coverage. `conftest.py` fakes torch/transformers/datasets/coremltools, so tests are offline and instant: add fakes there, never download weights or datasets.
- Swift (pending): `HelloWorldTests/` is an empty XCTest stub with no CI job yet (the runner has no iOS simulator matching the app's deployment target, and there is nothing to run). The Swift code (WhisperKit, microphone, SwiftUI, Core ML bundle) needs a testability refactor first; that planned change also adds the Swift CI job.
- Pruning (live): `pruning/` has its own suite with real CPU torch and a tiny random T5 (no downloads). Install CPU torch plus `pruning/requirements-test.txt`, then `cd pruning && pytest`; same 100% gate. It is separate because `model/tests` fakes torch globally.
- CI (`.github/workflows/test.yml`) runs the `model/` and `pruning/` suites on every push and PR.
- Bug fix = a failing test first, then the fix.
- Python is at 100% today. Never lower it. Swift predates these rules; when you touch a Swift file, make it testable and test it.

### What to run

| Change | Run |
|---|---|
| Small edit (one function/file, no interface change) | Only the tests that could be affected. Use the graph to find callers and their tests. |
| Major change (see below) | The **full** suite, plus the coverage gate. |

## 4. Plan major changes first

A change is **major** if it adds a feature, a module, or a dependency; changes a public interface, the model/data format, or CI/build config; or touches more than a few files. For these:

1. Switch to **plan mode**. Read the graph and the code, work out what must change, what is reused, and what tests are needed. Raise open decisions (section 2).
2. Get the plan approved.
3. Then execute, and run the full suite.

## 5. Parallelize

When steps do not depend on each other (reading files, graph queries, running independent test groups, separate subagent tasks), run them at the same time: independent tool calls go in one message, not one after another. Only sequence work that truly depends on an earlier result.

## 6. Conventions

- Keep the shortest working diff. No abstractions, options, or "for later" code that nobody asked for. Delete before adding.
- Don't add comments that restate the code. Comment only the non-obvious *why*.
- Commits: small, one purpose each. Message is one short imperative line (uber concise, no body unless essential) and never uses em dashes. Don't push to `main` without being asked.
- Swift passes `swiftlint` (config in `.swiftlint.yml`); Python stays as clean as the existing `model/` files.
- Do not commit secrets, model weights, datasets, or generated files listed above. Large artifacts go in GitHub releases (the current model is the `t5-math-v1` release).
