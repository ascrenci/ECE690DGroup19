# Handoff: T5 v2 (notes with inline math)

Read `AGENTS.md` first. Delete this file once the steps below are done.

## Goal

Record a lecture and get math notes: prose with inline `$...$` LaTeX. v1 (`t5-math-v1` release) maps a transcript to a single bare equation. v2 is trained to output the whole sentence, with the equation wrapped in `$...$`.

## Done (committed)

- `model/prepare_data.py`: rows are built from MathBridge's `context_before` + `spoken_English` + `context_after`.
  - `spoken`: the three parts joined, normalized
  - `latex`: `before $equation$ after`
  - `target`: `latex_chars.encode(latex)`
  - New filters: no context, `$` in context, more than 127 tokens on either side.
  - Tests are updated; the full suite is at 100%.
- `model/train_t5.py`: `--max-source` and `--max-target` defaults are both 127, to fit `convert_t5.py`'s 128 encoder and decoder lengths.
- App: SwiftMath was replaced by LaTeXSwiftUI 2.0.0, set up in `project.yml`.
  - `ContentView` renders `LaTeX(latex).parsingMode(.onlyEquations).errorMode(.original)` with the raw LaTeX underneath.
  - `T5Translator.translate` now generates up to 127 new tokens.
- `.github/workflows/build.yml` caches the Xcode `build/` folder. It hasn't yet been checked that the cache actually speeds up builds.
- None of the Swift changes have compiled yet. The next Build iOS App run is the first check.

## Training machine (RTX 4070 Super, 12 GB)

Train with v1's recipe: the `train_t5.py` defaults (batch 64, lr 3e-4) for 2 epochs on 300k rows. Do not change hyperparameters. If batch 64 runs out of memory, stop and ask the user.

For reference, on a 4 GB RTX 3050 Ti, batch 64 and 32 ran out of memory, and batch 16 peaked at about 3.7 GB. A 6.4k-row trial gave exact match 0.196 and character similarity 0.872.

## Next steps

1. Set up the environment and build the data. Data prep streams MathBridge and takes a few minutes; `model/data/` is gitignored, so it has to be built on this machine.
   ```bash
   uv venv model/.venv --python 3.11
   uv pip install --python model/.venv/bin/python -r model/requirements-train.txt
   model/.venv/bin/python -c "import torch; print(torch.cuda.is_available())"   # must print True
   model/.venv/bin/python model/prepare_data.py
   ```
   It should print `train 300000  eval 1000`. With the same seed the drop counts should come out close to these: duplicate 14858, no context 2262, not representable 2193, too long 17, empty spoken 10.
2. Train:
   ```bash
   model/.venv/bin/python model/train_t5.py --epochs 2 --out model/checkpoints/t5-math-v2
   ```
   - The script prints the eval score at the end; v1 scored exact 0.378 and similarity 0.820.
   - Look at the printed OK/BAD samples to check the prose is kept and the `$` signs land in the right places.
   - If the run is interrupted, re-run the same command with `--resume` added.
3. Zip the checkpoint with its files at the root of the zip (the CI step unzips and converts that folder directly):
   ```bash
   cd model/checkpoints/t5-math-v2/final && zip -r ../t5-math-v2.zip . && cd -
   ```
4. Create the release. Confirm with the user before running this.
   ```bash
   gh release create t5-math-v2 model/checkpoints/t5-math-v2/t5-math-v2.zip \
     --notes "flan-t5-small fine-tuned 2 epochs on 300k MathBridge sentences (context + spoken math -> prose with inline \$LaTeX\$). Held-out 1000 rows: exact match X, char similarity Y."
   ```
5. Set `T5_RELEASE: t5-math-v2` in `.github/workflows/build.yml`, then commit and push.
6. Run the Build iOS App workflow: `gh workflow run build.yml`, then `gh run watch`. If it fails, read the log with `gh run view <id> --log-failed`.
7. Report the eval numbers and build result to the user. The user then installs the IPA and tests it by speaking a sentence that contains math.

## Known limits / next changes

- The transcript is still cut off at 128 encoder tokens, so a long lecture is truncated. The next feature is to split the transcript into sentences and translate each one.
- MathBridge's context is written text (`km/s`, `101`), while Whisper writes what was said (`kilometers per second`). The model may echo the written form.
- LaTeXSwiftUI renders through MathJax, so the first render may be slow.
- Swift has no test job yet; see AGENTS.md section 3.
