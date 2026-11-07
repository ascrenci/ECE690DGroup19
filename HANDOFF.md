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

## State on the training laptop (RTX 3050 Ti, 4 GB, uv venv at `model/.venv`)

- `model/data/{train,eval}.jsonl` are already built for v2: 300k train rows and 1k eval rows. They are gitignored.
  - Drop counts: duplicate 14858, no context 2262, not representable 2193, too long 17, empty spoken 10.
  - To rebuild: `model/.venv/bin/python model/prepare_data.py`
- Batch size 64 and 32 run out of GPU memory. Batch 16 fits, peaking at about 3.7 GB, and runs at about 99 rows per second.
- Trial result: 6.4k rows, 1 epoch at batch 16 gave exact match 0.196 and character similarity 0.872 on the eval set.
- **Open decision, ask the user (AGENTS.md section 2):** v1 used batch 64 for 2 epochs. Two options:
  - **(a)** Plain batch 16. No code change.
  - **(b)** Add a `--grad-accum` flag, set `gradient_accumulation_steps` in `Seq2SeqTrainingArguments`, and add a test for it. This keeps the effective batch at 64, matching v1.

## Next steps

1. Train for about 1h40m, with the batch settings the user picks:
   ```bash
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True model/.venv/bin/python model/train_t5.py \
     --epochs 2 --batch-size 16 --out model/checkpoints/t5-math-v2
   ```
   - The script prints the eval score at the end; v1 scored exact 0.378 and similarity 0.820.
   - Look at the printed OK/BAD samples to check the prose is kept and the `$` signs land in the right places.
2. Zip the checkpoint with its files at the root of the zip (the CI step unzips and converts that folder directly):
   ```bash
   cd model/checkpoints/t5-math-v2/final && zip -r ../t5-math-v2.zip .
   ```
3. Create the release. Confirm with the user before running this.
   ```bash
   gh release create t5-math-v2 model/checkpoints/t5-math-v2/t5-math-v2.zip --notes "<recipe + eval numbers>"
   ```
4. Set `T5_RELEASE: t5-math-v2` in `.github/workflows/build.yml`, then commit and push.
5. Run the Build iOS App workflow: `gh workflow run build.yml`, then `gh run watch`. If it fails, read the log with `gh run view <id> --log-failed`.
6. Install the IPA, then speak a sentence that contains math. Expect prose with the math rendered inline.

## Known limits / next changes

- The transcript is still cut off at 128 encoder tokens, so a long lecture is truncated. The next feature is to split the transcript into sentences and translate each one.
- MathBridge's context is written text (`km/s`, `101`), while Whisper writes what was said (`kilometers per second`). The model may echo the written form.
- LaTeXSwiftUI renders through MathJax, so the first render may be slow.
- Swift has no test job yet; see AGENTS.md section 3.
