# LayoutLens

LayoutLens turns a UI screenshot into compact structured metadata: element boxes, fills, borders, corner radii, spacing and alignment from
OpenCV, and text with positions from Tesseract OCR. It is a research proof of concept. The question it is built to answer: **does giving a
model this metadata help it reproduce an interface, for models that can see images and for models that cannot?**

```text
$ layoutlens screen.png --mode combined --compact
{
"image": {"width":390,"height":844},
"background": "#f6f7f9",
"elements": [
{"id":"e0","type":"input","x":24,"y":269,"w":342,"h":48,"fill":"#ffffff","border":{"sides_px":[1,1,1,1],"color":"#d9dee6"},"radius":9.1},
{"id":"e2","type":"button","x":24,"y":440,"w":342,"h":50,"fill":"#2563eb","radius":13.3},
...
"texts": [
{"id":"t0","text":"Welcome back","x":24,"y":178,"w":196,"h":21,"conf":96.3,"font_px":29.2,"color":"#111827"},
{"id":"t5","text":"Sign in","x":172,"y":460,"w":45,"h":14,"conf":96.4,"font_px":15.1,"color":"#ffffff","in":"e2","align":"center"},
...
```

## Status

| Part | State |
|---|---|
| Geometry analyzer (OpenCV/Pillow) | done, evaluated |
| Tesseract OCR, full-page and geometry-guided | done, evaluated |
| Combined JSON, three modes | done |
| Dataset: 9 scenes × light/dark × 1×/2× (+3 at 3×), DOM ground truth | done |
| Resource benchmark | done |
| Model pipeline: providers, prompts, rendering, scoring | done; the scorer is validated against reference pages |
| Model experiments | done: 120 runs, scored |

## Results so far

All numbers are measured on the 36 screenshots in `data/screenshots/`. They are clean, lossless renders; real screenshots will score lower.

**Geometry** (1,724 visible ground-truth elements, matched one-to-one at IoU ≥ 0.5)

| | |
|---|---|
| Precision / recall / F1 | 0.92 / 0.94 / 0.93 |
| Mean IoU of matched boxes; share at IoU ≥ 0.9 | 0.977; 94% |
| Position / size error | 0.54 / 0.82 px |
| Fill colour (CIEDE2000) | exact on every matched fill |
| Gaps between neighbours within 1 px | 95% |

Reliable: buttons, inputs, avatars, image blocks, toggles and cards (recall 0.98–1.00). Weaker: small icons that look like letters (0.94),
text lines (0.91) and very large display text (letter bodies become regions). Chart: `results/charts/geometry_recall_by_kind.png`.

**OCR** (876 text elements; CER and WER per element, capped at 1)

| | Full-page OCR | Geometry-guided OCR (combined mode) |
|---|---|---|
| Texts found | 0.887 | **0.956** |
| Exact match | 0.80 | **0.89** |
| Mean CER / WER | 0.143 / 0.178 | **0.058 / 0.088** |
| Box placement (IoU with true text box) | 0.89 | 0.96 |
| Font size error (relative) | 7.8% | 6.8% |

Full-page Tesseract misses single characters and short labels ("+", "2", line numbers) and struggles with dense, mixed-polarity screens.
Reading each text line that the geometry found, as one stacked image, removes most of those errors. Chart: `results/charts/ocr_modes.png`.

**Cost** (Apple M4 Pro, CPU only; mean of 5 warm runs, SD under 3%)

| Image | Geometry | OCR | Combined | Peak memory, combined |
|---|---|---|---|---|
| 390×844 (0.33 MP) | 0.12 s | 0.17 s | 0.41 s | 104 MB |
| 1440×900 (1.3 MP) | 0.40 s | 0.51 s | 1.23 s | 236 MB |
| 2880×1800 (5.2 MP) | 1.34 s | 0.61 s | 2.30 s | 558 MB |
| 4320×2700 (11.7 MP)* | 0.84 s | 0.65 s | 1.83 s | 531 MB |

\* Above 8 MP the geometry analyses every second pixel. Tesseract itself peaks at 50–110 MB. Chart: `results/charts/resources.png`.

The full write-up, including what did not work, is in [docs/REPORT.md](docs/REPORT.md).

## Model results

6 screens × 5 conditions × 2 repetitions per model; identical prompts and settings (reasoning effort medium, 32k output tokens). Mean over all runs (a run with no output is scored as an empty page); SD and completed-only figures are in `results/model_summary.json`.

**Vision model: qwen3.8-flash** (passed the image check)

| Condition | complete/truncated/failed | Box IoU (all boxes) | Text CER | Texts reproduced | Text colour ΔE | SSIM | Latency | Tokens in / out |
|---|---|---|---|---|---|---|---|---|
| baseline | 12/0/0 | 0.58 ± 0.29 | 0.021 | 0.98 | 3.8 | 0.864 | 49 s | 991 / 3970 |
| geometry | 12/0/0 | 0.94 ± 0.05 | 0.051 | 0.95 | 1.1 | 0.942 | 73 s | 4388 / 5994 |
| ocr | 12/0/0 | 0.67 ± 0.25 | 0.036 | 0.95 | 3.5 | 0.880 | 62 s | 2130 / 4709 |
| combined | 12/0/0 | 0.94 ± 0.08 | 0.026 | 0.97 | 1.3 | 0.943 | 63 s | 4823 / 5040 |
| truth | 12/0/0 | 0.98 ± 0.03 | 0.024 | 0.98 | 0.3 | 0.949 | 58 s | 3521 / 4743 |

**Text model: deepseek-v4-flash-0731** (does not read images; never sent the screenshot)

| Condition | complete/truncated/failed | Box IoU (all boxes) | Text CER | Texts reproduced | Text colour ΔE | SSIM | Latency | Tokens in / out |
|---|---|---|---|---|---|---|---|---|
| baseline | 10/2/0 | 0.09 ± 0.11 | 0.711 | 0.07 | 40.0 | 0.479 | 202 s | 199 / 24899 |
| geometry | 11/1/0 | 0.85 ± 0.24 | 0.669 | 0.18 | 1.6 | 0.957 | 219 s | 3082 / 23919 |
| ocr | 9/2/1 | 0.51 ± 0.31 | 0.177 | 0.78 | 34.8 | 0.412 | 250 s | 1138 / 27201 |
| combined | 9/1/2 | 0.80 ± 0.38 | 0.211 | 0.78 | 0.9 | 0.949 | 240 s | 3508 / 25274 |
| truth | 10/1/1 | 0.84 ± 0.38 | 0.162 | 0.83 | 0.1 | 0.950 | 213 s | 2314 / 22572 |

Findings:
- Geometry JSON is what improves layout. Vision model: box IoU 0.58 → 0.94, better in 12/12 paired runs; text-only model: 0.09 → 0.85 (12/12).
- OCR JSON is what gives a text-only model the words (CER 0.71 → 0.18). The vision model already reads text from the screenshot (CER 0.02), so OCR adds nothing there and its boxes do not help layout.
- Combined is close to perfect metadata for both models, at about +3.5k prompt tokens.
- The text-only model hit the 32k output limit in 10 of 60 runs (reasoning used it up), mostly with large metadata; the vision model never did.
- Charts: `results/charts/models_vision.png`, `results/charts/models_text.png`.

## Install

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e '.[bench,dev]'
```

The analyzer itself needs only `numpy`, `opencv-python-headless` and `pillow`, plus the `tesseract` binary for OCR (`brew install tesseract`,
`apt install tesseract-ocr`; set `TESSERACT_CMD` if it is not on `PATH`). The benchmark adds Playwright, which uses the installed Google
Chrome, and matplotlib.

## Use

```bash
layoutlens screen.png --mode geometry      # boxes, colours, borders, radii, spacing; no text content
layoutlens screen.png --mode ocr           # text lines: content, box, confidence, font size
layoutlens screen.png --mode combined      # both, with each text joined to its container
```

```python
from layoutlens import analyze
data = analyze.analyze("screen.png", mode="combined")
prompt_block = analyze.to_text(data)        # one element per line, the form given to models
```

## Reproduce

```bash
make data            # write the scenes and render them with ground truth
make accuracy        # geometry and OCR against the ground truth  -> results/accuracy_*.csv, accuracy_summary.json
make ocr-variants    # the preprocessing / segmentation comparison  -> results/ocr_variants.csv
make resources       # time, CPU and memory per mode and size  -> results/resources.csv
make charts test
```

## Running the model experiments

Any OpenAI-compatible Chat Completions endpoint works. Configure it with environment variables or a `.env` file in the repository root
(git-ignored; variables already set in the shell take precedence).

```bash
export LAYOUTLENS_BASE_URL=https://your-endpoint/v1
export LAYOUTLENS_API_KEY=...
export LAYOUTLENS_VISION_MODEL=...     # gets the screenshot in every condition
export LAYOUTLENS_TEXT_MODEL=...       # never gets the screenshot
make check-models    # does each model actually read an image? (a random number drawn in a PNG)
make models          # 6 screens × 5 conditions × 2 models × 2 repetitions
make score charts    # render the replicas, score them, chart them
```

Conditions, with the same wording except for the inputs: **baseline** (vision: the screenshot; text-only: nothing but the screen type and
size), **geometry**, **ocr**, **combined** (the analyzer's JSON, plus the screenshot for the vision model), and **truth** (perfect metadata
from the DOM, in the same format). Replicas are rendered in the same browser and scored on layout (box IoU), text (CER and position),
colour and SSIM. The scorer gives the original page 1.0 / 0.0 and a blank page 0.0 / 1.0 (`results/score_references.csv`).

The per-run scores (`results/model_runs.csv`) and summary (`results/model_summary.json`) are included. Run `make charts` to regenerate
the figures from these results. To re-score replicas, run `make models` to generate the raw model outputs in the local
`results/model_runs/` directory, then run `make score`.

## Layout

```
layoutlens/   geometry.py (OpenCV regions, ink, structure)   ocr.py (Tesseract)   analyze.py (modes, JSON)   cli.py
bench/        scenes.py  render.py  evaluate.py  ocr_variants.py  resources.py  truth.py  models.py  score.py  charts.py  metrics.py
data/         scenes/ (HTML, hidden from models)   screenshots/ (PNG + ground truth JSON)   screenshots-highres/
results/      CSV / JSON results, charts/
tests/
```

## Limitations

- **Synthetic, lossless screenshots.** Gradients, photos, transparency, JPEG artefacts and scaled captures were not tested. The colour
  grouping depends on flat colours.
- **Types are guesses.** The geometry measures boxes and colours well, but calling something a button, an input or a card is a heuristic.
  Icon and text labels are the least reliable.
- **One OCR engine, one language.** Tesseract 5 with English data. CER is per element against the DOM text, so a correct reading split
  across two elements counts as an error.
- **Small model study.** Two models, 6 screens × 2 repetitions per condition: large effects are clear, small ones are not.

Tesseract is a separate program (Apache-2.0) and is not distributed with this repository. The scenes use the system's default sans-serif
font; no fonts or third-party images are included.

## License

MIT
