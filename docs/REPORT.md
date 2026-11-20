# LayoutLens: technical report

## Questions

1. How accurately can OpenCV/Pillow represent UI geometry and colours?
2. How much does Tesseract OCR add, on its own and together with the geometry?
3. Does structured metadata improve a vision-capable model beyond the screenshot alone?
4. How well can a text-only model recreate an interface from metadata alone?
5. What does each mode cost in time, CPU, memory and tokens?

All five questions are answered below; the model results are in the README's "Model results" section and `results/model_summary.json`.

## Dataset

Nine HTML scenes (`bench/scenes.py`):

| Platform | Scenes |
|---|---|
| Web | landing page, analytics dashboard, settings page |
| Desktop | dialog, mail client, code editor |
| Mobile | login, feed, cart |

Each scene has a light and a dark theme (CSS variables) and 10–90 scored elements.

Headless Chrome renders each scene at device pixel ratios 1 and 2, giving 36 screenshots. Three complex scenes are also rendered at 3× (up
to 4320×2700) for the cost measurements. Rendering is exact: CSS colours appear unchanged in the PNG, and a 1 CSS px border is exactly 1 or
2 device pixels.

The ground truth comes from the DOM of the same page (`bench/render.py`):
- clipped border boxes, fill and border colours per side, radius, and parent;
- text content, font size and colour;
- for text and icons, the *ink* box: the tight box of the pixels they paint, from a second render with them hidden;
- untagged elements that are visibly boxes (own background or border), included automatically.

Elements with nothing visible (layout-only groups) are excluded from scoring.

## Method

### Geometry (`layoutlens/geometry.py`)

1. **Colours.** The background is the commonest colour along the image border. The palette is an exact-colour histogram plus k-means.
2. **Regions.** Pixels are grouped by colour (64 levels per channel). Each colour's 4-connected components become regions if they are one of:
   - solid fills;
   - rectangular rings (a border on the same fill as the parent);
   - containers whose holes, once filled, leave a near-solid rectangle (a card mostly covered by an image).
3. **Borders and corners.** Borders are measured per side from the uniform band just outside a fill. The corner radius comes from the
   pixels a fill leaves empty in its corner squares: R = √(missing / (1 − π/4)).
4. **Ink.** A fill map is painted, rounded corners included. Pixels that differ from the fill under them are ink, grouped into text lines
   and icons.
5. **Clean-up.** A second pass removes regions that are pieces of large glyphs or icons. Container fills split by their own rows are
   joined back together.
6. **Structure.** Hierarchy, padding, gaps and alignment follow from the boxes.

### OCR (`layoutlens/ocr.py`)

- Tesseract 5.5 is called through its command line, with TSV output.
- Before recognition, the image is scaled so the median glyph is about 22 px tall. It is then converted to *distance from the local
  background*: the image minus a large median blur, inverted. Text of either polarity then becomes dark on white.
- Words are grouped into lines left to right. A word joins a line only if its vertical centre and height match the line's typical values.
- Short "words" much taller than typical text are icons read as letters, and are dropped.
- Font size is estimated from the line height and whether the text has ascenders or descenders.

### Combined mode (`layoutlens/analyze.py`)

**Geometry-guided OCR.**
1. Every text line found by the geometry is cropped and made dark-on-white per crop.
2. Each crop is scaled to the same height, and all are stacked into one column.
3. Tesseract reads the column in one call, as a uniform block.
4. Full-page OCR adds lines the geometry missed, except those over a detected icon and short low-confidence fragments.

Each text takes its colour from the geometry ink, its containing element, and its alignment inside that element. Spacing references to
geometry text boxes are renamed to the text ids.

**Output.** Compact JSON (see the README). `to_text` puts one element per line, which is the form given to models.

## Results

### 1. Geometry: 36 screenshots, 1,724 visible elements

Matching is greedy and one-to-one at IoU ≥ 0.5. Text and icons are compared by ink box.

| Metric | Value |
|---|---|
| Precision / recall / F1 | 0.923 / 0.942 / 0.932 |
| Mean IoU; share ≥ 0.9 | 0.977; 94.2% |
| Position / size error (mean, px) | 0.54 / 0.82 |
| Fill colour ΔE2000 | 0.0 on every matched fill |
| Gap between neighbouring siblings: error; ≤ 1 px | 0.72 px; 94.5% |

| Slice | Precision | Recall |
|---|---|---|
| Simple / medium / complex | 0.87 / 0.94 / 0.92 | 1.00 / 0.96 / 0.93 |
| Web / desktop / mobile | 0.92 / 0.91 / 0.95 | 0.95 / 0.95 / 0.92 |
| Light / dark | 0.92 / 0.92 | 0.94 / 0.94 |
| 1× / 2× | 0.95 / 0.90 | 0.93 / 0.95 |

**By kind** (recall): toggle, input, button, image, avatar 1.00; panel 0.99; icon 0.94; chip 0.93; text line 0.91; divider 0.50 (only 4).

**Where it fails.** The worst image is the landing page at 2×, with precision 0.62. Its 88 px heading has strokes 14 px wide, and the
letter bodies of the last word are flat-colour regions that each enclose their own anti-aliased edge pixels. The rule that removes glyph
pieces keeps a region that contains text, so these survive as "icons" and "boxes". A stricter rule would also remove real side-by-side
buttons, so it was left as a known limitation.

### 2. OCR: 876 text elements

**Choosing the settings** (`results/ocr_variants.csv`):

| Preprocessing | Mode 11 (sparse) | Mode 3 (auto) | Mode 12 |
|---|---|---|---|
| Plain (grey, inverted on dark pages): CER | 0.179 | 0.186 | 0.183 |
| Local-background distance: CER | 0.143 | **0.143** | 0.145 |
| Local-background distance: recall | 0.876 | **0.887** | 0.882 |

Local-background preprocessing is better in every mode. Mode 3 was chosen for its recall.

**Full-page OCR against geometry-guided OCR:**

| | Full-page | Guided (combined) |
|---|---|---|
| Texts found | 0.887 | 0.956 |
| Exact match | 0.799 | 0.888 |
| CER / WER (mean per element) | 0.143 / 0.178 | 0.058 / 0.088 |
| CER, simple / medium / complex screens (mean per image) | 0.026 / 0.066 / 0.217 | 0.001 / 0.031 / 0.093 |
| Placement IoU; centre error | 0.89; 1.6 px | 0.96; 0.6 px |
| Font size error (relative) | 7.8% | 6.8% |

Most remaining errors are single characters ("+", "−", line numbers), code with heavy punctuation, and the dense cart screen.

### 3. Cost

`results/resources.csv`; fresh process per mode and image, 1 cold and 5 warm runs; CPU % is CPU time over wall time.

| Image | Mode | Mean ± SD (s) | p95 | CPU | Peak RSS | Tesseract peak |
|---|---|---|---|---|---|---|
| 0.33 MP | geometry | 0.116 ± 0.003 | 0.118 | 172% | 97 MB | – |
| | ocr | 0.169 ± 0.001 | 0.171 | 201% | 90 MB | 47 MB |
| | combined | 0.411 ± 0.003 | 0.415 | 162% | 104 MB | 47 MB |
| 1.3 MP | geometry | 0.399 ± 0.003 | 0.402 | 268% | 195 MB | – |
| | ocr | 0.510 ± 0.004 | 0.515 | 243% | 139 MB | 97 MB |
| | combined | 1.233 ± 0.004 | 1.237 | 215% | 236 MB | 96 MB |
| 5.2 MP | geometry | 1.339 ± 0.015 | 1.358 | 322% | 555 MB | – |
| | ocr | 0.605 ± 0.005 | 0.610 | 259% | 208 MB | 109 MB |
| | combined | 2.302 ± 0.012 | 2.314 | 271% | 558 MB | 110 MB |
| 11.7 MP (geometry at every 2nd pixel) | geometry | 0.836 ± 0.010 | 0.846 | 255% | 481 MB | – |
| | ocr | 0.651 ± 0.009 | 0.664 | 273% | 355 MB | 111 MB |
| | combined | 1.830 ± 0.006 | 1.838 | 233% | 531 MB | 114 MB |

**What drives the cost:**
- **Geometry memory** grows with the pixels analysed: about 100 MB per megapixel.
- **OCR** grows slowly, because its input is rescaled to a fixed glyph size.
- **Combined mode** costs roughly geometry + OCR + one more Tesseract call.

## Model experiments

The pipeline is ready: `bench/models.py`, `bench/score.py` and `bench/truth.py`.

**Models.** Any OpenAI-compatible endpoint configured through `LAYOUTLENS_*` variables. `make check-models` first tests whether each model
reads an image: a random 4-digit number drawn in a PNG. "Native vision" is only claimed for a model that passes. The text-only model is
never sent an image.

**Design.**

| Item | Value |
|---|---|
| Screens | 6, at 1×, mixing platforms, complexity and themes |
| Conditions | 5: baseline, geometry, ocr, combined, truth |
| Models | 2: vision, text-only |
| Repetitions | 2, condition order rotated per repetition |

All prompts are identical except for their inputs, and all settings are identical. The scene HTML is never sent.

**Scoring.** Replicas are rendered in the same browser and read in the same way as the original:
- one-to-one box matching (box IoU over all original boxes, recall, precision, fill ΔE);
- per-text best-match CER, recall and position error;
- page SSIM and pixel ΔE.

Results are reported per condition as mean, SD, min, max and n, and as paired differences against the same model's baseline, with the
number of pairs that improved.

**Scorer validation** (`results/score_references.csv`): the original scene scores box IoU 1.0, text CER 0.0, SSIM 1.0. A blank page scores
0.0 / 1.0. On light screens SSIM gives a blank page 0.90–0.96, so SSIM alone says little for flat UIs; the box and text metrics are the
primary measures.

## What did not work, or needed correcting

- **Full-page OCR left alone** (TSV words, grouped top to bottom) produced word-order errors ("import from './app'; createApp") and merged
  rows: a tall avatar read as "O" linked a name with the line below it. It was fixed by grouping left to right, comparing against each
  line's typical centre and height, and dropping tall one- or two-character tokens. CER improved from 0.196 to 0.143.
- **Plain grey or global inversion** cannot handle mixed polarity (white text on blue buttons in a light theme). Local-background
  distance fixed most of that.
- **Containers holding large images were missed** (their visible fill is under half their box), and their borders then appeared as
  dividers. Fixed with the filled-container rule (feed screen, light, 2×: recall 0.27 → 1.00; precision 0.60–0.88 → 0.87–1.00 across the four feed screenshots).
- **That rule first accepted crisp 1 px closed borders as filled boxes.** A synthetic unit test caught it, and it was fixed with a minimum
  average thickness.
- **Very large display text at 2×** is still a source of false positives (see Results 1).

## Limitations

- **Synthetic, lossless screenshots** with flat colours and system fonts. Gradients, photos, blur, transparency, JPEG and scaled captures
  were not tested. The colour grouping would need tolerance changes for lossy input.
- **Element types are heuristic.** Box geometry and colour are the reliable outputs.
- **CER is computed per DOM text element.** A correct reading split differently from the DOM's elements is penalised.
- **The cost numbers come from one machine** (Apple M4 Pro, 12 cores, macOS).
- **Two models from one provider**; the text-only model's 10 runs truncated by the output limit are scored as written (or as an empty page).
- Rendering is near-deterministic: re-rendering changed at most 36 pixels by ≤ 11 levels in one 2× image.
