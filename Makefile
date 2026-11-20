# Reproduces the dataset and every result. Requires: uv, Google Chrome (or `playwright install chromium` and LAYOUTLENS_BROWSER=chromium),
# and tesseract on PATH (or TESSERACT_CMD).
PY := .venv/bin/python

.PHONY: setup data accuracy ocr-variants resources check-models models score charts test all

setup:
	uv venv .venv --python 3.12
	uv pip install --python $(PY) -e '.[bench,dev]'

data:
	$(PY) -m bench.scenes
	$(PY) -m bench.render
	$(PY) -m bench.render --dpr 3 --themes light --scenes web-complex,desktop-complex,mobile-complex --out data/screenshots-highres

accuracy:
	$(PY) -m bench.evaluate

ocr-variants:
	$(PY) -m bench.ocr_variants

resources:
	$(PY) -m bench.resources

check-models:
	$(PY) -m bench.models check

models:
	$(PY) -m bench.models run --reps 2

score:
	$(PY) -m bench.score --references
	$(PY) -m bench.score

charts:
	$(PY) -m bench.charts

test:
	$(PY) -m pytest -q

all: data accuracy ocr-variants resources charts test
