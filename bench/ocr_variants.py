"""Chooses the OCR preprocessing and page segmentation mode by measured error on the dataset.

    python -m bench.ocr_variants        writes results/ocr_variants.csv

Variants: preprocessing "plain" (grey, inverted when the page background is dark) or "local" (distance from the local background, so text
of either polarity becomes dark on white), each with Tesseract page segmentation modes 11 (sparse text), 3 (automatic) and 12 (sparse
text with orientation detection).
"""

from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image

from bench.evaluate import score_ocr
from layoutlens import ocr

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    data = ROOT / "data" / "screenshots"
    rows = []
    for prep in ("plain", "local"):
        for psm in (11, 3, 12):
            cers, recalls, exact, secs = [], [], [], []
            per_dpr: dict[str, list[float]] = {}
            for gt_file in sorted(data.glob("*.json")):
                gt = json.loads(gt_file.read_text())
                rgb = np.asarray(Image.open(data / gt["image"]).convert("RGB"))
                t0 = time.perf_counter()
                lines = ocr.analyze_text(rgb, psm=psm, mode=prep)["lines"]
                secs.append(time.perf_counter() - t0)
                texts = [{"text": ln["text"], "box": ln["box"], "conf": ln["conf"], "font_px": ln["font_px"]} for ln in lines]
                row, per = score_ocr(gt, {"texts": texts})
                cers += [p["cer"] for p in per]
                recalls += [float(p["found"]) for p in per]
                exact += [float(p["exact"]) for p in per]
                per_dpr.setdefault(str(gt["dpr"]), []).extend(p["cer"] for p in per)
            rows.append({"prep": prep, "psm": psm, "texts": len(cers), "cer_mean": round(statistics.mean(cers), 4), "cer_median": round(statistics.median(cers), 4),
                         "exact_match": round(statistics.mean(exact), 4), "text_recall": round(statistics.mean(recalls), 4),
                         **{f"cer_dpr_{k}": round(statistics.mean(v), 4) for k, v in sorted(per_dpr.items())}, "seconds_per_image": round(statistics.mean(secs), 3)})
            print(rows[-1], flush=True)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    with open(out / "ocr_variants.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
