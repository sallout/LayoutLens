"""Charts from the CSV/JSON results (whichever exist).

    python -m bench.charts        writes results/charts/*.png
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "charts"
COLOURS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]


def rows(name: str) -> list[dict]:
    path = RESULTS / name
    return list(csv.DictReader(open(path))) if path.exists() else []


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print("wrote", OUT / name)


def geometry_by_kind() -> None:
    data = rows("accuracy_geometry_by_kind.csv")
    if not data:
        return
    data.sort(key=lambda r: float(r["recall"] or 0))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh([f'{r["kind"]} ({r["truth"]})' for r in data], [float(r["recall"]) for r in data], color=COLOURS[0])
    ax.set_xlim(0, 1)
    ax.set_xlabel("recall (IoU ≥ 0.5)")
    ax.set_title("Geometry: detection by element kind")
    save(fig, "geometry_recall_by_kind.png")


def ocr_modes() -> None:
    summary = RESULTS / "accuracy_summary.json"
    if not summary.exists():
        return
    ocr = json.loads(summary.read_text())["ocr"]
    metrics = [("cer_mean", "CER (lower is better)"), ("text_recall", "texts found"), ("exact_match", "exact match")]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    width = 0.38
    for k, (mode, label) in enumerate((("ocr", "full-page OCR"), ("combined", "geometry-guided OCR"))):
        ax.bar([i + k * width for i in range(len(metrics))], [ocr[mode][m] for m, _ in metrics], width, label=label, color=COLOURS[k])
    ax.set_xticks([i + width / 2 for i in range(len(metrics))], [label for _, label in metrics])
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.set_title(f'OCR accuracy on {ocr["ocr"]["texts"]} text elements')
    save(fig, "ocr_modes.png")


def resources() -> None:
    data = rows("resources.csv")
    if not data:
        return
    by_mode = defaultdict(list)
    for r in data:
        by_mode[r["mode"]].append(r)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    for k, (mode, rs) in enumerate(sorted(by_mode.items())):
        rs.sort(key=lambda r: float(r["megapixels"]))
        mp = [float(r["megapixels"]) for r in rs]
        a.errorbar(mp, [float(r["mean_s"]) for r in rs], yerr=[float(r["sd_s"]) for r in rs], marker="o", label=mode, color=COLOURS[k], capsize=3)
        b.plot(mp, [float(r["peak_rss_mb"]) for r in rs], marker="o", label=mode, color=COLOURS[k])
    a.set_xlabel("megapixels")
    a.set_ylabel("seconds per image")
    a.set_title("Processing time (mean ± SD)")
    b.set_xlabel("megapixels")
    b.set_ylabel("peak RSS of the Python process (MB)")
    b.set_title("Peak memory")
    a.legend(frameon=False)
    save(fig, "resources.png")


def models() -> None:
    summary = RESULTS / "model_summary.json"
    if not summary.exists():
        return
    data = json.loads(summary.read_text())
    metrics = [("box_iou_all", "layout (box IoU, all boxes)"), ("text_recall", "texts reproduced (CER ≤ 0.2)"), ("ssim", "SSIM")]
    for role, info in data.items():
        conds = list(info["by_condition"])
        fig, axes = plt.subplots(1, len(metrics), figsize=(12, 3.4))
        for ax, (m, label) in zip(axes, metrics):
            means = [info["by_condition"][c]["all_runs"].get(m, {}).get("mean", 0) for c in conds]
            sds = [info["by_condition"][c]["all_runs"].get(m, {}).get("sd", 0) for c in conds]
            ax.bar(conds, means, yerr=sds, capsize=3, color=COLOURS[: len(conds)])
            ax.set_ylim(0, 1)
            ax.set_title(label)
            ax.tick_params(axis="x", rotation=30)
        fig.suptitle(f'{role} model ({info["model"]}): replication accuracy by condition, all runs (mean ± SD)')
        save(fig, f"models_{role}.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    geometry_by_kind()
    ocr_modes()
    resources()
    models()


if __name__ == "__main__":
    main()
