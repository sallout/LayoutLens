"""Time, CPU and peak memory of each analysis mode at several resolutions.

    python -m bench.resources              writes results/resources.csv and results/resources_summary.json

Every (mode, image) pair runs in a fresh Python process that analyses the image REPS times. The process reports, per run, wall time and
CPU time (its own plus Tesseract's, which runs as a child process), and at the end its own peak resident memory and the largest peak of
any Tesseract child. The first run of each process is reported separately ("cold": imports and first-call costs); mean, median and p95 are
over the remaining runs. CPU % is CPU time over wall time (100 % = one core).

Measurement uses getrusage(2) only: no sampling thread, no extra dependency. ru_maxrss is in bytes on macOS and kilobytes on Linux.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPS = 6
IMAGES = [
    ("mobile 1x", "data/screenshots/mobile-complex-light@1x.png"),
    ("web 1x", "data/screenshots/web-complex-light@1x.png"),
    ("web 2x", "data/screenshots/web-complex-light@2x.png"),
    ("web 3x", "data/screenshots-highres/web-complex-light@3x.png"),
]

WORKER = r'''
import json, resource, sys, time
from layoutlens import analyze as ll
mode, path, reps = sys.argv[1], sys.argv[2], int(sys.argv[3])
unit = 1 if sys.platform == "darwin" else 1024
rgb = ll.load(path)
def cpu():
    s, c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit
runs = []
for i in range(reps):
    ll._cache.clear()
    c0, t0 = cpu(), time.perf_counter()
    out = ll.analyze(rgb, mode=mode)
    wall = time.perf_counter() - t0
    runs.append({"wall": wall, "cpu": cpu() - c0, "elements": len(out.get("elements", [])), "texts": len(out.get("texts", []))})
print(json.dumps({"runs": runs, "rss_before_mb": base / 2**20,
                  "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit / 2**20,
                  "tesseract_peak_rss_mb": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * unit / 2**20}))
'''


def p95(xs):
    xs = sorted(xs)
    k = (len(xs) - 1) * 0.95
    f = int(k)
    return xs[f] + (xs[min(f + 1, len(xs) - 1)] - xs[f]) * (k - f)


def main() -> None:
    from PIL import Image

    rows = []
    for label, rel in IMAGES:
        path = ROOT / rel
        w, h = Image.open(path).size
        for mode in ("geometry", "ocr", "combined"):
            proc = subprocess.run([sys.executable, "-c", WORKER, mode, str(path), str(REPS)], capture_output=True, text=True, cwd=ROOT, env={**os.environ})
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr[-2000:])
            r = json.loads(proc.stdout.strip().splitlines()[-1])
            warm = r["runs"][1:]
            walls = [x["wall"] for x in warm]
            row = {
                "image": label, "file": path.name, "width": w, "height": h, "megapixels": round(w * h / 1e6, 2), "mode": mode,
                "cold_s": round(r["runs"][0]["wall"], 3), "mean_s": round(statistics.mean(walls), 3), "median_s": round(statistics.median(walls), 3),
                "p95_s": round(p95(walls), 3), "sd_s": round(statistics.stdev(walls), 3),
                "cpu_pct": round(100 * sum(x["cpu"] for x in warm) / sum(walls), 1),
                "rss_before_mb": round(r["rss_before_mb"], 1), "peak_rss_mb": round(r["peak_rss_mb"], 1),
                "tesseract_peak_rss_mb": round(r["tesseract_peak_rss_mb"], 1),
                "elements": warm[-1]["elements"], "texts": warm[-1]["texts"],
            }
            rows.append(row)
            print(row, flush=True)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    with open(out / "resources.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    machine = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(), "logical_cores": os.cpu_count()}
    try:
        from layoutlens.ocr import tesseract_version
        machine["tesseract"] = tesseract_version()
    except Exception:  # noqa: BLE001
        pass
    (out / "resources_summary.json").write_text(json.dumps({"machine": machine, "reps": REPS, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
