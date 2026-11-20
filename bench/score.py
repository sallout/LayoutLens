"""Renders the models' HTML replicas and scores them against the original screens.

    python -m bench.score                 every results/model_runs/*/*/replica.html -> results/model_runs.csv, results/model_summary.json
    python -m bench.score --references    scores the original scene against itself and a blank page (sanity bounds for the metrics)

Both the original and the replica are read the same way (bench/render.py page_items): visible boxes (own background unlike what is behind,
or a border) and text runs. Scores, in CSS pixels:

* boxes: one-to-one IoU matching (>= 0.5) of the original's boxes against the replica's: recall, precision, mean IoU of matches, and
  mean IoU over all original boxes with unmatched ones counted as 0 (box_iou_all); fill colour difference of matched boxes.
* text: for every text run of the original, the replica text run with the lowest CER (the nearest one when several tie, as repeated labels
  do). text_cer is that CER (1 when nothing is close);
  text_recall is the share with CER <= 0.2; for those, the centre distance (text_pos_err) and colour difference.
* page: SSIM and mean per-pixel colour difference of the two renders, and the page background colour difference.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from bench.metrics import centre_distance, cer, delta_e, match_boxes, pixel_delta_e, ssim
from bench.models import CONDITIONS, DATA, RUNS, TASKS, extract_html
from bench.render import SCENES, Renderer

ROOT = Path(__file__).resolve().parent.parent


def score_page(orig: dict, orig_png: np.ndarray, rep: dict, rep_png: np.ndarray) -> dict:
    ob, rb = orig["boxes"], rep["boxes"]
    pairs = match_boxes([b["box"] for b in ob], [b["box"] for b in rb])
    ious = [v for _, _, v in pairs]
    fill = [d for i, j, _ in pairs if (d := delta_e(ob[i]["fill"], rb[j]["fill"])) is not None]
    best_cer, pos, colour = [], [], []
    for t in orig["texts"]:
        cands = [(cer(r["text"], t["text"]), centre_distance(t["box"], r["box"]), r) for r in rep["texts"]]
        c, _, r = min(cands, key=lambda x: (x[0], x[1])) if cands else (1.0, 0.0, None)  # repeated texts ("Paid"): the nearest of the best matches
        best_cer.append(c)
        if r is not None and c <= 0.2:
            pos.append(centre_distance(t["box"], r["box"]))
            if (d := delta_e(t["color"], r["color"])) is not None:
                colour.append(d)
    mean = lambda xs: round(statistics.mean(xs), 4) if xs else None  # noqa: E731
    return {
        "boxes": len(ob), "replica_boxes": len(rb),
        "box_recall": round(len(pairs) / len(ob), 4) if ob else None, "box_precision": round(len(pairs) / len(rb), 4) if rb else 0.0,
        "box_iou": mean(ious), "box_iou_all": round(sum(ious) / len(ob), 4) if ob else None, "fill_de": mean(fill),
        "texts": len(orig["texts"]), "text_cer": mean(best_cer), "text_recall": round(sum(1 for c in best_cer if c <= 0.2) / len(best_cer), 4) if best_cer else None,
        "text_pos_err": mean(pos), "text_colour_de": mean(colour),
        "ssim": round(ssim(orig_png, rep_png), 4), "pixel_de": round(pixel_delta_e(orig_png, rep_png), 3),
        "background_de": round(delta_e(orig["page_background"], rep["page_background"]) or 0.0, 3),
    }


def _png(b: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))


def references(renderer: Renderer) -> list[dict]:
    rows = []
    for task in TASKS:
        gt = json.loads((DATA / f"{task}.json").read_text())
        orig_png = np.asarray(Image.open(DATA / gt["image"]).convert("RGB"))
        css = (SCENES / "theme.css").read_text()
        html = (SCENES / f'{gt["scene"]}.html').read_text().replace('<link rel="stylesheet" href="theme.css">', f"<style>{css}</style>")
        if gt["theme"] == "dark":
            html = html.replace("<html>", '<html class="dark">', 1)
        for name, page in (("original", html), ("blank", "<html><body></body></html>")):
            png, items = renderer.page_items(page, gt["css_width"], gt["css_height"])
            rows.append({"task": task, "reference": name, **score_page(gt["page_items"], orig_png, items, _png(png))})
            print(rows[-1], flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Score model replicas.")
    ap.add_argument("--references", action="store_true")
    args = ap.parse_args()
    out = ROOT / "results"
    renderer = Renderer()
    try:
        if args.references:
            rows = references(renderer)
            with open(out / "score_references.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            return
        rows = []
        for meta_file in sorted(RUNS.glob("*/*/meta.json")):
            run = meta_file.parent
            meta = json.loads(meta_file.read_text())
            gt = json.loads((DATA / f'{meta["task"]}.json').read_text())
            orig_png = np.asarray(Image.open(DATA / gt["image"]).convert("RGB"))
            response = (run / "response.txt").read_text()
            html = extract_html(response)
            status = "complete"
            if not html.strip():
                html = extract_html(response, partial=True)
                status = "truncated" if html.strip() else "failed"
            if status == "failed":
                html = "<html><body></body></html>"  # no output: scored as an empty page
            png, items = renderer.page_items(html, gt["css_width"], gt["css_height"])
            (run / "replica.png").write_bytes(png)
            score = score_page(gt["page_items"], orig_png, items, _png(png))
            rows.append({k: meta.get(k) for k in ("role", "model", "task", "condition", "rep", "seconds", "prompt_tokens", "completion_tokens", "reasoning_tokens", "metadata_chars")}
                        | {"status": status, "failure": meta.get("error") or ("hit the output token limit" if meta.get("completion_tokens") and meta["completion_tokens"] >= meta["settings"]["max_tokens"] else None)}
                        | score)
            print(run.parent.name, run.name, status, {k: score.get(k) for k in ("box_iou_all", "text_cer", "ssim")}, flush=True)
    finally:
        renderer.close()
    if not rows:
        raise SystemExit("No model runs found under results/model_runs/ (run `python -m bench.models run` first).")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "model_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (out / "model_summary.json").write_text(json.dumps(summarise(rows), indent=1))


METRICS = ["box_iou_all", "box_recall", "box_precision", "fill_de", "text_cer", "text_recall", "text_pos_err", "text_colour_de", "ssim", "pixel_de",
           "seconds", "prompt_tokens", "completion_tokens"]
HIGHER_IS_BETTER = {"box_iou_all", "box_recall", "box_precision", "text_recall", "ssim"}


def _stats(xs: list[float]) -> dict:
    return {"mean": round(statistics.mean(xs), 4), "sd": round(statistics.stdev(xs), 4) if len(xs) > 1 else 0.0, "min": round(min(xs), 4), "max": round(max(xs), 4), "n": len(xs)}


def summarise(rows: list[dict]) -> dict:
    """Per model role and condition: run counts by status; mean, SD, min, max, n of every metric over all runs (a run without output is
    scored as an empty page) and over completed runs only; and paired differences against the same role's baseline (same task and
    repetition, all runs), with how many pairs improved."""
    out: dict = {}
    for role in sorted({r["role"] for r in rows}):
        rr = [r for r in rows if r["role"] == role]
        out[role] = {"model": rr[0]["model"], "by_condition": {}, "paired_vs_baseline": {}}
        for c in CONDITIONS:
            cs = [r for r in rr if r["condition"] == c]
            if not cs:
                continue
            entry = {"runs": len(cs), "complete": sum(r["status"] == "complete" for r in cs), "truncated": sum(r["status"] == "truncated" for r in cs),
                     "failed": sum(r["status"] == "failed" for r in cs), "all_runs": {}, "completed_only": {}}
            for m in METRICS:
                xs = [r[m] for r in cs if r.get(m) is not None]
                done = [r[m] for r in cs if r.get(m) is not None and r["status"] == "complete"]
                if xs:
                    entry["all_runs"][m] = _stats(xs)
                if done:
                    entry["completed_only"][m] = _stats(done)
            out[role]["by_condition"][c] = entry
        base = {(r["task"], r["rep"]): r for r in rr if r["condition"] == "baseline"}
        for c in CONDITIONS[1:]:
            diffs = defaultdict(list)
            for r in rr:
                b = base.get((r["task"], r["rep"]))
                if r["condition"] != c or not b:
                    continue
                for m in METRICS:
                    if r.get(m) is not None and b.get(m) is not None:
                        diffs[m].append(r[m] - b[m])
            out[role]["paired_vs_baseline"][c] = {
                m: {"mean_diff": round(statistics.mean(v), 4), "sd": round(statistics.stdev(v), 4) if len(v) > 1 else 0.0, "n": len(v),
                    "better": sum(1 for x in v if (x > 0 if m in HIGHER_IS_BETTER else x < 0))}
                for m, v in diffs.items()}
    return out


if __name__ == "__main__":
    main()
