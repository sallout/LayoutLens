"""Accuracy of the geometry and OCR analysis against the rendered ground truth.

    python -m bench.evaluate                       all of data/screenshots, writes results/accuracy_*.csv|json
    python -m bench.evaluate --only web-simple-light@1x

Geometry: only elements that can be seen are scored -- text, icons, avatars, images, chips, toggles, buttons, inputs and dividers, plus
containers whose own background differs from what is behind them or that have a border. Text and icons are compared by their ink box.
Detections and ground truth are matched one-to-one by IoU >= 0.5.

OCR: every visible text element of the ground truth is compared with the OCR lines that fall on it (at least half of a line's area inside
the element's box, slightly enlarged). CER and WER are edit distances over characters and words, capped at 1; a text with no OCR line
counts as 1.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from bench.metrics import centre_distance, cer, delta_e, iou, match_boxes, normalise_text, overlap_of, wer
from layoutlens import analyze as ll

ROOT = Path(__file__).resolve().parent.parent
EXPECTED = {
    "panel": {"box", "control", "pill"}, "button": {"button", "control", "pill"}, "input": {"input", "control"}, "text": {"text"},
    "icon": {"icon"}, "avatar": {"circle"}, "chip": {"pill"}, "toggle": {"pill"}, "divider": {"divider"}, "image": {"box", "icon", "control"},
}


def visible(e: dict) -> bool:
    if e["kind"] != "panel":
        return True
    return (e["own_fill"] and (delta_e(e["fill"], e["behind"]) or 0) > 1.0) or e["border_width"] > 0


def truth_box(e: dict) -> list[int]:
    return e["ink"] if e["kind"] in ("text", "icon") and e.get("ink") else e["box"]


def mean(xs):
    return round(statistics.mean(xs), 4) if xs else None


def share(xs, limit):
    return round(sum(1 for x in xs if x <= limit) / len(xs), 4) if xs else None


def score_geometry(gt: dict, geo: dict) -> tuple[dict, dict]:
    truth = [e for e in gt["elements"] if visible(e)]
    found = geo["elements"]
    pairs = match_boxes([truth_box(e) for e in truth], [d["bbox"] for d in found])
    ious, pos, size, fill_de, text_de, radius, labels = [], [], [], [], [], [], []
    border_tp = border_fp = border_fn = 0
    kinds = defaultdict(lambda: {"truth": 0, "found": 0, "iou": []})
    for e in truth:
        kinds[e["kind"]]["truth"] += 1
    matched = {}
    for i, j, v in pairs:
        t, d = truth[i], found[j]
        matched[t["name"]] = d
        tb, db = truth_box(t), d["bbox"]
        ious.append(v)
        pos += [abs(db[0] - tb[0]), abs(db[1] - tb[1])]
        size += [abs(db[2] - tb[2]), abs(db[3] - tb[3])]
        labels.append(d["type"] in EXPECTED[t["kind"]])
        kinds[t["kind"]]["found"] += 1
        kinds[t["kind"]]["iou"].append(v)
        if t["kind"] == "text":
            de = delta_e(d.get("text_color") or d.get("color"), t["text_color"])
            if de is not None:
                text_de.append(de)
        elif t["kind"] != "icon" and d["source"] == "region":
            if t["own_fill"] or t["kind"] in ("avatar", "image", "toggle", "chip"):
                de = delta_e(d["fill"], t["fill"])
                if de is not None:
                    fill_de.append(de)
            has_t, has_d = t["border_width"] > 0, bool(d.get("border"))
            border_tp += has_t and has_d
            border_fn += has_t and not has_d
            border_fp += has_d and not has_t
            if d.get("radius") is not None:
                radius.append(abs(d["radius"] - min(t["radius"], min(t["box"][2], t["box"][3]) / 2)))
    # spacing: nearest sibling to the right / below in the ground truth, compared with the matched detections
    names = {e["name"] for e in truth}
    by_name = {e["name"]: e for e in gt["elements"]}
    parent = {}
    for e in truth:
        p = e["parent"]
        while p is not None and p not in names:
            p = by_name[p]["parent"] if p in by_name else None
        parent[e["name"]] = p
    boxes = {e["name"]: truth_box(e) for e in truth}
    siblings = defaultdict(list)
    for n, p in parent.items():
        siblings[p].append(n)
    gap_err = []
    for kids in siblings.values():
        for a in kids:
            ax, ay, aw, ah = boxes[a]
            for axis in ("x", "y"):
                if axis == "x":
                    cand = [k for k in kids if k != a and boxes[k][0] >= ax + aw and min(ay + ah, boxes[k][1] + boxes[k][3]) - max(ay, boxes[k][1]) > 0.3 * min(ah, boxes[k][3])]
                    key = lambda k: boxes[k][0]  # noqa: E731
                else:
                    cand = [k for k in kids if k != a and boxes[k][1] >= ay + ah and min(ax + aw, boxes[k][0] + boxes[k][2]) - max(ax, boxes[k][0]) > 0.3 * min(aw, boxes[k][2])]
                    key = lambda k: boxes[k][1]  # noqa: E731
                if not cand:
                    continue
                b = min(cand, key=key)
                if a in matched and b in matched:
                    da, db = matched[a]["bbox"], matched[b]["bbox"]
                    if axis == "x":
                        gap_err.append(abs((db[0] - da[0] - da[2]) - (boxes[b][0] - ax - aw)))
                    else:
                        gap_err.append(abs((db[1] - da[1] - da[3]) - (boxes[b][1] - ay - ah)))
    tp = len(pairs)
    row = {
        "truth": len(truth), "found": len(found), "matched": tp,
        "precision": round(tp / len(found), 4) if found else 0.0, "recall": round(tp / len(truth), 4) if truth else 0.0,
        "mean_iou": mean(ious), "pos_err_px": mean(pos), "size_err_px": mean(size), "label_accuracy": mean([float(x) for x in labels]),
        "fill_de": mean(fill_de), "text_colour_de": mean(text_de), "radius_err_px": mean(radius), "radius_le_2px": share(radius, 2),
        "border_precision": round(border_tp / (border_tp + border_fp), 4) if border_tp + border_fp else None,
        "border_recall": round(border_tp / (border_tp + border_fn), 4) if border_tp + border_fn else None,
        "gap_err_px": mean(gap_err), "gap_le_1px": share(gap_err, 1),
        "background_de": round(delta_e(geo["colors"]["background"], gt["page_background"]), 3),
    }
    raw = {"iou": ious, "pos": pos, "size": size, "fill_de": fill_de, "gap": gap_err, "kinds": kinds}
    return row, raw


def _reading_order(lines: list[dict]) -> list[dict]:
    """Rows top to bottom (lines that overlap vertically by half share a row), left to right within a row."""
    rows: list[list[dict]] = []
    for ln in sorted(lines, key=lambda ln: ln["box"][1] + ln["box"][3] / 2):
        y, h = ln["box"][1], ln["box"][3]
        for row in rows:
            ry, rh = row[0]["box"][1], row[0]["box"][3]
            if min(y + h, ry + rh) - max(y, ry) >= 0.5 * min(h, rh):
                row.append(ln)
                break
        else:
            rows.append([ln])
    return [ln for row in rows for ln in sorted(row, key=lambda ln: ln["box"][0])]


def as_lines(data: dict) -> list[dict]:
    """The text items of an analysis, in the form score_ocr reads."""
    return [{"text": t["text"], "box": [t["x"], t["y"], t["w"], t["h"]], "conf": t["conf"], "font_px": t["font_px"]} for t in data.get("texts", [])]


def score_ocr(gt: dict, ocr: dict) -> tuple[dict, list[dict]]:
    texts = [e for e in gt["elements"] if e["kind"] == "text" and e.get("text") and e.get("ink")]
    lines = ocr["texts"]
    assigned = defaultdict(list)
    unassigned = 0
    for ln in lines:
        best, best_ov = None, 0.0
        for t in texts:
            x, y, w, h = t["ink"]
            pad = max(2, int(0.25 * h))
            ov = overlap_of(ln["box"], [x - pad, y - pad, w + 2 * pad, h + 2 * pad])
            if ov > best_ov:
                best, best_ov = t, ov
        if best is not None and best_ov >= 0.5:
            assigned[best["name"]].append(ln)
        else:
            unassigned += 1
    per = []
    for t in texts:
        got = _reading_order(assigned.get(t["name"], []))
        text = " ".join(ln["text"] for ln in got)
        item = {"name": t["name"], "truth": t["text"], "ocr": text, "cer": cer(text, t["text"]), "wer": wer(text, t["text"]),
                "exact": normalise_text(text) == normalise_text(t["text"]), "found": bool(got)}
        if got:
            x0 = min(ln["box"][0] for ln in got)
            y0 = min(ln["box"][1] for ln in got)
            box = [x0, y0, max(ln["box"][0] + ln["box"][2] for ln in got) - x0, max(ln["box"][1] + ln["box"][3] for ln in got) - y0]
            item.update(iou=iou(box, t["ink"]), centre_err=centre_distance(box, t["ink"]),
                        conf=float(np.mean([ln["conf"] for ln in got])))
            if len(got) == 1 and got[0].get("font_px"):
                item["font_err"] = abs(got[0]["font_px"] - t["font_px"]) / t["font_px"]
        per.append(item)
    chars = sum(len(normalise_text(p["truth"])) for p in per)
    row = {
        "texts": len(per), "ocr_lines": len(lines), "unassigned_lines": unassigned,
        "text_recall": mean([float(p["found"]) for p in per]), "exact_match": mean([float(p["exact"]) for p in per]),
        "cer_mean": mean([p["cer"] for p in per]), "wer_mean": mean([p["wer"] for p in per]),
        "cer_micro": round(sum(p["cer"] * len(normalise_text(p["truth"])) for p in per) / max(1, chars), 4),
        "placement_iou": mean([p["iou"] for p in per if "iou" in p]), "centre_err_px": mean([p["centre_err"] for p in per if "centre_err" in p]),
        "font_size_err": mean([p["font_err"] for p in per if "font_err" in p]), "mean_conf": mean([p["conf"] for p in per if "conf" in p]),
    }
    return row, per


def ocr_summary(items: list[dict], rows: list[dict], group) -> dict:
    return {"texts": len(items), "text_recall": mean([float(p["found"]) for p in items]), "exact_match": mean([float(p["exact"]) for p in items]),
            "cer_mean": mean([p["cer"] for p in items]), "cer_median": round(statistics.median(p["cer"] for p in items), 4), "wer_mean": mean([p["wer"] for p in items]),
            "placement_iou": mean([p["iou"] for p in items if "iou" in p]), "centre_err_px": mean([p["centre_err"] for p in items if "centre_err" in p]),
            "font_size_err": mean([p["font_err"] for p in items if "font_err" in p]), "mean_conf": mean([p["conf"] for p in items if "conf" in p]),
            "by_dpr_cer": group(rows, "dpr", "cer_mean"), "by_theme_cer": group(rows, "theme", "cer_mean"),
            "by_complexity_cer": {c: mean([r["cer_mean"] for r in rows if r["scene"].endswith(c)]) for c in ("simple", "medium", "complex")}}


def main() -> None:
    ap = argparse.ArgumentParser(description="Accuracy of geometry and OCR against the ground truth.")
    ap.add_argument("--data", default=str(ROOT / "data" / "screenshots"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    geo_rows, ocr_rows, ocr_items, kinds_all = [], [], [], defaultdict(lambda: {"truth": 0, "found": 0, "iou": []})
    raw_all = defaultdict(list)
    for gt_file in sorted(data.glob("*.json")):
        stem = gt_file.stem
        if args.only and stem not in args.only.split(","):
            continue
        gt = json.loads(gt_file.read_text())
        rgb = np.asarray(Image.open(data / gt["image"]).convert("RGB"))
        geo = ll.geometry_full(rgb)
        meta = {"image": stem, "scene": gt["scene"], "theme": gt["theme"], "dpr": gt["dpr"], "width": gt["width"], "height": gt["height"]}
        g, raw = score_geometry(gt, geo)
        geo_rows.append({**meta, **g})
        for mode in ("ocr", "combined"):
            o, per = score_ocr(gt, {"texts": as_lines(ll.analyze(rgb, mode=mode))})
            ocr_rows.append({**meta, "mode": mode, **o})
            ocr_items += [{"image": stem, "mode": mode, **p} for p in per]
        for k, v in raw["kinds"].items():
            kinds_all[k]["truth"] += v["truth"]
            kinds_all[k]["found"] += v["found"]
            kinds_all[k]["iou"] += v["iou"]
        for k in ("iou", "pos", "size", "fill_de", "gap"):
            raw_all[k] += raw[k]
        oo, cc = ocr_rows[-2], ocr_rows[-1]
        print(f'{stem:30s} geometry P={g["precision"]:.2f} R={g["recall"]:.2f} IoU={g["mean_iou"]} | CER ocr={oo["cer_mean"]} combined={cc["cer_mean"]}', flush=True)
    for name, rows in (("accuracy_geometry.csv", geo_rows), ("accuracy_ocr.csv", ocr_rows), ("accuracy_ocr_texts.csv", ocr_items)):
        with open(out / name, "w", newline="") as f:
            fields = list(dict.fromkeys(k for row in rows for k in row))
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    with open(out / "accuracy_geometry_by_kind.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "truth", "found", "recall", "mean_iou"])
        for k, v in sorted(kinds_all.items()):
            w.writerow([k, v["truth"], v["found"], round(v["found"] / v["truth"], 4) if v["truth"] else "", mean(v["iou"])])
    tp, found, truth = (sum(r[k] for r in geo_rows) for k in ("matched", "found", "truth"))
    p, r = tp / found, tp / truth

    def group(rows, key, field):
        g = defaultdict(list)
        for row in rows:
            g[str(row[key])].append(row[field])
        return {k: mean([x for x in v if x is not None]) for k, v in sorted(g.items())}

    summary = {
        "images": len(geo_rows),
        "geometry": {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4), "mean_iou": mean(raw_all["iou"]),
                     "iou_ge_0.9": share([1 - x for x in raw_all["iou"]], 0.1), "pos_err_px": mean(raw_all["pos"]), "size_err_px": mean(raw_all["size"]),
                     "fill_de": mean(raw_all["fill_de"]), "fill_de_le_2": share(raw_all["fill_de"], 2), "gap_err_px": mean(raw_all["gap"]),
                     "gap_le_1px": share(raw_all["gap"], 1), "by_dpr_recall": group(geo_rows, "dpr", "recall"), "by_theme_recall": group(geo_rows, "theme", "recall")},
        "ocr": {mode: ocr_summary([p for p in ocr_items if p["mode"] == mode], [r for r in ocr_rows if r["mode"] == mode], group) for mode in ("ocr", "combined")},
    }
    (out / "accuracy_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
