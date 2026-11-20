"""Structured metadata for a UI screenshot: geometry, OCR text, or both.

    from layoutlens import analyze
    data = analyze.analyze("screen.png", mode="combined")   # "geometry" | "ocr" | "combined"
    print(analyze.to_text(data))

All coordinates are pixels of the input image. Images above MAX_PIXELS are analysed at every n-th pixel (which keeps the exact flat
colours the geometry relies on) and their coordinates scaled back; geometry needs about 100 MB of memory per analysed megapixel.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from layoutlens import geometry, ocr

MODES = ("geometry", "ocr", "combined")
MAX_PIXELS = 8_000_000
MAX_ELEMENTS = 400
CACHE_SIZE = 16

_cache: OrderedDict[tuple[str, str], dict] = OrderedDict()
_cache_lock = threading.Lock()


def load(image: Any) -> np.ndarray:
    """An RGB uint8 array from a path, encoded bytes, a PIL image or an array."""
    if isinstance(image, np.ndarray):
        return image[..., :3] if image.ndim == 3 else np.stack([image] * 3, axis=-1)
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    if isinstance(image, (bytes, bytearray)):
        return np.asarray(Image.open(io.BytesIO(image)).convert("RGB"))
    return np.asarray(Image.open(Path(image)).convert("RGB"))


def geometry_full(rgb: np.ndarray) -> dict:
    """The full geometry result (geometry.analyze_rgb), with coordinates in pixels of `rgb` even when it was analysed at a reduced size."""
    h, w = rgb.shape[:2]
    step = math.ceil(math.sqrt(w * h / MAX_PIXELS)) if w * h > MAX_PIXELS else 1
    result = geometry.analyze_rgb(np.ascontiguousarray(rgb[::step, ::step]) if step > 1 else rgb)
    if step > 1:
        for e in result["elements"]:
            e["bbox"] = [v * step for v in e["bbox"]]
            if e.get("radius"):
                e["radius"] = round(e["radius"] * step, 1)
            if e.get("border"):
                e["border"]["sides"] = [v * step for v in e["border"]["sides"]]
                e["border"]["width"] *= step
        lay = result["layout"]
        for g in lay["gaps"]:
            g["gap"] *= step
        for p in lay["padding"]:
            for k in ("left", "top", "right", "bottom"):
                p[k] *= step
        for a in lay["alignment"]:
            a["at"] = round(a["at"] * step, 1)
        result["image"].update(width=w, height=h, analysed_step=step)
    return result


def _element(e: dict) -> dict:
    x, y, w, h = e["bbox"]
    d: dict[str, Any] = {"id": e["id"], "type": e["type"], "x": int(x), "y": int(y), "w": int(w), "h": int(h)}
    if e.get("fill"):
        d["fill"] = e["fill"]
    if e.get("border"):
        d["border"] = {"sides_px": [int(v) for v in e["border"]["sides"]], "color": e["border"]["color"]}
    if e.get("radius"):
        d["radius"] = e["radius"]
    if e["type"] in ("text", "icon") and e.get("color"):
        d["color"] = e["color"]
    if e.get("parent"):
        d["parent"] = e["parent"]
    return d


def _spacing(layout: dict, rename: dict[str, str] | None = None) -> dict:
    """Gaps, padding and alignment. `rename` maps geometry text boxes that were replaced by OCR text to the text's id."""
    r = (lambda i: rename.get(i, i)) if rename else (lambda i: i)
    return {
        "gaps": [{"a": r(g["a"]), "b": r(g["b"]), "axis": g["axis"], "px": int(g["gap"])} for g in layout["gaps"][:300]],
        "padding": [{"parent": p["parent"], "left": int(p["left"]), "top": int(p["top"]), "right": int(p["right"]), "bottom": int(p["bottom"])} for p in layout["padding"][:300]],
        "aligned": [{"edge": a["edge"], "at": a["at"], "ids": [r(i) for i in a["ids"]]} for a in layout["alignment"][:120]],
    }


def _overlap(a, b) -> float:
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy / float(max(1, a[2] * a[3]))


def _alignment(text_box, container) -> str:
    left = text_box[0] - container[0]
    right = container[0] + container[2] - (text_box[0] + text_box[2])
    if abs(left - right) <= max(2, 0.06 * container[2]):
        return "center"
    return "left" if left < right else "right"


def _combine(geo: dict, lines: list[dict]) -> tuple[list[dict], list[dict]]:
    """OCR text lines joined to the geometry: each line takes the colour of the ink it covers, the smallest region that contains it, and
    its alignment there. Geometry text boxes that OCR read are replaced by the text; the ones it missed stay as boxes without content."""
    els = geo["elements"]
    ink = [e for e in els if e["type"] == "text"]
    regions = sorted((e for e in els if e["source"] == "region" and e["type"] not in ("divider",)), key=lambda e: e["bbox"][2] * e["bbox"][3])
    used: dict[str, str] = {}
    texts = []
    for i, ln in enumerate(lines):
        box = ln["box"]
        covered = [e for e in ink if _overlap(e["bbox"], box) >= 0.5 or _overlap(box, e["bbox"]) >= 0.5]
        for e in covered:
            used.setdefault(e["id"], f"t{i}")
        item: dict[str, Any] = {"id": f"t{i}", "text": ln["text"], "x": box[0], "y": box[1], "w": box[2], "h": box[3], "conf": ln["conf"], "font_px": ln["font_px"]}
        if covered and covered[0].get("text_color"):
            item["color"] = covered[0]["text_color"]
        container = next((r for r in regions if _overlap(box, r["bbox"]) >= 0.9 and r["bbox"][2] * r["bbox"][3] > box[2] * box[3] * 1.2), None)
        if container:
            item["in"] = container["id"]
            item["align"] = _alignment(box, container["bbox"])
        texts.append(item)
    kept = [e for e in els if not (e["type"] == "text" and e["id"] in used)]
    return kept, texts, used


def analyze(image: Any, mode: str = "combined", *, psm: int = ocr.DEFAULT_PSM) -> dict:
    """Compact structured metadata for one image. Results are cached by image content and mode."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    rgb = load(image)
    key = (hashlib.sha256(rgb.tobytes() + str(rgb.shape).encode()).hexdigest(), f"{mode}:{psm}")
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    h, w = rgb.shape[:2]
    out: dict[str, Any] = {"image": {"width": w, "height": h}, "mode": mode}
    if mode in ("geometry", "combined"):
        geo = geometry_full(rgb)
        out.update(background=geo["colors"]["background"], theme=geo["colors"]["theme"], palette=[c["hex"] for c in geo["colors"]["top_colors"][:8]])
    if mode == "geometry":
        out["elements"] = [_element(e) for e in geo["elements"][:MAX_ELEMENTS]]
        out["spacing"] = _spacing(geo["layout"])
    elif mode == "ocr":
        lines = ocr.analyze_text(rgb, psm=psm)["lines"]
        out["texts"] = [{"id": f"t{i}", "text": ln["text"], "x": ln["box"][0], "y": ln["box"][1], "w": ln["box"][2], "h": ln["box"][3],
                         "conf": ln["conf"], "font_px": ln["font_px"]} for i, ln in enumerate(lines)]
    else:
        # OCR guided by the geometry: each text line the geometry found is read on its own (single characters and short labels that
        # full-page OCR skips), then full-page OCR adds any text the geometry missed.
        ink = [e for e in geo["elements"] if e["type"] == "text"]
        icons = [e["bbox"] for e in geo["elements"] if e["type"] in ("icon", "circle")]
        guided = [ln for ln in ocr.read_boxes(rgb, [e["bbox"] for e in ink]) if ln]
        extra = [ln for ln in ocr.analyze_text(rgb, psm=psm)["lines"]
                 if not any(_overlap(ln["box"], g["box"]) > 0.3 or _overlap(g["box"], ln["box"]) > 0.3 for g in guided)
                 and not any(_overlap(ln["box"], b) >= 0.5 for b in icons)  # an icon read as a letter
                 and not (len(ln["text"]) <= 2 and ln["conf"] < 70)]
        kept, texts, rename = _combine(geo, guided + extra)
        out["elements"] = [_element(e) for e in kept[:MAX_ELEMENTS]]
        out["texts"] = texts
        out["spacing"] = _spacing(geo["layout"], rename)
    with _cache_lock:
        _cache[key] = out
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    return out


def to_text(data: dict) -> str:
    """The metadata as a model reads it: one element or text per line."""
    lines = ["{"]
    for k, v in data.items():
        if k in ("elements", "texts"):
            lines.append(f'"{k}": [')
            lines += [json.dumps(e, separators=(",", ":")) + ("," if i < len(v) - 1 else "") for i, e in enumerate(v)]
            lines.append("],")
        else:
            lines.append(f"{json.dumps(k)}: {json.dumps(v, separators=(',', ':'))},")
    lines[-1] = lines[-1].rstrip(",")
    lines.append("}")
    return "\n".join(lines)
