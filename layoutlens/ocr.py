"""Text extraction with the Tesseract command-line tool.

Tesseract is called directly (``tesseract <image> stdout --psm N tsv``); set TESSERACT_CMD if the binary is not on PATH.

UI screenshots are hard for Tesseract in two ways: text is small (11-14 px at 1x), and its polarity changes across the image (dark text on
light cards, white text on a blue button, a whole dark theme). Before recognition the image is therefore

* scaled so the median glyph is about TARGET_GLYPH_PX tall, and
* turned into "distance from the local background": each pixel minus a large median-blurred copy of the image, so any text that stands out
  from what surrounds it becomes dark on white, whatever its own colour.

Words are grouped into lines (same baseline band, gaps under about one glyph height), and each line gets an estimated font size from its
height and the shape of its letters.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

TARGET_GLYPH_PX = 22
DEFAULT_PSM = 3  # automatic layout: measured best on the dataset (results/ocr_variants.csv)


class TesseractNotFound(RuntimeError):
    pass


def tesseract_cmd() -> str:
    cmd = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if not cmd:
        for candidate in ("/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract", "/usr/bin/tesseract"):
            if Path(candidate).exists():
                return candidate
        raise TesseractNotFound("tesseract not found: install it (e.g. `brew install tesseract`, `apt install tesseract-ocr`) or set TESSERACT_CMD")
    return cmd


def tesseract_version() -> str:
    out = subprocess.run([tesseract_cmd(), "--version"], capture_output=True, text=True).stdout
    return out.splitlines()[0] if out else "unknown"


@dataclass
class Word:
    text: str
    box: tuple[int, int, int, int]  # x, y, w, h in pixels of the original image
    conf: float


def glyph_height(gray: np.ndarray) -> float:
    """Median height of small dark-or-light blobs that stand out from their surroundings: a cheap estimate of the text size."""
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 12)
    hs = []
    for mask in (255 - edges, edges):
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        hs.extend(int(h) for _, _, w, h, a in stats[1:] if 4 <= h <= 80 and w <= 3 * h and a >= 6)
    return float(np.median(hs)) if hs else 12.0


def prepare(rgb: np.ndarray, *, mode: str = "local") -> tuple[np.ndarray, float]:
    """The image Tesseract reads, and the scale applied to it. mode: "local" (local-background contrast) or "plain" (grey, inverted for dark backgrounds)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    scale = float(np.clip(TARGET_GLYPH_PX / max(glyph_height(gray), 1.0), 1.0, 4.0))
    if scale > 1.05:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    else:
        scale = 1.0
    if mode == "plain":
        border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
        return (255 - gray if np.median(border) < 110 else gray), scale
    k = int(TARGET_GLYPH_PX * 3) | 1
    background = cv2.medianBlur(gray, k)
    diff = cv2.absdiff(gray, background)
    out = 255 - cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    return out, scale


def run_tesseract(image: np.ndarray, *, psm: int = DEFAULT_PSM, lang: str = "eng") -> list[dict]:
    """Raw word rows from Tesseract's TSV output (pixel units of `image`)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "in.png"
        cv2.imwrite(str(path), image)
        proc = subprocess.run([tesseract_cmd(), str(path), "stdout", "--psm", str(psm), "-l", lang, "tsv"], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"tesseract failed: {proc.stderr.strip()[:300]}")
    rows = []
    lines = proc.stdout.splitlines()
    header = lines[0].split("\t") if lines else []
    for line in lines[1:]:
        cells = line.split("\t")
        if len(cells) != len(header):
            continue
        row = dict(zip(header, cells))
        if row.get("level") != "5" or not row.get("text", "").strip():
            continue
        rows.append(row)
    return rows


def recognise(rgb: np.ndarray, *, psm: int = DEFAULT_PSM, mode: str = "local", min_conf: float = 30.0) -> list[Word]:
    image, scale = prepare(rgb, mode=mode)
    words = []
    for row in run_tesseract(image, psm=psm):
        conf = float(row["conf"])
        text = row["text"].strip()
        if conf < min_conf or not (any(ch.isalnum() for ch in text) or conf >= 60):  # punctuation alone only when Tesseract is fairly sure
            continue
        x, y, w, h = (int(round(int(row[k]) / scale)) for k in ("left", "top", "width", "height"))
        words.append(Word(text, (x, y, max(1, w), max(1, h)), conf))
    return words


_ASCENDERS = set("bdfhklt0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ!?$%&#@/\\|()[]{}")
_DESCENDERS = set("gjpqy,;()[]{}|")


def font_size(text: str, height: int) -> float:
    """Approximate CSS font size from a line's ink height. Typical sans-serif metrics: caps/ascenders ~0.72 em, x-height ~0.52 em,
    ascender to descender ~0.93 em."""
    asc = any(ch in _ASCENDERS for ch in text)
    desc = any(ch in _DESCENDERS for ch in text)
    ratio = 0.93 if asc and desc else 0.72 if asc else 0.74 if desc else 0.52
    return round(height / ratio, 1)


def group_lines(words: list[Word]) -> list[dict]:
    """Words left to right on the same row (centre within half a word height of the line's typical centre, similar height) with gaps
    under about one glyph height form a line. Very short "words" much taller than the typical word are icons or avatars read as letters
    and are dropped."""
    if not words:
        return []
    typical = float(np.median([w.box[3] for w in words]))
    words = [w for w in words if not (w.box[3] > 1.8 * typical and len(w.text) <= 2)]
    lines: list[dict] = []
    for w in sorted(words, key=lambda w: w.box[0]):
        x, y, bw, bh = w.box
        cy = y + bh / 2
        best = None
        for ln in lines:
            lx, _, lw, _ = ln["box"]
            same_row = abs(cy - ln["cy"]) <= 0.5 * min(bh, ln["h_med"]) and max(bh, ln["h_med"]) <= 1.8 * min(bh, ln["h_med"])
            if same_row and x - (lx + lw) <= 1.0 * max(bh, ln["h_med"]):
                best = ln
                break
        if best is None:
            lines.append({"words": [w], "box": list(w.box), "h_med": bh, "cy": cy})
        else:
            best["words"].append(w)
            lx, ly, lw, lh = best["box"]
            x0, y0 = min(lx, x), min(ly, y)
            best["box"] = [x0, y0, max(lx + lw, x + bw) - x0, max(ly + lh, y + bh) - y0]
            best["h_med"] = float(np.median([v.box[3] for v in best["words"]]))
            best["cy"] = float(np.median([v.box[1] + v.box[3] / 2 for v in best["words"]]))
    out = []
    for ln in lines:
        ws = sorted(ln["words"], key=lambda w: w.box[0])
        text = " ".join(w.text for w in ws)
        out.append({"text": text, "box": ln["box"], "conf": round(float(np.mean([w.conf for w in ws])), 1),
                    "font_px": font_size(text, ln["box"][3]), "words": len(ws)})
    return sorted(out, key=lambda ln: (ln["box"][1], ln["box"][0]))


def read_boxes(rgb: np.ndarray, boxes: list, *, line_px: int = 32, gap: int = 24) -> list[dict | None]:
    """Reads one line of text from each box (e.g. the text lines found by the geometry analysis) with a single Tesseract call: every
    crop is converted to dark-on-white, scaled to `line_px` tall and stacked into one column; the result is mapped back by position.
    Returns, per box, {"text", "box", "conf", "font_px"} or None when nothing was read."""
    if not boxes:
        return []
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h_img, w_img = gray.shape
    strips, spans, width = [], [], 0
    for x, y, w, h in boxes:
        pad = max(2, int(0.3 * h))
        x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(w_img, x + w + pad), min(h_img, y + h + pad)
        crop = gray[y0:y1, x0:x1]
        if crop.size == 0:
            strips.append(None)
            continue
        border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
        bg = float(np.median(border))
        ink = crop[np.abs(crop.astype(float) - bg) > 40]
        if ink.size and float(np.median(ink)) > bg:  # light text on a dark fill: invert so text is dark
            crop = 255 - crop
        scale = line_px / max(1, h + 2 * pad) * 1.6
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        crop = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX)
        strips.append(crop)
        width = max(width, crop.shape[1])
    canvas_h = sum(s.shape[0] + gap for s in strips if s is not None) + gap
    canvas = np.full((canvas_h, width + 2 * gap), 255, np.uint8)
    top = gap
    for s in strips:
        if s is None:
            spans.append(None)
            continue
        canvas[top:top + s.shape[0], gap:gap + s.shape[1]] = s
        spans.append((top, top + s.shape[0]))
        top += s.shape[0] + gap
    words: list[list[tuple[str, float, int]]] = [[] for _ in boxes]
    for row in run_tesseract(canvas, psm=6):
        conf, text = float(row["conf"]), row["text"].strip()
        if conf < 20 or not text:
            continue
        cy = int(row["top"]) + int(row["height"]) / 2
        for i, span in enumerate(spans):
            if span and span[0] - gap / 2 <= cy <= span[1] + gap / 2:
                words[i].append((text, conf, int(row["left"])))
                break
    out: list[dict | None] = []
    for box, ws in zip(boxes, words):
        if not ws:
            out.append(None)
            continue
        ws.sort(key=lambda t: t[2])
        text = " ".join(t[0] for t in ws)
        out.append({"text": text, "box": list(box), "conf": round(float(np.mean([t[1] for t in ws])), 1), "font_px": font_size(text, box[3])})
    return out


def analyze_text(rgb: np.ndarray, *, psm: int = DEFAULT_PSM, mode: str = "local") -> dict:
    """Text lines with content, box, mean word confidence (0-100) and estimated font size, in pixels of the image."""
    return {"engine": tesseract_version(), "psm": psm, "lines": group_lines(recognise(rgb, psm=psm, mode=mode))}
