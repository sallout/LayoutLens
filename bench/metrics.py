"""Scoring helpers: box overlap, colour difference, edit distance, image similarity."""

from __future__ import annotations

import math

import cv2
import numpy as np


def iou(a, b) -> float:
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / float(a[2] * a[3] + b[2] * b[3] - inter) if inter else 0.0


def overlap_of(a, b) -> float:
    """Share of box a's area that lies inside box b."""
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy / float(max(1, a[2] * a[3]))


def centre_distance(a, b) -> float:
    return math.hypot(a[0] + a[2] / 2 - b[0] - b[2] / 2, a[1] + a[3] / 2 - b[1] - b[3] / 2)


def match_boxes(truth: list, found: list, threshold: float = 0.5) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matching by IoU (highest first), pairs at or above `threshold`: (truth index, found index, iou)."""
    pairs = sorted(((iou(t, f), i, j) for i, t in enumerate(truth) for j, f in enumerate(found)), reverse=True)
    used_t, used_f, out = set(), set(), []
    for v, i, j in pairs:
        if v < threshold:
            break
        if i in used_t or j in used_f:
            continue
        used_t.add(i)
        used_f.add(j)
        out.append((i, j, v))
    return out


def levenshtein(a, b) -> int:
    """Edit distance between two sequences (strings or token lists)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalise_text(s: str) -> str:
    return " ".join(s.split())


def cer(found: str, truth: str) -> float:
    truth, found = normalise_text(truth), normalise_text(found)
    return min(1.0, levenshtein(found, truth) / max(1, len(truth)))


def wer(found: str, truth: str) -> float:
    t, f = normalise_text(truth).split(), normalise_text(found).split()
    return min(1.0, levenshtein(f, t) / max(1, len(t)))


def _rgb(hex_colour: str) -> np.ndarray:
    h = hex_colour.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float)


def _lab(rgb: np.ndarray) -> np.ndarray:
    c = rgb / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    xyz = m @ c / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def delta_e(a: str | None, b: str | None) -> float | None:
    """CIEDE2000 difference between two hex colours (about 1 is just noticeable)."""
    if not a or not b:
        return None
    L1, a1, b1 = _lab(_rgb(a))
    L2, a2, b2 = _lab(_rgb(b))
    C1, C2 = math.hypot(a1, b1), math.hypot(a2, b2)
    Cm = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt(Cm ** 7 / (Cm ** 7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p, h2p = math.degrees(math.atan2(b1, a1p)) % 360, math.degrees(math.atan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    dhp = 0.0 if C1p * C2p == 0 else (h2p - h1p - 360 if h2p - h1p > 180 else h2p - h1p + 360 if h2p - h1p < -180 else h2p - h1p)
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2))
    Lm, Cmp = (L1 + L2) / 2, (C1p + C2p) / 2
    if C1p * C2p == 0:
        hm = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hm = (h1p + h2p) / 2
    else:
        hm = (h1p + h2p + 360) / 2 if h1p + h2p < 360 else (h1p + h2p - 360) / 2
    T = 1 - 0.17 * math.cos(math.radians(hm - 30)) + 0.24 * math.cos(math.radians(2 * hm)) + 0.32 * math.cos(math.radians(3 * hm + 6)) - 0.20 * math.cos(math.radians(4 * hm - 63))
    dth = 30 * math.exp(-(((hm - 275) / 25) ** 2))
    Rc = 2 * math.sqrt(Cmp ** 7 / (Cmp ** 7 + 25 ** 7))
    Sl = 1 + 0.015 * (Lm - 50) ** 2 / math.sqrt(20 + (Lm - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cmp, 1 + 0.015 * Cmp * T
    Rt = -math.sin(math.radians(2 * dth)) * Rc
    return math.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM of two RGB images (grey, 11x11 Gaussian window). `b` is resized to `a` if they differ."""
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    y = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5)  # noqa: E731
    mx, my = blur(x), blur(y)
    sx, sy, sxy = blur(x * x) - mx ** 2, blur(y * y) - my ** 2, blur(x * y) - mx * my
    return float((((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sx + sy + c2))).mean())


def pixel_delta_e(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-pixel CIE76 colour difference."""
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    la = cv2.cvtColor(a.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    lb = cv2.cvtColor(b.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    return float(np.linalg.norm(la - lb, axis=2).mean())
