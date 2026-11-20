"""Geometry analysis of UI screenshots with OpenCV, Pillow and numpy.

No OCR and no learned model: the image is read as flat-coloured regions and "ink" on top of them.

1. Colours. The background is the most common colour along the image border. The palette is an exact-colour histogram (UIs use few flat
   colours) plus a small k-means.
2. Regions. Pixels are grouped by colour, quantised to 64 levels per channel. Each colour's 4-connected components that are solid (a fill)
   or a rectangular ring (a border drawn on the same fill as its parent) become regions. Borders are measured per side from the uniform
   band just outside a fill. The corner radius comes from how many pixels the fill leaves empty in each corner square. A container whose
   fill is cut into pieces by its own rows is joined back together.
3. Ink. A fill map paints every region's colour over its box. Pixels that differ from the fill under them are text glyphs or icons; they
   are grouped into text lines (gaps up to about one glyph height) and icons (squarish, taller than text, no similar neighbour).
4. Structure. An element's parent is the smallest region containing it; padding, gaps and edge alignment follow from the boxes.

Element types and confidence values are heuristics. Positions, sizes and colours are measurements of the image.
The method assumes lossless screenshots with flat colours; gradients, photos and JPEG artefacts break the colour grouping.
"""

from __future__ import annotations

import math
import time

import cv2
import numpy as np

cv2.ocl.setUseOpenCL(False)


def hexc(c) -> str:
    return "#%02x%02x%02x" % tuple(int(round(v)) for v in c[:3])




def iou(a, b) -> float:
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / float(a[2] * a[3] + b[2] * b[3] - inter) if inter else 0.0


def _gap_x(a, b) -> int:
    return max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2])


def _gap_y(a, b) -> int:
    return max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3])


def _union(a, b) -> list[int]:
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    return [x0, y0, max(a[0] + a[2], b[0] + b[2]) - x0, max(a[1] + a[3], b[1] + b[3]) - y0]


def _merge(boxes: list[list[int]], close) -> list[list[int]]:
    """Merges boxes for which close(a, b) holds, transitively (sweep in x order; repeat until nothing changes)."""
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        boxes.sort(key=lambda b: b[0])
        out: list[list[int]] = []
        for b in boxes:
            for o in reversed(out[-60:]):
                if close(o, b):
                    o[:] = _union(o, b)
                    changed = True
                    break
            else:
                out.append(b)
        boxes = out
    return boxes


def _on_border(line, r) -> bool:
    """Whether a thin line lies along one side of region r's box (it is that region's border)."""
    if not r.get("border"):
        return False
    x, y, w, h = r["bbox"]
    lx, ly, lw, lh = line
    horizontal = lw >= lh
    if horizontal:
        return lx >= x - 2 and lx + lw <= x + w + 2 and (abs(ly - y) <= 2 or abs(ly + lh - (y + h)) <= 2)
    return ly >= y - 2 and ly + lh <= y + h + 2 and (abs(lx - x) <= 2 or abs(lx + lw - (x + w)) <= 2)


def _mask_radius(comp: np.ndarray) -> float:
    """Corner radius from a fill's own mask: a quarter-circle corner of radius R leaves (1 - pi/4) R^2 pixels of its corner square empty. Counted from
    each corner as the run of empty pixels before the fill starts, row by row, then the median of the four corners (one may be cut by a neighbour)."""
    h, w = comp.shape
    lim = min(h, w) // 2
    rs = []
    for c in (comp, comp[:, ::-1], comp[::-1, :], comp[::-1, ::-1]):
        missing = 0
        for row in c[:lim, :lim]:
            run = int(np.argmax(row)) if row.any() else lim
            if run == 0:
                break
            missing += run
        rs.append(math.sqrt(missing / (1 - math.pi / 4)) if missing else 0.0)
    return float(np.median(rs))


def contrast(a, b) -> float:
    """Euclidean distance in RGB (0..441): a cheap, monotone contrast measure used for thresholds and confidence."""
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


# ----------------------------------------------------------------------------------------------- analysis
class Analyzer:
    def __init__(self, rgb: np.ndarray) -> None:
        self.rgb = rgb
        self.h, self.w = rgb.shape[:2]
        self.timings: dict[str, float] = {}

    def _t(self, name: str, t0: float) -> float:
        now = time.perf_counter()
        self.timings[name] = round((now - t0) * 1000, 1)
        return now

    # 1. colours ------------------------------------------------------------------------------------------
    def colours(self) -> dict:
        rgb = self.rgb
        q = (rgb >> 2).astype(np.uint32)
        packed = (q[..., 0] << 12) | (q[..., 1] << 6) | q[..., 2]
        self.packed = packed
        border = np.concatenate([packed[0], packed[-1], packed[:, 0], packed[:, -1]])
        vals, counts = np.unique(border, return_counts=True)
        bgq = vals[np.argmax(counts)]
        self.bg_key = int(bgq)
        bg_pixels = rgb[packed == bgq]
        self.bg = np.median(bg_pixels, axis=0)
        vals, counts = np.unique(packed, return_counts=True)
        order = np.argsort(-counts)
        total = packed.size
        top = []
        for i in order[:12]:
            k = vals[i]
            col = ((k >> 12) & 63, (k >> 6) & 63, k & 63)
            top.append({"hex": hexc([v * 4 + 2 for v in col]), "share": round(float(counts[i]) / total, 4)})
        self.color_keys = vals[order]
        self.color_counts = counts[order]
        # k-means palette on a subsample (dominant colours including anti-aliased mixtures)
        flat = rgb.reshape(-1, 3)
        step = max(1, flat.shape[0] // 60000)
        sample = flat[::step].astype(np.float32)
        k = 6
        _, labels, centers = cv2.kmeans(sample, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 2, cv2.KMEANS_PP_CENTERS)
        share = np.bincount(labels.ravel(), minlength=k) / labels.size
        palette = [{"hex": hexc(centers[i]), "share": round(float(share[i]), 4)} for i in np.argsort(-share)]
        luminance = float(np.dot(self.bg, [0.2126, 0.7152, 0.0722]))
        return {"background": hexc(self.bg), "theme": "dark" if luminance < 110 else "light", "top_colors": top, "kmeans_palette": palette}

    # 2. regions ------------------------------------------------------------------------------------------
    def regions(self) -> list[dict]:
        rgb, packed = self.rgb, self.packed
        self.glyph_guess = getattr(self, "glyph", None) or max(8.0, self.w / 160)
        min_side, min_area = 8, 90
        out: list[dict] = []
        lines: list[dict] = []
        # one sort of all pixels by colour; each colour is then labelled only inside its own bounding box (not a full-image mask per colour)
        flat = packed.ravel()
        order = np.argsort(flat, kind="stable")
        sorted_keys = flat[order]
        starts = np.searchsorted(sorted_keys, self.color_keys[self.color_counts >= 24])
        for key, start in zip(self.color_keys[self.color_counts >= 24], starts):
            end = np.searchsorted(sorted_keys, key, side="right")
            idx = order[start:end]
            ys, xs = np.divmod(idx, self.w)
            oy, ox = int(ys.min()), int(xs.min())
            mask = np.zeros((int(ys.max()) - oy + 1, int(xs.max()) - ox + 1), np.uint8)
            mask[ys - oy, xs - ox] = 1
            n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
            for i in range(1, n):
                x, y, w, h, area = (int(v) for v in stats[i])
                x += ox
                y += oy
                if min(w, h) <= 4 and max(w, h) >= 24 and area >= 0.9 * w * h:
                    col = rgb[y + h // 2, x + w // 2].astype(float)
                    if w >= h:
                        sides = [rgb[max(0, y - 1), x + w // 2], rgb[min(self.h - 1, y + h), x + w // 2]]
                    else:
                        sides = [rgb[y + h // 2, max(0, x - 1)], rgb[y + h // 2, min(self.w - 1, x + w)]]
                    if min(contrast(col, sd) for sd in sides) > 20:
                        lines.append({"bbox": [x, y, w, h], "fill": col, "key": int(key)})
                    continue
                if area < min_area or w < min_side or h < min_side:
                    continue
                solidity = area / float(w * h)
                ring = False
                if solidity < 0.55:
                    # a rectangular ring: its pixels hug the bounding box edges all round
                    comp = lab[y - oy:y - oy + h, x - ox:x - ox + w] == i
                    bw = max(1, min(4, min(w, h) // 8))
                    edge = np.zeros_like(comp)
                    edge[:bw, :] = edge[-bw:, :] = True
                    edge[:, :bw] = edge[:, -bw:] = True
                    on_edge = (comp & edge).sum() / max(1, edge.sum())
                    inside = comp[bw:-bw, bw:-bw].mean() if h > 2 * bw and w > 2 * bw else 1.0
                    ring = on_edge > 0.8 and inside < 0.05
                    if not ring and not self._filled_container(comp):
                        continue
                comp = lab[y - oy:y - oy + h, x - ox:x - ox + w] == i
                fill = np.median(rgb[y:y + h, x:x + w][comp], axis=0)
                is_root = key == self.bg_key and w * h > 0.25 * self.w * self.h
                out.append({"bbox": [x, y, w, h], "fill": fill, "solidity": solidity, "ring": ring, "key": int(key), "root": bool(is_root),
                            "mask_radius": None if ring or is_root else _mask_radius(comp)})
        for r in out:
            self._measure_region(r)
        out += self._unions(out, lines)
        # a ring that duplicates a fill region (the border of an element already found) is dropped
        fills = [r for r in out if not r["ring"]]
        out = [r for r in out if not r["ring"] or not any(iou(r["bbox"], f["bbox"]) > 0.85 for f in fills)]
        # lines that are one side of a region's border are part of that region, the rest are dividers
        for ln in lines:
            if any(_on_border(ln["bbox"], r) for r in out):
                continue
            out.append({"bbox": ln["bbox"], "fill": ln["fill"], "solidity": 1.0, "ring": False, "key": ln["key"], "root": False, "border": None, "radius": 0.0, "line": True})
        return out

    def _filled_container(self, comp: np.ndarray) -> bool:
        """A container whose fill is mostly covered by its content (a card holding a large image): once its holes are filled it is a
        near-solid rectangle, the holes are a real share of it, and it is several glyphs tall and wide (letters with counters are not)."""
        h, w = comp.shape
        if min(h, w) < 2.5 * self.glyph_guess or comp.sum() < 4 * (h + w):  # a thin closed outline (a border) is not a fill
            return False
        contours, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros((h, w), np.uint8)
        cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
        area = float(filled.sum())
        return area >= 0.92 * w * h and (area - comp.sum()) >= 0.2 * area

    def _unions(self, regions: list[dict], lines: list[dict]) -> list[dict]:
        """A container whose fill is cut into pieces by its own rows (a border line between list rows, a highlighted row, a strip of tabs) is not one
        colour region. Pieces of the same fill that line up edge to edge, separated only by a thin line or by a strip of other elements spanning
        their whole width (or height), are joined back into the container. The pieces stay: rows are visible boxes too."""
        g = self.glyph_guess
        cand = [r for r in regions if not r["ring"] and not r["root"] and r["bbox"][2] * r["bbox"][3] >= (3 * g) ** 2]  # container pieces, not icons
        others = [r["bbox"] for r in regions] + [ln["bbox"] for ln in lines]
        def contains(o, b):
            return o[0] <= b[0] and o[1] <= b[1] and o[0] + o[2] >= b[0] + b[2] and o[1] + o[3] >= b[1] + b[3]
        def strip_covered(sx, sy, sw, sh, a=None, b=None):
            if sw <= 0 or sh <= 0:
                return True
            if min(sw, sh) <= 4:
                return True
            cov = np.zeros((sh, sw), bool)
            for b in [o for o in others if not (a is not None and (contains(o, a) or contains(o, b)))]:  # an ancestor's box covers everything: not evidence
                ix0, iy0 = max(sx, b[0]), max(sy, b[1])
                ix1, iy1 = min(sx + sw, b[0] + b[2]), min(sy + sh, b[1] + b[3])
                if ix1 > ix0 and iy1 > iy0:
                    cov[iy0 - sy:iy1 - sy, ix0 - sx:ix1 - sx] = True
            return cov.mean() >= 0.9
        parent = list(range(len(cand)))
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for i, a in enumerate(cand):
            ax, ay, aw, ah = a["bbox"]
            for j in range(i + 1, len(cand)):
                b = cand[j]
                if b["key"] != a["key"]:
                    continue
                bx, by, bw, bh = b["bbox"]
                if abs(ax - bx) <= 3 and abs(ax + aw - bx - bw) <= 3:  # stacked
                    top, bot = (a, b) if ay < by else (b, a)
                    g0 = top["bbox"][1] + top["bbox"][3]
                    gap = bot["bbox"][1] - g0
                    if -2 <= gap <= 8 * self.glyph_guess and strip_covered(max(ax, bx), g0, min(ax + aw, bx + bw) - max(ax, bx), gap, a["bbox"], b["bbox"]):
                        parent[find(i)] = find(j)
                elif abs(ay - by) <= 3 and abs(ay + ah - by - bh) <= 3:  # side by side
                    lft, rgt = (a, b) if ax < bx else (b, a)
                    g0 = lft["bbox"][0] + lft["bbox"][2]
                    gap = rgt["bbox"][0] - g0
                    if -2 <= gap <= 8 * self.glyph_guess and strip_covered(g0, max(ay, by), gap, min(ay + ah, by + bh) - max(ay, by), a["bbox"], b["bbox"]):
                        parent[find(i)] = find(j)
        groups: dict = {}
        for i in range(len(cand)):
            groups.setdefault(find(i), []).append(cand[i])
        out = []
        for members in groups.values():
            if len(members) < 2:
                continue
            x0 = min(m["bbox"][0] for m in members)
            y0 = min(m["bbox"][1] for m in members)
            x1 = max(m["bbox"][0] + m["bbox"][2] for m in members)
            y1 = max(m["bbox"][1] + m["bbox"][3] for m in members)
            # undo the pieces' own outer borders before measuring the joined box's border
            u = {"bbox": [x0, y0, x1 - x0, y1 - y0], "fill": members[0]["fill"], "solidity": 1.0, "ring": False, "key": members[0]["key"], "root": False, "union": len(members)}
            bw = [max((m["border"]["sides"][k] if m.get("border") else 0) for m in members) for k in range(4)]
            u["bbox"] = [x0 + bw[3], y0 + bw[0], x1 - x0 - bw[1] - bw[3], y1 - y0 - bw[0] - bw[2]]
            self._measure_region(u)
            out.append(u)
        return out

    def _measure_region(self, r: dict) -> None:
        """Each side separately: the band just outside the fill, if it is a uniform colour unlike both the fill and what lies beyond it, is a border on
        that side (a card has four, a header often only a bottom one). The box is grown by the border on each side that has one."""
        rgb = self.rgb
        x, y, w, h = r["bbox"]
        r["border"] = None
        if r["ring"]:
            b = max(1, min(4, min(w, h) // 8))
            r["border"] = {"width": b, "sides": [b, b, b, b], "color": hexc(r["fill"])}
            r["radius"] = self._radius(r)
            return
        sides, colours = [0, 0, 0, 0], [None] * 4
        for si in range(4):
            for bw in (1, 2, 3, 4, 5, 6):
                if si == 0:   # top
                    if y - bw - 2 < 0:
                        break
                    band, beyond = rgb[y - bw, x + w // 4: x + 3 * w // 4], rgb[y - bw - 2, x + w // 2]
                elif si == 1:  # right
                    if x + w + bw + 2 > self.w:
                        break
                    band, beyond = rgb[y + h // 4: y + 3 * h // 4, x + w - 1 + bw], rgb[y + h // 2, x + w - 1 + bw + 2]
                elif si == 2:  # bottom
                    if y + h + bw + 2 > self.h:
                        break
                    band, beyond = rgb[y + h - 1 + bw, x + w // 4: x + 3 * w // 4], rgb[y + h - 1 + bw + 2, x + w // 2]
                else:          # left
                    if x - bw - 2 < 0:
                        break
                    band, beyond = rgb[y + h // 4: y + 3 * h // 4, x - bw], rgb[y + h // 2, x - bw - 2]
                band = band.astype(float)
                if band.size == 0:
                    break
                med = np.median(band, axis=0)
                uniform = np.mean(np.linalg.norm(band - med, axis=1) < 12) > 0.85
                if uniform and contrast(med, r["fill"]) > 8 and contrast(med, beyond) > 6 and (colours[si] is None or contrast(med, colours[si]) < 12):
                    sides[si], colours[si] = bw, med
                    continue
                break
        if any(sides):
            col = next(c for c in colours if c is not None)
            r["border"] = {"width": max(sides), "sides": sides, "color": hexc(col)}
            r["bbox"] = [x - sides[3], y - sides[0], w + sides[1] + sides[3], h + sides[0] + sides[2]]
        if not r["ring"] and not r.get("line") and not r.get("root"):
            # the blended edge pixel just outside a shape that is not pixel-aligned (a circle, a rounded side) belongs to it
            x, y, w, h = r["bbox"]
            grow = [0, 0, 0, 0]
            probes = [((y - 1, x + w // 2), (y - 3, x + w // 2)), ((y + h // 2, x + w), (y + h // 2, x + w + 2)), ((y + h, x + w // 2), (y + h + 2, x + w // 2)), ((y + h // 2, x - 1), (y + h // 2, x - 3))]
            inner = np.asarray(r["fill"] if not r.get("border") else [int(r["border"]["color"][i:i + 2], 16) for i in (1, 3, 5)], float)
            for si, ((py, px), (oy, ox)) in enumerate(probes):
                if not (0 <= py < self.h and 0 <= px < self.w and 0 <= oy < self.h and 0 <= ox < self.w):
                    continue
                if r.get("border") and r["border"]["sides"][si]:
                    continue
                p, o = self.rgb[py, px].astype(float), self.rgb[oy, ox].astype(float)
                span = contrast(inner, o)
                if span < 20:
                    continue
                t = float(np.dot(p - o, inner - o) / (span * span))
                if 0.12 < t <= 1.05 and contrast(p, o + min(t, 1.0) * (inner - o)) < 14:
                    grow[si] = 1
            r["bbox"] = [x - grow[3], y - grow[0], w + grow[1] + grow[3], h + grow[0] + grow[2]]
        if r.get("mask_radius") is not None:
            # from the fill's own pixels (exact), plus the border that surrounds it
            r["radius"] = round(r["mask_radius"] + (max(r["border"]["sides"]) if r.get("border") else 0), 1) if r["mask_radius"] > 0.5 else 0.0
        else:
            r["radius"] = self._radius(r)

    def _radius(self, r: dict) -> float | None:
        """Distance from each bbox corner, along the diagonal, to the first pixel of the element's own colour (fill, or border if it has one);
        a circular corner of radius R puts it at R*(1-1/sqrt2) along each axis."""
        x, y, w, h = r["bbox"]
        bc = r.get("border")
        target = np.array([int(bc["color"][i:i + 2], 16) for i in (1, 3, 5)], float) if bc else np.asarray(r["fill"], float)
        outside = None
        ds = []
        limit = min(w, h) // 2
        for cx, cy, sx, sy in ((x, y, 1, 1), (x + w - 1, y, -1, 1), (x, y + h - 1, 1, -1), (x + w - 1, y + h - 1, -1, -1)):
            if not (0 <= cx < self.w and 0 <= cy < self.h):
                continue
            outside = self.rgb[cy, cx].astype(float)
            span = contrast(target, outside)
            if span < 6:
                continue
            for d in range(0, limit):
                px = self.rgb[cy + sy * d, cx + sx * d].astype(float)
                # sub-pixel: how far this pixel is from 'outside' toward 'target'
                t = np.clip(np.dot(px - outside, target - outside) / (span * span), 0, 1)
                if t > 0.5:
                    ds.append(d + (0.5 - (t - 0.5)) if d else 0.0)
                    break
        if not ds:
            return None
        d = float(np.median(ds))
        return round(min(d / (1 - 1 / math.sqrt(2)), min(w, h) / 2), 1) if d > 0.3 else 0.0

    # 3. ink (text and icons) -----------------------------------------------------------------------------
    def ink(self, regions: list[dict]) -> list[dict]:
        rgb = self.rgb.astype(np.int16)
        fillmap = np.empty_like(rgb)
        fillmap[:] = self.bg
        region_px = np.zeros((self.h, self.w), bool)
        for r in sorted(regions, key=lambda r: -(r["bbox"][2] * r["bbox"][3])):
            x, y, w, h = r["bbox"]
            if r["ring"]:
                continue
            rad = int(round(min(r.get("radius") or 0, min(w, h) / 2)))
            if rad >= 2:
                # paint the fill with its rounded corners, so the parent's colour shows in them (not a false "ink" speck)
                m = np.ones((h, w), bool)
                yy, xx = np.mgrid[0:rad, 0:rad]
                corner = (xx - rad + 0.5) ** 2 + (yy - rad + 0.5) ** 2 > rad * rad
                m[:rad, :rad] &= ~corner
                m[:rad, -rad:] &= ~corner[:, ::-1]
                m[-rad:, :rad] &= ~corner[::-1, :]
                m[-rad:, -rad:] &= ~corner[::-1, ::-1]
                fillmap[y:y + h, x:x + w][m] = r["fill"]
            else:
                fillmap[y:y + h, x:x + w] = r["fill"]
        for r in regions:
            x, y, w, h = r["bbox"]
            region_px[y:y + h, x:x + w] |= self.packed[y:y + h, x:x + w] == r["key"]
            # the anti-aliased arc of a rounded corner is the element's edge, not ink (content never sits in the corner square)
            if not r.get("line") and not r["root"]:
                # the anti-aliased edge just inside the box is the element's own outline, not ink (content keeps a padding from it)
                e = 2  # 2 px inside the box and 1 px outside it (the blended edge of a fill lies just outside its exact pixels)
                y0, y1, x0, x1 = max(0, y - 1), min(self.h, y + h + 1), max(0, x - 1), min(self.w, x + w + 1)
                region_px[y0:y + e, x0:x1] = True
                region_px[y + h - e:y1, x0:x1] = True
                region_px[y0:y1, x0:x + e] = True
                region_px[y0:y1, x + w - e:x1] = True
            rad = int(math.ceil(min(r.get("radius") or 0, min(w, h) / 2)))
            if rad >= 3 and not r.get("line"):
                band = 2 + (r["border"]["width"] if r.get("border") else 0)
                yy, xx = np.mgrid[0:rad, 0:rad]
                arc = np.hypot(xx - rad + 0.5, yy - rad + 0.5) >= rad - band  # the arc band and what lies outside it, not the corner's inside
                for cy, cx, a in ((y, x, arc), (y, x + w - rad, arc[:, ::-1]), (y + h - rad, x, arc[::-1, :]), (y + h - rad, x + w - rad, arc[::-1, ::-1])):
                    if cy >= 0 and cx >= 0 and cy + rad <= self.h and cx + rad <= self.w:
                        region_px[cy:cy + rad, cx:cx + rad] |= a
            if r.get("border"):
                b = r["border"]["width"]
                region_px[y:y + b, x:x + w] = True
                region_px[y + h - b:y + h, x:x + w] = True
                region_px[y:y + h, x:x + b] = True
                region_px[y:y + h, x + w - b:x + w] = True
        diff = np.abs(rgb - fillmap).sum(axis=2)
        mask = ((diff > 60) & ~region_px).astype(np.uint8)
        self.ink_mask = mask
        # the region each pixel sits in (smallest box painted last), so ink is only grouped within one region
        idmap = np.full((self.h, self.w), -1, np.int32)
        for idx, r in sorted(enumerate(regions), key=lambda t: -(t[1]["bbox"][2] * t[1]["bbox"][3])):
            if r.get("line") or r.get("small"):
                continue
            x, y, w, h = r["bbox"]
            idmap[y:y + h, x:x + w] = idx
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        blobs = [[int(v) for v in stats[i][:4]] for i in range(1, n) if stats[i][4] >= 2]
        hs = np.array([b[3] for b in blobs if 3 <= b[3] <= 120])
        glyph = float(np.median(hs)) if hs.size else 10.0
        self.glyph = glyph
        # 1. parts that touch (letters such as i, icons made of several shapes) form one blob
        blobs = _merge(blobs, lambda a, b: _gap_x(a, b) <= max(1, 0.12 * max(a[3], b[3])) and _gap_y(a, b) <= max(1, 0.12 * max(a[3], b[3])))
        # 2. icons: squarish, clearly taller than the text glyphs, with no neighbour of similar height close by on the same line
        def same_line(a, b):
            ov = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
            return ov >= 0.5 * min(a[3], b[3])
        icons, rest = [], []
        for b in blobs:
            aspect = b[2] / max(1, b[3])
            tall = b[3] >= 1.45 * glyph
            # a neighbour of similar height close by, or any glyph-sized neighbour almost touching it (letters of large text), makes it text
            near = any(o is not b and same_line(b, o) and ((o[3] >= 0.8 * b[3] and _gap_x(b, o) <= 0.5 * b[3]) or (o[3] >= 0.5 * b[3] and _gap_x(b, o) <= 0.3 * b[3])) for o in blobs)
            if 0.6 <= aspect <= 1.6 and tall and not near:
                icons.append(b)
            else:
                rest.append(b)
        # 3. text lines: same region, same line, gap up to ~3/4 of the taller glyph
        def rid(b):
            return int(idmap[min(self.h - 1, b[1] + b[3] // 2), min(self.w - 1, b[0] + b[2] // 2)])
        lines = _merge(rest, lambda a, b: rid(a) == rid(b) and same_line(a, b) and _gap_x(a, b) <= 1.0 * max(a[3], b[3]) and _gap_y(a, b) <= 0)
        out = []
        for kind, group in (("icon", icons), ("text", lines)):
            for x0, y0, bw, bh in group:
                if bh < max(3, 0.35 * glyph) and bw < glyph:
                    continue  # specks
                sub = mask[y0:y0 + bh, x0:x0 + bw].astype(bool)
                if sub.sum() < 6:
                    continue
                pix = self.rgb[y0:y0 + bh, x0:x0 + bw][sub].astype(float)
                under = fillmap[y0:y0 + bh, x0:x0 + bw][sub].astype(float)
                dist = np.linalg.norm(pix - under, axis=1)
                colour = np.median(pix[dist >= np.quantile(dist, 0.75)], axis=0)
                k = kind
                if bh <= 3 and bw >= 8 * bh:
                    k = "divider"
                out.append({"bbox": [int(x0), int(y0), int(bw), int(bh)], "kind": k, "color": colour, "contrast": contrast(colour, np.median(under, axis=0)), "density": float(sub.mean())})
        return out

    # 4. classification, hierarchy, layout ----------------------------------------------------------------
    def classify(self, regions: list[dict], inks: list[dict]) -> list[dict]:
        g = self.glyph
        els = []
        for r in regions:
            if r["root"]:
                continue
            x, y, w, h = r["bbox"]
            aspect = w / max(1, h)
            radius = r["radius"] or 0
            if r.get("line") or min(w, h) <= 3:
                kind = "divider"
            elif 0.85 <= aspect <= 1.18 and 0.70 <= r["solidity"] <= 0.84 and not r["ring"]:
                kind = "circle"
            elif radius >= 0.4 * h and aspect > 1.2 and h < 4.5 * g:
                kind = "pill"
            elif h <= 6.5 * g and w <= 40 * g and aspect >= 1.2:
                kind = "control"  # button or input
            elif max(w, h) <= 3.2 * g and r["solidity"] > 0.4:
                kind = "icon"
            else:
                kind = "box"
            outside = self.rgb[max(0, y - 2), min(self.w - 1, x + w // 2)]
            cont = contrast(r["fill"], outside) if not r.get("border") else max(contrast(r["fill"], outside), 30)
            conf = min(1.0, 0.35 + cont / 120) * (0.9 if r["ring"] else 1.0)
            els.append({"type": kind, "bbox": r["bbox"], "fill": hexc(r["fill"]), "border": r["border"], "radius": r["radius"], "confidence": round(conf, 2), "source": "region"})
        small = [e["bbox"] for e in els if e["type"] in ("icon", "circle")]
        for k in inks:
            # the anti-aliased edge of a solid icon or circle already found as a region is not a second element
            if any(iou(k["bbox"], b) > 0.4 or (b[0] - 2 <= k["bbox"][0] and b[1] - 2 <= k["bbox"][1] and k["bbox"][0] + k["bbox"][2] <= b[0] + b[2] + 2 and k["bbox"][1] + k["bbox"][3] <= b[1] + b[3] + 2) for b in small):
                continue
            conf = min(1.0, 0.3 + k["contrast"] / 200) * (0.85 if k["kind"] == "icon" else 1.0)
            els.append({"type": k["kind"], "bbox": k["bbox"], "fill": None, "text_color": hexc(k["color"]) if k["kind"] == "text" else None, "color": hexc(k["color"]), "border": None, "radius": None, "confidence": round(conf, 2), "source": "ink"})
        # hierarchy: smallest region box containing the element
        def contains(a, b, tol=2):
            return a[0] - tol <= b[0] and a[1] - tol <= b[1] and a[0] + a[2] + tol >= b[0] + b[2] and a[1] + a[3] + tol >= b[1] + b[3]
        order = sorted(range(len(els)), key=lambda i: els[i]["bbox"][2] * els[i]["bbox"][3])
        for i, e in enumerate(els):
            e["id"] = f"e{i}"
        for i in range(len(els)):
            e = els[i]
            parent = None
            for j in order:
                p = els[j]
                if j == i or p["source"] != "region" or p["type"] in ("divider", "circle", "pill", "icon"):
                    continue
                if p["bbox"][2] * p["bbox"][3] <= e["bbox"][2] * e["bbox"][3]:
                    continue
                if contains(p["bbox"], e["bbox"]):
                    parent = p["id"]
                    break
            e["parent"] = parent
        # a control that holds exactly one text line, centred -> button; left-aligned -> input
        by_parent: dict = {}
        for e in els:
            by_parent.setdefault(e["parent"], []).append(e)
        for e in els:
            if e["source"] != "region" or e["type"] in ("pill", "circle", "divider") or e["bbox"][3] > 6.5 * g:
                continue
            kids = [k for k in by_parent.get(e["id"], []) if k["type"] in ("text", "icon")]
            if len(kids) == 1:
                kx = kids[0]["bbox"][0] + kids[0]["bbox"][2] / 2
                cx = e["bbox"][0] + e["bbox"][2] / 2
                e["type"] = "button" if abs(kx - cx) < 0.12 * e["bbox"][2] else ("input" if kids[0]["type"] == "text" else e["type"])
            elif not kids and e["type"] == "control":
                e["type"] = "box"
        return els

    def layout(self, els: list[dict]) -> dict:
        by_parent: dict = {}
        for e in els:
            by_parent.setdefault(e["parent"], []).append(e)
        index = {e["id"]: e for e in els}
        padding, gaps, aligned = [], [], []
        for pid, kids in by_parent.items():
            if pid is None or not kids:
                continue
            p = index[pid]["bbox"]
            left = min(k["bbox"][0] for k in kids) - p[0]
            t = min(k["bbox"][1] for k in kids) - p[1]
            r = p[0] + p[2] - max(k["bbox"][0] + k["bbox"][2] for k in kids)
            b = p[1] + p[3] - max(k["bbox"][1] + k["bbox"][3] for k in kids)
            padding.append({"parent": pid, "left": left, "top": t, "right": r, "bottom": b})
        for pid, kids in by_parent.items():
            kids = [k for k in kids if k["type"] != "divider"]
            for a in kids:
                ax, ay, aw, ah = a["bbox"]
                # nearest sibling to the right that overlaps vertically, and below that overlaps horizontally
                right = [k for k in kids if k is not a and k["bbox"][0] >= ax + aw and min(ay + ah, k["bbox"][1] + k["bbox"][3]) - max(ay, k["bbox"][1]) > 0.3 * min(ah, k["bbox"][3])]
                if right:
                    k = min(right, key=lambda k: k["bbox"][0])
                    gaps.append({"a": a["id"], "b": k["id"], "axis": "x", "gap": k["bbox"][0] - (ax + aw)})
                below = [k for k in kids if k is not a and k["bbox"][1] >= ay + ah and min(ax + aw, k["bbox"][0] + k["bbox"][2]) - max(ax, k["bbox"][0]) > 0.3 * min(aw, k["bbox"][2])]
                if below:
                    k = min(below, key=lambda k: k["bbox"][1])
                    gaps.append({"a": a["id"], "b": k["id"], "axis": "y", "gap": k["bbox"][1] - (ay + ah)})
            for edge, f in (("left", lambda b: b[0]), ("right", lambda b: b[0] + b[2]), ("center_x", lambda b: b[0] + b[2] / 2), ("top", lambda b: b[1]), ("center_y", lambda b: b[1] + b[3] / 2)):
                groups: dict = {}
                for k in kids:
                    v = f(k["bbox"])
                    for key in list(groups):
                        if abs(key - v) <= 1:
                            groups[key].append(k["id"])
                            break
                    else:
                        groups[v] = [k["id"]]
                for v, ids in groups.items():
                    if len(ids) >= 2:
                        aligned.append({"edge": edge, "at": round(v, 1), "ids": ids})
        return {"padding": padding, "gaps": gaps, "alignment": aligned}


def analyze_rgb(rgb: np.ndarray) -> dict:
    """The full analysis of one RGB image (H x W x 3, uint8): colours, elements, layout. Pixel units of this image."""
    t_all = time.perf_counter()
    a = Analyzer(rgb)
    t = time.perf_counter()
    colours = a.colours()
    t = a._t("colours", t)
    regions = a.regions()
    t = a._t("regions", t)
    # pass 1: regions smaller than a few glyphs must not split text lines
    for r in regions:
        r["small"] = r["bbox"][2] * r["bbox"][3] < 9 * 64  # refined below once the glyph height is known
    inks = a.ink(regions)
    g = a.glyph
    for r in regions:
        r["small"] = r["bbox"][2] * r["bbox"][3] < (3 * g) ** 2
    inks = a.ink(regions)
    # pass 2: flat-coloured pieces of large glyphs (letter bodies, counters, stems) found as regions are part of a text line, not elements
    lines = [k["bbox"] for k in inks if k["kind"] == "text"]
    def in_text(b):
        return any(b[0] >= ln[0] - 2 and b[1] >= ln[1] - 2 and b[0] + b[2] <= ln[0] + ln[2] + 2 and b[1] + b[3] <= ln[1] + ln[3] + 2 and b[2] * b[3] < 0.5 * ln[2] * ln[3] for ln in lines)
    # a solid region the colour of a glyph next to it on the same line, with no text inside, is a piece of large text (a letter body), not an element
    def glyph_piece(r):
        x, y, w, h = r["bbox"]
        if r["root"] or r["ring"] or r.get("line") or max(w, h) > 8 * g:
            return False
        if any(x <= k["bbox"][0] + k["bbox"][2] / 2 <= x + w and y <= k["bbox"][1] + k["bbox"][3] / 2 <= y + h for k in inks):
            return False
        def near(b, col):
            if x <= b[0] + b[2] / 2 <= x + w and y <= b[1] + b[3] / 2 <= y + h:
                return False  # inside the region itself (its own edge pixels), not a neighbour
            ov = min(y + h, b[1] + b[3]) - max(y, b[1])
            gap = max(x, b[0]) - min(x + w, b[0] + b[2])
            return ov >= 0.5 * min(h, b[3]) and gap <= 0.6 * h and contrast(col, r["fill"]) < 25
        return any(near(k["bbox"], k["color"]) for k in inks) or any(o is not r and not o["root"] and near(o["bbox"], o["fill"]) for o in regions)
    icons = [k["bbox"] for k in inks if k["kind"] == "icon"]
    def in_icon(b):
        return any(b[0] >= i[0] - 2 and b[1] >= i[1] - 2 and b[0] + b[2] <= i[0] + i[2] + 2 and b[1] + b[3] <= i[1] + i[3] + 2 and b[2] * b[3] < 0.8 * i[2] * i[3] for i in icons)
    kept = [r for r in regions if r["root"] or not (in_text(r["bbox"]) or glyph_piece(r) or in_icon(r["bbox"]))]
    if len(kept) != len(regions):
        regions = kept
        inks = a.ink(regions)
    t = a._t("ink", t)
    els = a.classify(regions, inks)
    t = a._t("classify", t)
    layout = a.layout(els)
    t = a._t("layout", t)
    result = {
        "image": {"width": a.w, "height": a.h, "glyph_height_px": round(a.glyph, 1)},
        "colors": colours,
        "elements": [{k: v for k, v in e.items()} for e in els],
        "layout": layout,
        "counts": {k: sum(1 for e in els if e["type"] == k) for k in sorted({e["type"] for e in els})},
        "timings_ms": {**a.timings, "total": round((time.perf_counter() - t_all) * 1000, 1)},
        "method": "pillow+opencv heuristics (colour regions, ink grouping); measured values are pixel measurements of this image; types and confidence are estimates",
    }
    return result
