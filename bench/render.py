"""Renders HTML to PNG with headless Chrome (Playwright) and reads what is on the page from the DOM.

Two readings of a page, both in device pixels:

* ``ground_truth`` (scenes): every element tagged ``data-gt``, plus every untagged element that is visibly a box (own background unlike
  what is behind it, or a border). Text and icons also get an ``ink`` box: the tight box of the pixels they paint, found by rendering the
  page a second time with them hidden and diffing.
* ``page_items`` (scenes and model replicas alike, for scoring replicas): visible boxes and text runs, with no tags needed.

    python -m bench.render                    renders every scene at 1x and 2x, light and dark, into data/screenshots/
    python -m bench.render --dpr 3 --themes light --scenes web-complex --out data/screenshots-highres

Uses the installed Google Chrome (``channel="chrome"``); set LAYOUTLENS_BROWSER=chromium to use Playwright's own Chromium instead.
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SCENES = ROOT / "data" / "scenes"
SIZES = {"web": (1440, 900), "desktop": (1280, 800), "mobile": (390, 844)}

_HELPERS = """
  const parse = (c) => { const m = /rgba?\\(([^)]+)\\)/.exec(c || ""); if (!m) return null; const p = m[1].split(",").map(parseFloat);
    return { r: p[0], g: p[1], b: p[2], a: p.length > 3 ? p[3] : 1 }; };
  const hex = (c) => c ? "#" + [c.r, c.g, c.b].map((v) => Math.round(v).toString(16).padStart(2, "0")).join("") : null;
  const visibleBg = (el) => { for (let e = el; e; e = e.parentElement) { const c = parse(getComputedStyle(e).backgroundColor);
    if (c && c.a > 0.5) return hex(c); } return "#ffffff"; };
  const clipOf = (el) => { let r = { l: 0, t: 0, r: innerWidth, b: innerHeight };
    for (let e = el.parentElement; e; e = e.parentElement) { if (getComputedStyle(e).overflow !== "visible") { const b = e.getBoundingClientRect();
      r = { l: Math.max(r.l, b.left), t: Math.max(r.t, b.top), r: Math.min(r.r, b.right), b: Math.min(r.b, b.bottom) }; } } return r; };
  const visibleBox = (el, b) => { const c = clipOf(el); const x0 = Math.max(b.left, c.l), y0 = Math.max(b.top, c.t);
    const x1 = Math.min(b.right, c.r), y1 = Math.min(b.bottom, c.b); return x1 - x0 >= 1 && y1 - y0 >= 1 ? [x0, y0, x1 - x0, y1 - y0] : null; };
  const borders = (cs) => ["Top", "Right", "Bottom", "Left"].map((s) => parseFloat(cs["border" + s + "Width"]) || 0);
  const isBox = (el, cs) => { if (borders(cs).some((w) => w > 0)) return true; const own = parse(cs.backgroundColor);
    return !!(own && own.a > 0.5 && hex(own) !== visibleBg(el.parentElement)); };
"""

_GROUND_TRUTH = """() => {""" + _HELPERS + """
  const out = []; let auto = 0;
  for (const el of document.body.querySelectorAll("*")) {
    const cs = getComputedStyle(el);
    if (!el.dataset.gt) {
      if (el.closest("[data-gt]:not([data-gt=panel])") || el.closest("svg") || !isBox(el, cs)) continue;
      el.dataset.gt = "panel"; el.dataset.name = "auto-" + (++auto); el.dataset.auto = "1";
    }
    const box = visibleBox(el, el.getBoundingClientRect()); if (!box) continue;
    const own = parse(cs.backgroundColor);
    let parent = null; for (let e = el.parentElement; e; e = e.parentElement) if (e.dataset && e.dataset.gt) { parent = e.dataset.name; break; }
    const b = borders(cs); const side = ["Top", "Right", "Bottom", "Left"].find((s) => parseFloat(cs["border" + s + "Width"]));
    out.push({ name: el.dataset.name, kind: el.dataset.gt, auto: !!el.dataset.auto, parent, box,
      fill: own && own.a > 0.5 ? hex(own) : visibleBg(el.parentElement || el), own_fill: !!(own && own.a > 0.5),
      behind: visibleBg(el.parentElement || document.body), radius: parseFloat(cs.borderTopLeftRadius) || 0,
      borders: b, border_width: Math.max(...b), border_color: side ? hex(parse(cs["border" + side + "Color"])) : null,
      text_color: hex(parse(cs.color)), text: el.dataset.gt === "text" ? el.textContent.replace(/\\s+/g, " ").trim() : null,
      font_px: parseFloat(cs.fontSize), font_weight: parseInt(cs.fontWeight, 10), text_align: cs.textAlign });
  }
  return { elements: out, page_background: hex(parse(getComputedStyle(document.body).backgroundColor)) };
}"""

_PAGE_ITEMS = """() => {""" + _HELPERS + """
  const boxes = [], texts = [];
  for (const el of document.body.querySelectorAll("*")) {
    if (el.closest("svg")) continue;
    const cs = getComputedStyle(el);
    if (cs.visibility === "hidden" || cs.display === "none" || parseFloat(cs.opacity) < 0.1) continue;
    if (isBox(el, cs)) { const box = visibleBox(el, el.getBoundingClientRect());
      if (box) boxes.push({ box, fill: hex(parse(cs.backgroundColor)) || visibleBg(el), border_width: Math.max(...borders(cs)), radius: parseFloat(cs.borderTopLeftRadius) || 0 }); }
    const own = [...el.childNodes].filter((n) => n.nodeType === 3 && n.textContent.trim());
    if (own.length) { const r = document.createRange(); r.setStartBefore(own[0]); r.setEndAfter(own[own.length - 1]);
      const box = visibleBox(el, r.getBoundingClientRect());
      if (box) texts.push({ text: own.map((n) => n.textContent).join(" ").replace(/\\s+/g, " ").trim(), box, color: hex(parse(cs.color)), font_px: parseFloat(cs.fontSize) }); }
  }
  return { boxes, texts, page_background: hex(parse(getComputedStyle(document.body).backgroundColor)) };
}"""

_HIDE_INK = '[data-gt="text"], [data-gt="text"] * { color: transparent !important; } svg[data-gt="icon"] { visibility: hidden !important; }'


def _scale(items: list[dict], dpr: float, keys=("box",)) -> None:
    for it in items:
        for k in keys:
            if it.get(k) is not None:
                it[k] = [int(round(v * dpr)) for v in it[k]]


class Renderer:
    """One headless browser, reused for many pages."""

    def __init__(self) -> None:
        self._pw = sync_playwright().start()
        channel = None if os.environ.get("LAYOUTLENS_BROWSER") == "chromium" else "chrome"
        self._browser = self._pw.chromium.launch(channel=channel, headless=True)

    def close(self) -> None:
        self._browser.close()
        self._pw.stop()

    def _page(self, width: int, height: int, dpr: float):
        context = self._browser.new_context(viewport={"width": width, "height": height}, device_scale_factor=dpr)
        return context, context.new_page()

    def ground_truth(self, html_file: Path, width: int, height: int, dpr: float, theme: str) -> tuple[bytes, dict]:
        context, page = self._page(width, height, dpr)
        try:
            page.goto(html_file.resolve().as_uri())
            page.evaluate(f'document.documentElement.classList.toggle("dark", {str(theme == "dark").lower()})')
            page.evaluate("document.fonts.ready.then(() => true)")
            png = page.screenshot(type="png")
            gt = page.evaluate(_GROUND_TRUTH)
            items = page.evaluate(_PAGE_ITEMS)  # before the ink is hidden below: it reads text colours
            page.add_style_tag(content=_HIDE_INK)
            hidden = page.screenshot(type="png")
        finally:
            context.close()
        a = np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(int)
        b = np.asarray(Image.open(io.BytesIO(hidden)).convert("RGB")).astype(int)
        changed = np.abs(a - b).sum(axis=2) > 24
        h, w = changed.shape
        for e in gt["elements"]:
            e["box"] = [int(round(v * dpr)) for v in e["box"]]
            e["radius"] = round(e["radius"] * dpr, 1)
            e["borders"] = [round(v * dpr, 1) for v in e["borders"]]
            e["border_width"] = round(e["border_width"] * dpr, 1)
            e["font_px"] = round(e["font_px"] * dpr, 1)
            if e["kind"] in ("text", "icon"):
                x, y, bw, bh = e["box"]
                sub = changed[max(0, y - 2):min(h, y + bh + 2), max(0, x - 2):min(w, x + bw + 2)]
                ys, xs = np.nonzero(sub)
                e["ink"] = [int(xs.min() + max(0, x - 2)), int(ys.min() + max(0, y - 2)), int(np.ptp(xs) + 1), int(np.ptp(ys) + 1)] if xs.size else None
        _scale(items["boxes"], dpr)
        _scale(items["texts"], dpr)
        for t in items["texts"]:
            t["font_px"] = round(t["font_px"] * dpr, 1)
        gt["page_items"] = items
        return png, gt

    def page_items(self, html: str, width: int, height: int, dpr: float = 1.0) -> tuple[bytes, dict]:
        """Render an HTML string (a model's replica) and read its boxes and text runs."""
        context, page = self._page(width, height, dpr)
        try:
            page.set_content(html, wait_until="load", timeout=30000)
            page.evaluate("document.fonts.ready.then(() => true)")
            png = page.screenshot(type="png")
            items = page.evaluate(_PAGE_ITEMS)
        finally:
            context.close()
        _scale(items["boxes"], dpr)
        _scale(items["texts"], dpr)
        return png, items


def render_dataset(out: Path, dprs: list[float], themes: list[str], scenes: list[str] | None = None) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    names = scenes or sorted(p.stem for p in SCENES.glob("*.html"))
    r = Renderer()
    written = []
    try:
        for name in names:
            width, height = SIZES[name.split("-")[0]]
            for dpr in dprs:
                for theme in themes:
                    png, gt = r.ground_truth(SCENES / f"{name}.html", width, height, dpr, theme)
                    stem = f"{name}-{theme}@{dpr:g}x"
                    (out / f"{stem}.png").write_bytes(png)
                    meta = {"image": f"{stem}.png", "scene": name, "theme": theme, "dpr": dpr, "css_width": width, "css_height": height,
                            "width": int(width * dpr), "height": int(height * dpr), **gt}
                    (out / f"{stem}.json").write_text(json.dumps(meta, indent=1))
                    written.append(stem)
    finally:
        r.close()
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT / "data" / "screenshots"))
    ap.add_argument("--dpr", default="1,2")
    ap.add_argument("--themes", default="light,dark")
    ap.add_argument("--scenes", default="")
    args = ap.parse_args()
    stems = render_dataset(Path(args.out), [float(d) for d in args.dpr.split(",")], args.themes.split(","), args.scenes.split(",") if args.scenes else None)
    print(f"{len(stems)} screenshots in {args.out}")


if __name__ == "__main__":
    main()
