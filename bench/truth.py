"""Perfect metadata for a scene, in the same format as ``layoutlens.analyze(mode="combined")``, built from the DOM ground truth.

Used as the upper-bound condition in the model experiments: what a model could do if the extraction were flawless.
"""

from __future__ import annotations

from bench.evaluate import truth_box, visible
from layoutlens.analyze import _alignment

TYPE = {"panel": "box", "button": "button", "input": "input", "icon": "icon", "avatar": "circle", "chip": "pill", "toggle": "pill", "divider": "divider", "image": "box"}


def truth_metadata(gt: dict) -> dict:
    shown = [e for e in gt["elements"] if visible(e)]
    names = {e["name"] for e in shown}
    by_name = {e["name"]: e for e in gt["elements"]}

    def nearest_visible_parent(e):
        p = e["parent"]
        while p is not None and p not in names:
            p = by_name[p]["parent"] if p in by_name else None
        return p

    ids = {e["name"]: f"e{i}" for i, e in enumerate(x for x in shown if x["kind"] != "text")}
    elements, texts = [], []
    for e in shown:
        x, y, w, h = truth_box(e)
        parent = nearest_visible_parent(e)
        if e["kind"] == "text":
            item = {"id": f"t{len(texts)}", "text": e["text"], "x": x, "y": y, "w": w, "h": h, "conf": 100.0, "font_px": e["font_px"], "color": e["text_color"]}
            if parent in ids:
                item["in"] = ids[parent]
                item["align"] = _alignment([x, y, w, h], truth_box(by_name[parent]))
            texts.append(item)
            continue
        d = {"id": ids[e["name"]], "type": TYPE[e["kind"]], "x": x, "y": y, "w": w, "h": h}
        if e["kind"] != "icon":
            d["fill"] = e["fill"]
        else:
            d["color"] = e["text_color"]
        if e["border_width"]:
            d["border"] = {"sides_px": [int(round(v)) for v in e["borders"]], "color": e["border_color"]}
        if e["radius"]:
            d["radius"] = round(min(e["radius"], min(e["box"][2], e["box"][3]) / 2), 1)
        if parent in ids:
            d["parent"] = ids[parent]
        elements.append(d)
    return {"image": {"width": gt["width"], "height": gt["height"]}, "mode": "truth", "background": gt["page_background"], "elements": elements, "texts": texts}
