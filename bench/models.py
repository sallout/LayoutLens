"""Model experiments: replicate screenshots as HTML under each metadata condition.

    python -m bench.models check              does each configured model read an image? (a random number drawn in a PNG)
    python -m bench.models run [--reps 2]     runs every task x condition x model, saving each response under results/model_runs/
    python -m bench.score                     renders and scores the replicas (see bench/score.py)

Configuration (environment variables, or a git-ignored `.env` file in the repository root; variables already set in the environment win):

    LAYOUTLENS_BASE_URL        e.g. https://api.openai.com/v1
    LAYOUTLENS_API_KEY         sent as a Bearer token; never written anywhere
    LAYOUTLENS_VISION_MODEL    a model that accepts images (gets the screenshot in every condition)
    LAYOUTLENS_TEXT_MODEL      a text-only model (never gets the screenshot)
    LAYOUTLENS_MAX_TOKENS      default 32000
    LAYOUTLENS_REASONING_EFFORT, LAYOUTLENS_TEMPERATURE   optional; sent only when set, identically to every model call
    LAYOUTLENS_WORKERS         parallel requests, default 3

Conditions (identical wording apart from the inputs): baseline, geometry, ocr, combined, truth. The HTML source of a scene is never sent.
Nothing is reported for a model that is not configured.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import string
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bench.truth import truth_metadata
from layoutlens import analyze as ll

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "screenshots"
RUNS = ROOT / "results" / "model_runs"
TASKS = ["web-simple-light@1x", "web-medium-dark@1x", "desktop-simple-dark@1x", "desktop-medium-light@1x", "mobile-simple-light@1x", "mobile-complex-dark@1x"]
CONDITIONS = ["baseline", "geometry", "ocr", "combined", "truth"]
PLATFORM = {"web": "web page", "desktop": "desktop application window", "mobile": "mobile app screen"}


# ------------------------------------------------------------------ provider


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """KEY=VALUE lines from `path` into the environment, without overriding variables that are already set."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()


class AuthError(RuntimeError):
    pass


def settings() -> dict:
    s = {
        "base_url": os.environ.get("LAYOUTLENS_BASE_URL", "").rstrip("/"),
        "vision_model": os.environ.get("LAYOUTLENS_VISION_MODEL", ""),
        "text_model": os.environ.get("LAYOUTLENS_TEXT_MODEL", ""),
        "max_tokens": int(os.environ.get("LAYOUTLENS_MAX_TOKENS", "32000")),
        "workers": int(os.environ.get("LAYOUTLENS_WORKERS", "3")),
    }
    if os.environ.get("LAYOUTLENS_REASONING_EFFORT"):
        s["reasoning_effort"] = os.environ["LAYOUTLENS_REASONING_EFFORT"]
    if os.environ.get("LAYOUTLENS_TEMPERATURE"):
        s["temperature"] = float(os.environ["LAYOUTLENS_TEMPERATURE"])
    return s


def chat(model: str, content, s: dict, timeout: float = 1200) -> dict:
    """One Chat Completions request. Returns {"text", "usage", "seconds"} or {"error", "seconds"}."""
    key = os.environ.get("LAYOUTLENS_API_KEY")
    if not key or not s["base_url"]:
        raise SystemExit("Set LAYOUTLENS_BASE_URL and LAYOUTLENS_API_KEY (and the model names) first; see the module docstring.")
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": s["max_tokens"]}
    for k in ("reasoning_effort", "temperature"):
        if k in s:
            body[k] = s[k]
    req = urllib.request.Request(f'{s["base_url"]}/chat/completions', data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:400]
        if e.code in (401, 403):
            raise AuthError(f"HTTP {e.code}: the endpoint refused the API key ({detail[:160]})") from None
        return {"error": f"HTTP {e.code}: {detail}", "seconds": round(time.time() - t0, 2)}
    except Exception as e:  # noqa: BLE001 -- network errors are recorded, not raised
        return {"error": f"{type(e).__name__}: {e}"[:400], "seconds": round(time.time() - t0, 2)}
    msg = data["choices"][0]["message"]
    return {"text": msg.get("content") or "", "usage": data.get("usage") or {}, "seconds": round(time.time() - t0, 2)}


def image_part(png: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode(), "detail": "high"}}


# ------------------------------------------------------------------ vision check


def number_png(code: str) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (360, 140), "white")
    ImageDraw.Draw(img).text((40, 25), code, fill="black", font=ImageFont.load_default(size=80))
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def check() -> None:
    s = settings()
    report = {}
    for role in ("vision_model", "text_model"):
        model = s[role]
        if not model:
            report[role] = {"model": None, "result": "not configured"}
            continue
        results = []
        for _ in range(2):
            code = "".join(random.choice(string.digits) for _ in range(4))
            r = chat(model, [{"type": "text", "text": "What number is written in this image? Reply with the digits only."}, image_part(number_png(code))], s, timeout=300)
            results.append({"code": code, "answer": (r.get("text") or r.get("error", ""))[:120], "read": code in "".join(ch for ch in (r.get("text") or "") if ch.isdigit())})
        report[role] = {"model": model, "reads_images": any(x["read"] for x in results), "attempts": results}
        print(role, json.dumps(report[role]))
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS.parent / "vision_check.json").write_text(json.dumps(report, indent=1))


# ------------------------------------------------------------------ prompts


def metadata(task: str, condition: str) -> tuple[dict | None, float]:
    """The metadata a condition gets, and how long extracting it took (0 for baseline and truth)."""
    if condition == "baseline":
        return None, 0.0
    gt = json.loads((DATA / f"{task}.json").read_text())
    if condition == "truth":
        return truth_metadata(gt), 0.0
    rgb = ll.load(DATA / gt["image"])
    t0 = time.perf_counter()  # repeated runs of a task hit the cache: extraction cost is measured properly in bench/resources.py
    data = ll.analyze(rgb, mode=condition)
    return data, time.perf_counter() - t0


DESCRIBE = {
    "geometry": "Layout measurements of the screenshot (Pillow/OpenCV): elements with type, x, y, w, h, fill, border, corner radius, colour and parent, plus spacing. Types are estimates; text content is not included.",
    "ocr": "Text found in the screenshot by OCR (Tesseract): each line's text, x, y, w, h, confidence (0-100) and estimated font size in px.",
    "combined": "Layout measurements of the screenshot (Pillow/OpenCV) with OCR text (Tesseract): elements with type, x, y, w, h, fill, border, corner radius and parent; texts with content, position, confidence, font size, colour, containing element and alignment; plus spacing. Types are estimates.",
    "truth": "Exact layout metadata of the screen: elements with type, x, y, w, h, fill, border, corner radius and parent; texts with content, position, font size, colour, containing element and alignment.",
}


def prompt(task: str, condition: str, vision: bool, meta: dict | None) -> str:
    gt = json.loads((DATA / f"{task}.json").read_text())
    w, h = gt["css_width"], gt["css_height"]
    kind = PLATFORM[gt["scene"].split("-")[0]]
    what = "shown in the attached screenshot" if vision else ("described by the data below" if meta else f"of a {kind}")
    parts = [
        f"Recreate the user interface {what} as one self-contained HTML file: inline CSS only, no JavaScript, no external fonts or images "
        f"(draw icons with inline SVG or CSS). The page is exactly {w} x {h} CSS pixels (html and body: that width and height, margin 0, overflow hidden).",
        "Match positions, sizes, spacing, colours, borders, corner radii and text as closely as you can.",
    ]
    if meta is None and not vision:
        parts.append(f"Nothing else is known about the {kind}: make your best guess.")
    if meta is not None:
        parts.append(f"{DESCRIBE[condition]} Coordinates are in pixels; 1 pixel = 1 CSS pixel.\n```json\n{ll.to_text(meta)}\n```")
    parts.append("Reply with only the complete HTML in a single ```html code block.")
    return "\n\n".join(parts)


def extract_html(text: str, *, partial: bool = False) -> str:
    """The HTML in a reply. With `partial`, a reply cut off by the token limit (an opening ```html fence and no closing one) still yields
    what was written; browsers render an unterminated document."""
    import re

    m = re.search(r"```html\s*(.*?)```", text, re.S | re.I) or re.search(r"(<!doctype html.*</html>|<html.*</html>)", text, re.S | re.I)
    if m:
        return m.group(1)
    if partial:
        m = re.search(r"```html\s*(.*)", text, re.S | re.I) or re.search(r"(<!doctype html.*|<html.*)", text, re.S | re.I)
        return m.group(1) if m else ""
    return ""


# ------------------------------------------------------------------ runner


def run_one(role: str, model: str, task: str, condition: str, rep: int, s: dict) -> str:
    out = RUNS / role / f"{task}-{condition}-r{rep}"
    if (out / "meta.json").exists() and not json.loads((out / "meta.json").read_text()).get("error"):
        return f"skip {out.name}"  # done; a run that failed is tried again
    out.mkdir(parents=True, exist_ok=True)
    vision = role == "vision"
    meta, extract_s = metadata(task, condition)
    text = prompt(task, condition, vision, meta)
    content = [{"type": "text", "text": text}]
    if vision:
        content.append(image_part((DATA / f"{task}.png").read_bytes()))
    r = chat(model, content, s)
    html = extract_html(r.get("text", ""))
    (out / "prompt.txt").write_text(text)
    (out / "response.txt").write_text(r.get("text", r.get("error", "")))
    (out / "replica.html").write_text(html)
    usage = r.get("usage", {})
    record = {"role": role, "model": model, "task": task, "condition": condition, "rep": rep, "vision_input": vision, "seconds": r["seconds"],
              "error": r.get("error"), "html_chars": len(html), "extract_seconds": round(extract_s, 3), "metadata_chars": len(ll.to_text(meta)) if meta else 0,
              "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
              "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
              "settings": {k: v for k, v in s.items() if k not in ("workers",)}}
    (out / "meta.json").write_text(json.dumps(record, indent=1))
    return f'{out.parent.name}/{out.name} {r["seconds"]}s in={record["prompt_tokens"]} out={record["completion_tokens"]} html={len(html)}{" ERROR " + r["error"] if r.get("error") else ""}'


def run(reps: int, roles: list[str]) -> None:
    s = settings()
    jobs = []
    for rep in range(1, reps + 1):
        for task in TASKS:
            order = CONDITIONS[rep % len(CONDITIONS):] + CONDITIONS[:rep % len(CONDITIONS)]  # rotate so no condition always runs first
            for condition in order:
                for role in roles:
                    model = s[f"{role}_model"]
                    if model:
                        jobs.append((role, model, task, condition, rep))
    if not jobs:
        raise SystemExit("No model configured: set LAYOUTLENS_VISION_MODEL and/or LAYOUTLENS_TEXT_MODEL.")
    print(f"{len(jobs)} requests, {s['workers']} at a time")
    with ThreadPoolExecutor(max_workers=s["workers"]) as pool:
        try:
            for line in pool.map(lambda j: run_one(*j, s), jobs):
                print(line, flush=True)
        except AuthError as e:
            pool.shutdown(wait=False, cancel_futures=True)
            raise SystemExit(f"Stopped: {e}") from None


def main() -> None:
    ap = argparse.ArgumentParser(description="Model experiments (see module docstring).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    r = sub.add_parser("run")
    r.add_argument("--reps", type=int, default=2)
    r.add_argument("--roles", default="vision,text")
    sub.add_parser("prompt").add_argument("task_condition", nargs=2)
    args = ap.parse_args()
    if args.cmd == "check":
        check()
    elif args.cmd == "run":
        run(args.reps, args.roles.split(","))
    else:
        task, condition = args.task_condition
        meta, _ = metadata(task, condition)
        print(prompt(task, condition, vision=False, meta=meta))


if __name__ == "__main__":
    main()
