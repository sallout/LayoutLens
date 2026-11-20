"""The analyzer on small synthetic screens with known geometry and text."""

from __future__ import annotations

import shutil

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from layoutlens import analyze, geometry, ocr

HAS_TESSERACT = bool(shutil.which("tesseract") or any(__import__("os").path.exists(p) for p in ("/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract", "/usr/bin/tesseract")))
needs_tesseract = pytest.mark.skipif(not HAS_TESSERACT, reason="tesseract is not installed")


def screen() -> Image.Image:
    """A light page with a bordered white card holding a blue button with a white label, and a dark heading above the card."""
    img = Image.new("RGB", (480, 300), (246, 247, 249))
    d = ImageDraw.Draw(img)
    d.text((24, 16), "Account settings", fill=(17, 24, 39), font=ImageFont.load_default(size=26))
    d.rectangle([24, 70, 455, 279], fill=(255, 255, 255), outline=(217, 222, 230))
    d.rectangle([48, 200, 207, 247], fill=(37, 99, 235))
    d.text((80, 211), "Save", fill=(255, 255, 255), font=ImageFont.load_default(size=22))
    return img


def boxes(data: dict) -> dict:
    return {(e["x"], e["y"], e["w"], e["h"]): e for e in data["elements"]}


def test_geometry_measures_fills_borders_and_parents():
    data = analyze.analyze(screen(), mode="geometry")
    assert data["image"] == {"width": 480, "height": 300} and data["background"] == "#f6f7f9" and data["theme"] == "light"
    b = boxes(data)
    card, button = b[(24, 70, 432, 210)], b[(48, 200, 160, 48)]
    assert card["fill"] == "#ffffff" and card["border"] == {"sides_px": [1, 1, 1, 1], "color": "#d9dee6"}
    assert button["fill"] == "#2563eb" and button["parent"] == card["id"]
    assert any(e["type"] == "text" for e in data["elements"]), "the label and heading are found as text boxes (without content)"


def test_rounded_corner_radius_is_measured_from_the_fill():
    img = Image.new("RGB", (200, 120), (255, 255, 255))
    ImageDraw.Draw(img).rounded_rectangle([20, 20, 179, 99], radius=16, fill=(37, 99, 235))
    el = next(e for e in analyze.analyze(img, mode="geometry")["elements"] if e.get("fill") == "#2563eb")
    assert abs(el["radius"] - 16) <= 2


@needs_tesseract
def test_ocr_reads_dark_and_light_text():
    data = analyze.analyze(screen(), mode="ocr")
    texts = {t["text"] for t in data["texts"]}
    assert "Account settings" in texts and "Save" in texts, texts
    assert all(0 <= t["conf"] <= 100 and t["font_px"] > 0 for t in data["texts"])


@needs_tesseract
def test_combined_joins_text_to_its_container_with_colour_and_alignment():
    data = analyze.analyze(screen(), mode="combined")
    button = boxes(data)[(48, 200, 160, 48)]
    save = next(t for t in data["texts"] if t["text"] == "Save")
    from bench.metrics import delta_e

    assert save["in"] == button["id"] and save["align"] == "left"
    assert delta_e(save["color"], "#ffffff") < 5, "text colour comes from anti-aliased ink: close to, not exactly, white"
    ids = {e["id"] for e in data["elements"]} | {t["id"] for t in data["texts"]}
    for g in data["spacing"]["gaps"]:
        assert g["a"] in ids and g["b"] in ids, "spacing refers only to elements and texts that are in the output"


def test_results_are_cached_by_content_and_mode():
    img = screen()
    assert analyze.analyze(img, mode="geometry") is analyze.analyze(screen(), mode="geometry")


def test_large_images_are_analysed_at_every_nth_pixel_with_coordinates_scaled_back(monkeypatch):
    monkeypatch.setattr(analyze, "MAX_PIXELS", 40_000)
    img = Image.new("RGB", (400, 300), (255, 255, 255))
    ImageDraw.Draw(img).rectangle([100, 60, 299, 179], fill=(220, 38, 38))
    geo = analyze.geometry_full(np.asarray(img))
    assert geo["image"]["analysed_step"] == 2
    el = next(e for e in geo["elements"] if e["fill"] == "#dc2626")
    assert all(abs(a - b) <= 2 for a, b in zip(el["bbox"], [100, 60, 200, 120]))


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        analyze.analyze(screen(), mode="pixels")


def test_lines_are_grouped_left_to_right_and_icons_read_as_letters_are_dropped():
    W = ocr.Word
    words = [W("world", (60, 10, 40, 12), 95), W("hello", (10, 11, 44, 12), 95), W("next", (10, 40, 30, 12), 95), W("O", (200, 5, 30, 30), 70)]
    lines = ocr.group_lines(words)
    assert [ln["text"] for ln in lines] == ["hello world", "next"]


def test_font_size_estimate_follows_letter_shapes():
    assert ocr.font_size("Save", 10) == pytest.approx(10 / 0.72, rel=0.01)
    assert ocr.font_size("ace", 7) == pytest.approx(7 / 0.52, rel=0.01)
    assert ocr.font_size("Typing", 13) == pytest.approx(13 / 0.93, rel=0.01)


def test_to_text_is_valid_json():
    import json

    data = analyze.analyze(screen(), mode="geometry")
    assert json.loads(analyze.to_text(data)) == json.loads(json.dumps(data))


def test_geometry_module_has_no_ocr_dependency():
    assert "subprocess" not in vars(geometry)
