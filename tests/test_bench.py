"""Scoring helpers and prompt construction."""

from __future__ import annotations

import json

import pytest

from bench import metrics, models


def test_edit_distances():
    assert metrics.levenshtein("kitten", "sitting") == 3
    assert metrics.cer("Save chnges", "Save changes") == pytest.approx(1 / 12)
    assert metrics.wer("Save chnges", "Save changes") == 0.5
    assert metrics.cer("", "abc") == 1.0 and metrics.cer("x" * 20, "ab") == 1.0


def test_boxes_and_matching():
    assert metrics.iou([0, 0, 10, 10], [5, 0, 10, 10]) == pytest.approx(50 / 150)
    pairs = metrics.match_boxes([[0, 0, 10, 10], [50, 50, 10, 10]], [[51, 50, 10, 10], [0, 0, 10, 9], [200, 200, 5, 5]])
    assert sorted((i, j) for i, j, _ in pairs) == [(0, 1), (1, 0)]


def test_colour_difference():
    assert metrics.delta_e("#ffffff", "#ffffff") == 0
    assert metrics.delta_e("#000000", "#ffffff") == pytest.approx(100, abs=0.5)
    assert 0 < metrics.delta_e("#2563eb", "#2662ea") < 1


def test_prompts_differ_only_in_their_inputs_and_text_models_never_get_the_image():
    meta = {"image": {"width": 10, "height": 10}, "elements": [], "texts": []}
    task = models.TASKS[0]
    vision = models.prompt(task, "geometry", vision=True, meta=meta)
    text = models.prompt(task, "geometry", vision=False, meta=meta)
    assert "attached screenshot" in vision and "attached screenshot" not in text
    assert "```json" in text and models.DESCRIBE["geometry"] in text
    base = models.prompt(task, "baseline", vision=False, meta=None)
    assert "```json" not in base and "best guess" in base


def test_html_is_taken_from_the_reply():
    assert models.extract_html("Here:\n```html\n<html><body>x</body></html>\n```") == "<html><body>x</body></html>\n"
    assert models.extract_html("<!DOCTYPE html><html><body></body></html>").startswith("<!DOCTYPE")
    assert models.extract_html("no code") == ""


def test_no_request_is_made_without_credentials(monkeypatch):
    monkeypatch.delenv("LAYOUTLENS_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        models.chat("m", "hi", models.settings())


def test_ground_truth_metadata_has_the_combined_format():
    from bench.truth import truth_metadata

    gt = json.loads((models.DATA / f"{models.TASKS[0]}.json").read_text())
    meta = truth_metadata(gt)
    assert {"elements", "texts", "background"} <= set(meta)
    assert any(t["text"] == "Ship dashboards your team will read" for t in meta["texts"])
