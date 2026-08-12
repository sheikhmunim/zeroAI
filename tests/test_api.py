"""In-process API tests.

These run the real app via FastAPI's TestClient, which drives the ASGI
application directly without binding a socket. That means the lifespan handler
runs -- so the model really is loaded and these are genuine end-to-end tests of
the inference path, just without the network.

They are slow (~10 s) because loading CLIP is slow. That is the honest cost of
testing the thing you actually ship rather than a mock.

Run:
    pytest tests/ -v
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def client():
    """One client for the whole session -- the model loads once, not per test."""
    from src.api import app

    # Using TestClient as a context manager is what triggers the lifespan
    # startup/shutdown. Without the `with`, the model is never loaded and every
    # request returns 503 -- a genuinely confusing failure the first time.
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    # Health must describe the model, not just say "ok" -- this is what makes
    # it usable as a deploy gate.
    assert body["model"]["backbone"].startswith("ViT-B-32")
    assert body["model"]["class_names"] == ["real", "ai"]


@pytest.mark.parametrize(
    "fixture_name,expected_label",
    [("real_sample.png", "real"), ("ai_sample.png", "ai")],
)
def test_predict_classifies_known_samples(client, fixture_name, expected_label):
    """The model should get these two right; they are easy, unambiguous cases.

    This asserts on the label rather than an accuracy figure. A smoke test's
    job is 'the wiring is intact and predictions are sane', not 'the model is
    good' -- model quality belongs in src/eval.py against the full test set.
    """
    with open(FIXTURES / fixture_name, "rb") as handle:
        response = client.post("/predict", files={"file": (fixture_name, handle, "image/png")})

    assert response.status_code == 200
    body = response.json()
    assert body["label"] == expected_label
    assert 0.0 <= body["p_ai"] <= 1.0
    assert 0.5 <= body["confidence"] <= 1.0
    assert body["filename"] == fixture_name


def test_threshold_changes_the_decision(client):
    """A near-1.0 threshold should force even a confident 'ai' to read 'real'."""
    with open(FIXTURES / "ai_sample.png", "rb") as handle:
        payload = handle.read()

    def classify(threshold: float) -> dict:
        return client.post(
            f"/predict?threshold={threshold}",
            files={"file": ("ai_sample.png", io.BytesIO(payload), "image/png")},
        ).json()

    low, high = classify(0.5), classify(0.999999)
    assert low["label"] == "ai"
    assert high["label"] == "real"
    # p_ai is a model output and must not depend on the threshold; only the
    # label does. If this ever fails, the threshold has leaked into inference.
    assert low["p_ai"] == pytest.approx(high["p_ai"])


def test_corrupt_image_is_a_client_error(client):
    """Garbage bytes with an image content-type must be 400, never 500.

    PIL raises UnidentifiedImageError here. Letting it escape would surface as
    a 500 and page whoever is on call for what is really just a bad upload.
    """
    response = client.post(
        "/predict",
        files={"file": ("evil.png", io.BytesIO(b"definitely not a png"), "image/png")},
    )
    assert response.status_code == 400
    assert "could not decode" in response.json()["detail"]


def test_non_image_content_type_rejected(client):
    response = client.post(
        "/predict", files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")}
    )
    assert response.status_code == 415


def test_missing_file_field_is_422(client):
    """Pydantic/FastAPI validation, not our code -- 422 with field detail."""
    response = client.post("/predict")
    assert response.status_code == 422


def test_out_of_range_threshold_is_422(client):
    with open(FIXTURES / "real_sample.png", "rb") as handle:
        response = client.post(
            "/predict?threshold=5", files={"file": ("real_sample.png", handle, "image/png")}
        )
    assert response.status_code == 422


def test_oversized_upload_is_413(client):
    """Guard the memory limit. 12 MB of noise exceeds the 10 MB cap."""
    big = io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * (12 * 1024 * 1024))
    response = client.post("/predict", files={"file": ("big.png", big, "image/png")})
    assert response.status_code == 413


def test_greyscale_and_rgba_images_are_handled(client):
    """Real uploads are not all RGB. Conversion happens in Detector.predict."""
    for mode in ("L", "RGBA", "P"):
        buffer = io.BytesIO()
        Image.new(mode, (64, 64), color=128 if mode == "L" else None).save(buffer, format="PNG")
        buffer.seek(0)
        response = client.post("/predict", files={"file": (f"{mode}.png", buffer, "image/png")})
        assert response.status_code == 200, f"{mode} failed: {response.text}"


def test_frontend_is_served_at_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AI Image Detector" in response.text
