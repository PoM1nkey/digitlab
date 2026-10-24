import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageOps

from digitlab.api import app
from digitlab.model import preprocess, predict, ROOT, MAX_BYTES


def encoded(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def client():
    with TestClient(app) as value:
        yield value


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


@pytest.mark.parametrize("data", [b"", b"not an image", encoded(Image.new("L", (28, 28), 255))])
def test_invalid_input(client, data):
    assert client.post("/predict", files={"file": ("x.png", data)}).status_code == 422


def test_missing_file(client):
    assert client.post("/predict").status_code == 422


def test_large_file(client):
    assert client.post("/predict", files={"file": ("x.png", b"x" * (MAX_BYTES + 1))}).status_code == 413


def test_real_model_and_probability_contract(client):
    sample = next((ROOT / "samples").glob("digit_*.png"))
    expected = int(sample.stem.split("_")[1])
    response = client.post("/predict", files={"file": (sample.name, sample.read_bytes())})
    assert response.status_code == 200
    result = response.json()
    assert result["digit"] == expected
    assert len(result["probabilities"]) == 10
    assert sum(result["probabilities"]) == pytest.approx(1, abs=1e-5)
    assert all(0 <= p <= 1 for p in result["probabilities"])


def test_polarity_and_tensor():
    sample = next((ROOT / "samples").glob("digit_*.png"))
    image = Image.open(sample).convert("L")
    first = preprocess(encoded(image))
    second = preprocess(encoded(ImageOps.invert(image)))
    assert first.shape == (1, 1, 28, 28)
    assert first.dtype == np.float32
    np.testing.assert_allclose(first, second, atol=0.01)


def test_all_samples():
    for sample in (ROOT / "samples").glob("digit_*.png"):
        assert predict(sample.read_bytes())["digit"] == int(sample.stem.split("_")[1])
