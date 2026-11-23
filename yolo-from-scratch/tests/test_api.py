import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="module")
def client():
    import api
    from src.model import YOLOv1
    from utils.helpers import load_config

    cfg = load_config(Path(api.PROJECT_ROOT) / "config/train.yaml")
    cfg["train"]["device"] = "cpu"
    model = YOLOv1(num_classes=20, num_boxes=2, grid_size=7, backbone_type="custom", pretrained_backbone=False)
    api.app.state.detector = api.Detector(model, cfg, "cpu")  # untrained weights: contract test only
    with TestClient(api.app) as c:
        yield c


def _png(size=(320, 240)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (120, 160, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_detect_contract_and_coordinates_in_original_image(client):
    response = client.post("/detect?conf=0.01&draw=true", files={"file": ("x.png", _png(), "image/png")})
    body = response.json()
    assert response.status_code == 200 and body["image_size"] == [320, 240]
    assert body["annotated_png_base64"]
    for det in body["detections"]:
        x1, y1, x2, y2 = det["box_xyxy"]
        assert det["label"] in client.get("/health").json()["classes"]
        assert x1 <= x2 and y1 <= y2


def test_rejects_non_images(client):
    assert client.post("/detect", files={"file": ("x.txt", b"hello", "text/plain")}).status_code == 422
