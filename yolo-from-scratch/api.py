"""Detection service: upload an image, get boxes in original-image pixels (and optionally the drawn image).

    YOLO_WEIGHTS=checkpoints/yolov1/last.pt uvicorn api:app --port 8000
"""

from __future__ import annotations

import base64
import io
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from PIL import Image
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from detect import draw_boxes, load_image, scale_boxes_to_image  # noqa: E402
from evaluate import VOC_CLASSES, batched_nms  # noqa: E402
from src.model import YOLOv1  # noqa: E402
from utils.boxes import decode_predictions  # noqa: E402
from utils.helpers import get_device, load_checkpoint, load_config  # noqa: E402

MAX_BYTES = 10 * 1024 * 1024


class Detection(BaseModel):
    label: str
    score: float
    box_xyxy: list[float]


class DetectResponse(BaseModel):
    detections: list[Detection]
    image_size: list[int]
    latency_ms: float
    annotated_png_base64: str | None = None


class Detector:
    def __init__(self, model: YOLOv1, cfg: dict, device):
        self.model, self.cfg, self.device = model.eval(), cfg, device

    @classmethod
    def load(cls, weights: Path, config: Path = PROJECT_ROOT / "config/train.yaml") -> Detector:
        cfg = load_config(config)
        device = get_device(cfg["train"]["device"])
        model = YOLOv1(num_classes=cfg["model"]["num_classes"], num_boxes=cfg["model"]["num_boxes"],
                       grid_size=cfg["model"]["grid_size"], backbone_type=cfg["model"]["backbone"],
                       pretrained_backbone=False).to(device)
        load_checkpoint(weights, model, device=device)
        return cls(model, cfg, device)

    @torch.no_grad()
    def __call__(self, image: Image.Image, conf: float) -> tuple[list[Detection], torch.Tensor, torch.Tensor,
                                                                  torch.Tensor]:
        size = self.cfg["data"]["image_size"]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        tensor, original = load_image(buffer, size)
        det = decode_predictions(self.model(tensor.to(self.device)).float().cpu(),
                                 grid_size=self.cfg["model"]["grid_size"], num_boxes=self.cfg["model"]["num_boxes"],
                                 num_classes=self.cfg["model"]["num_classes"], conf_threshold=conf,
                                 image_size=size)[0]
        keep = batched_nms(det["boxes"], det["scores"], det["labels"], self.cfg["inference"]["nms_iou_threshold"])
        boxes = scale_boxes_to_image(det["boxes"][keep], original, size)  # (cx, cy, w, h), original pixels
        scores, labels = det["scores"][keep], det["labels"][keep]
        out = [Detection(label=VOC_CLASSES[int(lab)], score=round(float(s), 4), box_xyxy=to_xyxy(b, original))
               for b, s, lab in zip(boxes, scores, labels, strict=True)]
        return out, boxes, scores, labels


def to_xyxy(box: torch.Tensor, image_size: tuple[int, int]) -> list[float]:
    """(cx, cy, w, h) -> (x1, y1, x2, y2) clipped to the image."""
    cx, cy, w, h = box.tolist()
    width, height = image_size
    return [round(max(0.0, cx - w / 2), 1), round(max(0.0, cy - h / 2), 1),
            round(min(width, cx + w / 2), 1), round(min(height, cy + h / 2), 1)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "detector", None) is None:
        app.state.detector = Detector.load(PROJECT_ROOT / os.getenv("YOLO_WEIGHTS", "checkpoints/yolov1/last.pt"))
    yield


app = FastAPI(title="YOLOv1 detection API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok" if getattr(request.app.state, "detector", None) else "loading", "classes": VOC_CLASSES}


@app.post("/detect", response_model=DetectResponse)
async def detect(request: Request, file: UploadFile = File(...), conf: float = Query(0.3, ge=0.01, le=1.0),
                 draw: bool = False) -> DetectResponse:
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image larger than 10 MB")
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(422, "not a readable image") from exc
    start = time.perf_counter()
    detections, boxes, scores, labels = request.app.state.detector(image, conf)
    latency = (time.perf_counter() - start) * 1000
    annotated = None
    if draw:
        buffer = io.BytesIO()
        draw_boxes(image.copy(), boxes, scores, labels, VOC_CLASSES).save(buffer, format="PNG")
        annotated = base64.b64encode(buffer.getvalue()).decode()
    return DetectResponse(detections=detections, image_size=list(image.size), latency_ms=round(latency, 1),
                          annotated_png_base64=annotated)
