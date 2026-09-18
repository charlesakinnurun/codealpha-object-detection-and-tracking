"""Ultralytics YOLO detector.

Wraps ``ultralytics.YOLO`` (a modern, easily deployable pre-trained YOLO
implementation: YOLOv8 / YOLO11 / YOLO-NAS style models). A pre-trained
model is used — nothing is trained here.

Detection thresholding
----------------------
The detector is deliberately configured with a *low* confidence floor
(``conf``) so that medium-confidence boxes are still emitted as
candidates for the tracker's second association stage. The user-facing
``--conf`` threshold is applied downstream in the tracker/most relevant
display logic.
"""

from typing import List, Optional, Sequence

from .base import Detection, Detector

try:
    import torch  # noqa: F401  (used only for CUDA detection)
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - exercised only without deps
    raise ImportError(
        "Missing inference dependencies. Install them with:\n"
        "    pip install ultralytics\n"
        "See requirements.txt for the full list."
    ) from exc


def resolve_device(device: Optional[str] = None) -> str:
    """Pick the best available compute device.

    ``auto`` uses a CUDA GPU when one is visible and falls back to CPU.
    Explicit values (``cpu``, ``0``, ``cuda:0``) are passed through.
    """
    if device is None or str(device).lower() in ("auto", ""):
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return str(device)


class YOLODetector(Detector):
    """Detection backend built on Ultralytics YOLO."""

    def __init__(
        self,
        weights: str,
        device: Optional[str] = "auto",
        imgsz: int = 640,
        conf: float = 0.1,
        iou: float = 0.6,
        classes: Optional[Sequence[int]] = None,
    ) -> None:
        try:
            self.model = YOLO(weights, task="detect")
        except Exception as exc:  # pragma: no cover - depends on weights
            raise RuntimeError(
                f"Failed to load model weights from '{weights}'. "
                "Make sure the file exists and is a valid Ultralytics "
                "model (PT/ONNX/Engine). Run "
                "'python scripts/download_models.py' to fetch a pre-trained model."
            ) from exc

        self.device_str = resolve_device(device)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.classes = list(classes) if classes else None

    def detect(self, frame) -> List[Detection]:
        result = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device_str,
            classes=self.classes,
            verbose=False,
        )[0]

        detections: List[Detection] = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        names = self.model.names

        for i in range(len(scores)):
            detections.append(
                Detection(
                    bbox_xyxy=tuple(xyxy[i].tolist()),
                    confidence=float(scores[i]),
                    class_id=int(cls_ids[i]),
                    class_name=str(names[cls_ids[i]]),
                )
            )
        return detections

    def warmup(self) -> None:
        import numpy as np

        seed = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.detect(seed)

    @property
    def device(self) -> str:
        return self.device_str

    @property
    def names(self) -> Optional[Sequence[str]]:
        names = getattr(self.model, "names", None)
        if names is None:
            return None
        return [names[i] for i in range(len(names))]