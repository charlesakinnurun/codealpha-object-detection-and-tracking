"""Base abstractions for object detectors.

A ``Detector`` turns one raw BGR frame into a list of ``Detection`` objects.
Implementing this interface is all that is required to swap the detector
(e.g. YOLOv8 <-> YOLO11 <-> a custom ONNX runtime model).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass
class Detection:
    """A single object detection in a frame.

    Attributes:
        bbox_xyxy: Bounding box as ``(x1, y1, x2, y2)`` in pixel coordinates.
        confidence: Model confidence score in ``[0, 1]``.
        class_id: Integer class index (models differ in their label maps).
        class_name: Human readable class label.
    """

    bbox_xyxy: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str

    @property
    def xywh(self) -> Tuple[float, float, float, float]:
        """Bounding box as ``(center_x, center_y, width, height)``."""
        x1, y1, x2, y2 = self.bbox_xyxy
        return (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
            x2 - x1,
            y2 - y1,
        )


class Detector(ABC):
    """Contract every detector implementation must fulfil."""

    @abstractmethod
    def detect(self, frame) -> List[Detection]:
        """Run inference on one BGR frame and return all detections.

        The implementation should keep the confidence floor low enough
        that low-scoring detections remain available for the tracker's
        second association stage.
        """

    @abstractmethod
    def warmup(self) -> None:
        """Run one throw-away inference to trigger lazy model loading.

        Moves the expensive first-call cost (weight loading, CUDA
        context creation) outside the measured FPS window.
        """

    @property
    @abstractmethod
    def device(self) -> str:
        """Human readable device this detector is running on."""

    @property
    def names(self) -> Optional[Sequence[str]]:
        """Optional class-name lookup (index -> label)."""
        return None