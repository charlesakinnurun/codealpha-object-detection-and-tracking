"""Object detection abstractions.

This package defines a small, driver-like interface so any detection
backend (Ultralytics YOLO, ONNX, TensorRT, etc.) can be plugged in
without changing the rest of the pipeline.
"""

from .base import Detection, Detector
from .yolo_detector import YOLODetector

__all__ = ["Detection", "Detector", "YOLODetector"]