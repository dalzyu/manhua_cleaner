"""Ports - interfaces for external dependencies (Dependency Inversion)."""

from .ocr_engine import OCREngine, TextDetectionResult
from .image_model import ImageModel, InpaintResult
from .cache import Cache
from .event_publisher import EventPublisher, ProcessingEvent

__all__ = [
    'OCREngine',
    'TextDetectionResult',
    'ImageModel',
    'InpaintResult',
    'Cache',
    'EventPublisher',
    'ProcessingEvent',
]
