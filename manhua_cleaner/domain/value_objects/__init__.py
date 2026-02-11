"""Value objects - immutable data with validation."""

from .geometry import Point, Quadrilateral, Contour, BoundingBox
from .config import ProcessingConfig, ModelType, Backend, OCRModelType

__all__ = [
    'Point',
    'Quadrilateral',
    'Contour',
    'BoundingBox',
    'ProcessingConfig',
    'ModelType',
    'Backend',
    'OCRModelType',
]
