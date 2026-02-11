"""Domain layer - pure business logic, zero external dependencies."""

from .entities.image import Image, Mask, ProcessingResult
from .entities.text_region import TextRegion, Textbox
from .value_objects.config import ProcessingConfig, ModelType, Backend, OCRModelType
from .value_objects.geometry import Point, Quadrilateral, Contour, BoundingBox

__all__ = [
    # Entities
    'Image',
    'Mask',
    'ProcessingResult',
    'TextRegion',
    'Textbox',
    # Value Objects
    'ProcessingConfig',
    'ModelType',
    'Backend',
    'OCRModelType',
    'Point',
    'Quadrilateral',
    'Contour',
    'BoundingBox',
]
