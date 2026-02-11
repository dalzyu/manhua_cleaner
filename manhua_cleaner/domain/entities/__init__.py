"""Domain entities."""

from .image import Image, Mask, ProcessingResult
from .text_region import TextRegion, Textbox

__all__ = ['Image', 'Mask', 'ProcessingResult', 'TextRegion', 'Textbox']
