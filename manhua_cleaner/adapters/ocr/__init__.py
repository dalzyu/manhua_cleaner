"""OCR adapters - implementations of OCREngine port."""

from .paddle_adapter import PaddleOCRAdapter

try:
    from .easyocr_adapter import EasyOCRAdapter
    __all__ = ['PaddleOCRAdapter', 'EasyOCRAdapter']
except ImportError:
    __all__ = ['PaddleOCRAdapter']
