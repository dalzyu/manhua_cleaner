"""Core processing functionality."""

from .ocr_base import BaseOCR, TextRegions, Contour
from .ocr_factory import OCRFactory, OCRModelType, get_ocr_model
from .ocr_paddle import PaddleOCRTextSpotter
from .text_region import TextRegion, Textbox
from .textbox_grouper import TextboxGrouper
from .whitelist_filter import (
    WhitelistFilter,
    WhitelistConfig,
    create_filter_from_config,
    WHITELIST_PRESETS,
)
from .image_ops import (
    Quadrilateral,
    expand_quadrilateral,
    expand_quadrilaterals,
    merge_intersecting_boxes,
    create_blend_mask,
    get_edge_average,
    get_edge_variance,
)
from .processor import (
    BatchProcessor,
    ProcessingConfig,
    ProcessingResult,
)
from .worker import OCRWorkerPool, OCRResult, OCRTask

__all__ = [
    # OCR Base
    'BaseOCR',
    'TextRegions',
    'Contour',
    # OCR Factory
    'OCRFactory',
    'OCRModelType',
    'get_ocr_model',
    # OCR Implementations
    'PaddleOCRTextSpotter',
    # Text Region and Whitelist
    'TextRegion',
    'Textbox',
    'TextboxGrouper',
    'WhitelistFilter',
    'WhitelistConfig',
    'create_filter_from_config',
    'WHITELIST_PRESETS',
    # Image operations
    'Quadrilateral',
    'expand_quadrilateral',
    'expand_quadrilaterals',
    'merge_intersecting_boxes',
    'create_blend_mask',
    'get_edge_average',
    'get_edge_variance',
    # Processing
    'BatchProcessor',
    'ProcessingConfig',
    'ProcessingResult',
    # Worker
    'OCRWorkerPool',
    'OCRResult',
    'OCRTask',
]

# Try to import EasyOCR (optional)
try:
    from .ocr_easyocr import EasyOCRTextSpotter
    __all__.append('EasyOCRTextSpotter')
except ImportError:
    pass
