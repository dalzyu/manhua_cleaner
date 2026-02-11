"""OCR Engine port - interface for text detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from ...domain.entities.image import Image
from ...domain.entities.text_region import TextRegion


@dataclass(frozen=True, slots=True)
class TextDetectionResult:
    """Result of text detection."""
    regions: list[TextRegion]
    processing_time_ms: float = 0.0
    
    @property
    def region_count(self) -> int:
        return len(self.regions)
    
    @property
    def is_empty(self) -> bool:
        return len(self.regions) == 0


@runtime_checkable
class OCREngine(Protocol):
    """Port for OCR text detection engines.
    
    Implementations: PaddleOCR, EasyOCR, Tesseract, etc.
    """
    
    @property
    def name(self) -> str:
        """Engine name."""
        ...
    
    @property
    def is_available(self) -> bool:
        """Check if engine dependencies are installed."""
        ...
    
    def load(self) -> None:
        """Load model into memory."""
        ...
    
    def unload(self) -> None:
        """Unload model and free memory."""
        ...
    
    def detect(self, image: Image | Path) -> TextDetectionResult:
        """Detect text regions in image.
        
        Args:
            image: Image to process
            
        Returns:
            Detection result with text regions
        """
        ...
    
    def detect_with_text(self, image: Image | Path) -> list[TextRegion]:
        """Detect text regions with recognized text.
        
        Args:
            image: Image to process
            
        Returns:
            List of regions with text content
        """
        ...
