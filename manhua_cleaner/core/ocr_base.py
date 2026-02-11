"""Abstract base class for OCR text detection."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt

from .text_region import TextRegion

logger = logging.getLogger(__name__)

# Type aliases
Contour = npt.NDArray[np.int32]  # Shape (N, 1, 2) for cv2 contours
TextRegions = list[Contour]  # List of detected text region contours


class BaseOCR(ABC):
    """Abstract base class for OCR text detection.
    
    This class defines the interface that all OCR implementations must follow.
    It provides lazy loading, consistent return types, and proper cleanup.
    
    Example:
        class MyOCR(BaseOCR):
            def load(self) -> None:
                self._model = load_my_model()
            
            def detect(self, image_path: Path) -> TextRegions:
                results = self._model.predict(image_path)
                return self._convert_to_contours(results)
            
            def unload(self) -> None:
                del self._model
                self._model = None
    """
    
    def __init__(self, device: str = "auto"):
        """Initialize OCR detector.
        
        Args:
            device: Compute device ('cuda', 'cpu', 'auto')
        """
        self.device = self._resolve_device(device)
        self._model: Optional[object] = None
    
    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device.
        
        Args:
            device: Device string ('cuda', 'cpu', 'auto')
            
        Returns:
            Resolved device string
        """
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
            return "cpu"
        return device
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None
    
    @abstractmethod
    def load(self) -> None:
        """Load the OCR model into memory.
        
        This method should:
        1. Import the OCR library
        2. Initialize the model
        3. Store it in self._model
        
        Raises:
            OCRError: If model loading fails
        """
        pass
    
    @abstractmethod
    def detect(self, image_path: Path | str) -> TextRegions:
        """Detect text regions in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of contour points for detected text regions.
            Each contour is an array of shape (N, 1, 2) where N is the
            number of points in the polygon.
            
        Raises:
            OCRError: If detection fails
            FileNotFoundError: If image not found
        """
        pass
    
    @abstractmethod
    def detect_with_text(self, image_path: Path | str) -> list[TextRegion]:
        """Detect text regions with their recognized text.
        
        This method returns TextRegion objects containing both the contour
        and the recognized text content, which is needed for whitelist filtering.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of TextRegion objects with contour and text.
            
        Raises:
            OCRError: If detection fails
            FileNotFoundError: If image not found
        """
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Unload the model and free memory.
        
        This method should:
        1. Delete the model reference
        2. Run garbage collection if needed
        3. Clear any caches
        """
        pass
    
    def detect_safe(self, image_path: Path | str) -> TextRegions:
        """Detect text regions with automatic loading.
        
        This is a convenience method that ensures the model is loaded
        before detection and handles common errors.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of contours for text regions (empty if none found)
        """
        from ..exceptions import OCRError
        
        # Ensure loaded
        if not self.is_loaded:
            try:
                self.load()
            except Exception as e:
                raise OCRError(f"Failed to load OCR model: {e}") from e
        
        # Validate path
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Run detection
        try:
            return self.detect(image_path)
        except Exception as e:
            if isinstance(e, OCRError):
                raise
            raise OCRError(f"OCR detection failed: {e}") from e
    
    def detect_with_text_safe(self, image_path: Path | str) -> list[TextRegion]:
        """Detect text regions with text content and automatic loading.
        
        This is a convenience method that ensures the model is loaded
        before detection and handles common errors.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of TextRegion objects with contour and text.
        """
        from ..exceptions import OCRError
        
        # Ensure loaded
        if not self.is_loaded:
            try:
                self.load()
            except Exception as e:
                raise OCRError(f"Failed to load OCR model: {e}") from e
        
        # Validate path
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Run detection
        try:
            return self.detect_with_text(image_path)
        except Exception as e:
            if isinstance(e, OCRError):
                raise
            raise OCRError(f"OCR detection failed: {e}") from e
    
    def __enter__(self):
        """Context manager entry."""
        self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload()
