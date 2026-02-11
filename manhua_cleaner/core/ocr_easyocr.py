"""EasyOCR implementation of text detection."""

import logging
from pathlib import Path

import numpy as np

from ..exceptions import OCRError
from .ocr_base import BaseOCR, TextRegions, Contour
from .text_region import TextRegion

logger = logging.getLogger(__name__)


class EasyOCRTextSpotter(BaseOCR):
    """Text detection using EasyOCR.
    
    EasyOCR is an alternative OCR engine that supports 80+ languages
    and has simpler dependencies than PaddleOCR. It's a good choice
    for users who have trouble installing PaddlePaddle.
    
    Note: EasyOCR may be slower than PaddleOCR and has different
    accuracy characteristics. It's recommended to test with your
    specific image types.
    
    Args:
        lang_list: List of language codes to recognize (default: ['en'])
        gpu: Whether to use GPU acceleration
        device: Compute device ('cuda', 'cpu', 'auto')
    
    Example:
        >>> ocr = EasyOCRTextSpotter(lang_list=['en', 'ch_sim'])
        >>> ocr.load()
        >>> regions = ocr.detect('page_01.jpg')
        >>> print(f"Found {len(regions)} text regions")
        >>> ocr.unload()
    
    Installation:
        pip install easyocr
    """
    
    def __init__(
        self,
        lang_list: list[str] | None = None,
        gpu: bool = True,
        device: str = "auto"
    ):
        """Initialize EasyOCR text spotter."""
        super().__init__(device)
        self.lang_list = lang_list or ['en']
        self.gpu = gpu and (device in ('cuda', 'auto'))
        self._reader = None
    
    def load(self) -> None:
        """Load the EasyOCR model.
        
        Raises:
            OCRError: If EasyOCR is not installed or loading fails
        """
        if self.is_loaded:
            logger.debug("EasyOCR model already loaded")
            return
        
        logger.debug(
            f"Loading EasyOCR model (languages={self.lang_list}, gpu={self.gpu})..."
        )
        
        try:
            import easyocr
            self._reader = easyocr.Reader(
                self.lang_list,
                gpu=self.gpu,
                verbose=False  # Reduce console output
            )
            logger.debug("EasyOCR model loaded successfully")
        except ImportError as e:
            raise OCRError(
                "EasyOCR not installed. "
                "Install with: pip install easyocr"
            ) from e
        except Exception as e:
            raise OCRError(f"Failed to load EasyOCR model: {e}") from e
    
    def detect(self, image_path: Path | str) -> TextRegions:
        """Detect text regions in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of contour points for detected text regions.
            Each contour is a quadrilateral (4 points).
            
        Raises:
            OCRError: If detection fails
            FileNotFoundError: If image not found
        """
        # Use detect_with_text and extract contours
        regions_with_text = self.detect_with_text(image_path)
        return [r.contour for r in regions_with_text]
    
    def detect_with_text(self, image_path: Path | str) -> list[TextRegion]:
        """Detect text regions with their recognized text.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of TextRegion objects with contour and text.
            
        Raises:
            OCRError: If detection fails
            FileNotFoundError: If image not found
        """
        if not self.is_loaded:
            self.load()
        
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        logger.debug(f"Running EasyOCR on {image_path.name}")
        
        try:
            # EasyOCR returns list of (bbox, text, confidence)
            # bbox format: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]] (top-left, top-right, bottom-right, bottom-left)
            results = self._reader.readtext(str(image_path))
        except Exception as e:
            raise OCRError(f"EasyOCR detection failed: {e}") from e
        
        # Convert to TextRegion objects
        regions: list[TextRegion] = []
        for bbox, text, confidence in results:
            # Convert bbox to contour format (N, 1, 2)
            points = np.array(bbox, dtype=np.int32).reshape(-1, 1, 2)
            
            regions.append(TextRegion(
                contour=points,
                text=str(text),
                confidence=float(confidence)
            ))
            
            logger.debug(f"Detected text: '{text}' (confidence: {confidence:.2f})")
        
        logger.debug(f"Detected {len(regions)} text regions")
        return regions
    
    def unload(self) -> None:
        """Unload the EasyOCR model and free memory."""
        if self._reader is not None:
            logger.debug("Unloading EasyOCR model...")
            del self._reader
            self._reader = None
            self._model = None  # Also clear base class reference
            
            import gc
            gc.collect()
            logger.debug("EasyOCR model unloaded")
    
    def get_model_info(self) -> dict:
        """Get information about the configured model.
        
        Returns:
            Dictionary with model configuration
        """
        return {
            "name": "EasyOCR",
            "languages": self.lang_list,
            "gpu": self.gpu,
            "device": self.device,
        }
