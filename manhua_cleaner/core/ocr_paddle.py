"""PaddleOCR implementation of text detection."""

import logging
from pathlib import Path

import numpy as np

from ..config import IMAGE_CONFIG
from ..exceptions import OCRError
from .ocr_base import BaseOCR, TextRegions, Contour
from .text_region import TextRegion

logger = logging.getLogger(__name__)


class PaddleOCRTextSpotter(BaseOCR):
    """Text detection using PaddleOCR-VL.
    
    This is the default OCR implementation optimized for
    detecting text in manga/manhua/comic images.
    
    Args:
        pipeline_version: PaddleOCR-VL pipeline version ('v1.5' or 'v1.0')
        use_tensorrt: Use TensorRT for acceleration if available
        precision: Model precision ('fp16' or 'fp32')
        device: Compute device ('cuda', 'cpu', 'auto')
    
    Example:
        >>> ocr = PaddleOCRTextSpotter(pipeline_version='v1.5')
        >>> ocr.load()
        >>> regions = ocr.detect('page_01.jpg')
        >>> print(f"Found {len(regions)} text regions")
        >>> ocr.unload()
    """
    
    def __init__(
        self,
        pipeline_version: str = "v1.5",
        use_tensorrt: bool = True,
        precision: str = "fp16",
        device: str = "auto"
    ):
        """Initialize PaddleOCR text spotter."""
        super().__init__(device)
        self.pipeline_version = pipeline_version
        self.use_tensorrt = use_tensorrt
        self.precision = precision
    
    def load(self) -> None:
        """Load the PaddleOCR model.
        
        Raises:
            OCRError: If PaddleOCR is not installed or loading fails
        """
        if self.is_loaded:
            logger.debug("PaddleOCR model already loaded")
            return
        
        logger.debug(
            f"Loading PaddleOCR model (version={self.pipeline_version}, "
            f"tensorrt={self.use_tensorrt}, precision={self.precision})..."
        )
        
        try:
            from paddleocr import PaddleOCRVL
            self._model = PaddleOCRVL()
            logger.debug("PaddleOCR model loaded successfully")
        except ImportError as e:
            raise OCRError(
                "PaddleOCR not installed. "
                "Install PaddlePaddle from https://www.paddlepaddle.org.cn/ "
                "Then run: pip install paddleocr"
            ) from e
        except Exception as e:
            raise OCRError(f"Failed to load PaddleOCR model: {e}") from e
    
    def detect(self, image_path: Path | str) -> TextRegions:
        """Detect text regions in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of contour points for detected text regions
            
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
            List of TextRegion objects with contour and text
            
        Raises:
            OCRError: If detection fails
            FileNotFoundError: If image not found
        """
        if not self.is_loaded:
            self.load()
        
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        logger.debug(f"Running PaddleOCR on {image_path.name}")
        
        try:
            result = self._model.predict(
                str(image_path),
                pipeline_version=self.pipeline_version,
                use_layout_detection=False,
                prompt_label="spotting",
                use_tensorrt=self.use_tensorrt,
                precision=self.precision,
                vlm_extra_args={'ocr_max_pixels': IMAGE_CONFIG.ocr_max_pixels}
            )
        except Exception as e:
            raise OCRError(f"PaddleOCR detection failed: {e}") from e
        
        # Parse results
        parsed = self._parse_result_with_text(result[0], image_path)
        logger.debug(f"Detected {len(parsed)} text regions")
        
        return parsed
    
    def _parse_result_with_text(
        self,
        raw_result: object,
        image_path: Path
    ) -> list[TextRegion]:
        """Parse PaddleOCR result into TextRegion objects.
        
        Args:
            raw_result: Raw result from PaddleOCR
            image_path: Path to the image
            
        Returns:
            List of TextRegion objects
        """
        result_dict = dict(raw_result)
        spotting_result = result_dict.get('spotting_res', {})
        
        texts = list(spotting_result.get("rec_texts", []))
        polygons = list(spotting_result.get("rec_polys", []))
        scores = list(spotting_result.get("rec_scores", []))
        
        regions: list[TextRegion] = []
        
        for i, text in enumerate(texts):
            if i >= len(polygons):
                logger.warning(f"Mismatch between texts and polygons at index {i}")
                break
            
            points = np.array(polygons[i], dtype=np.int32)
            contour = points.reshape(-1, 1, 2)
            confidence = scores[i] if i < len(scores) else 0.0
            
            regions.append(TextRegion(
                contour=contour,
                text=str(text),
                confidence=float(confidence)
            ))
        
        return regions
    
    def _parse_result(
        self,
        raw_result: object,
        image_path: Path
    ) -> TextRegions:
        """Parse PaddleOCR result into contours only (backwards compatibility).
        
        Args:
            raw_result: Raw result from PaddleOCR
            image_path: Path to the image
            
        Returns:
            List of contour points
        """
        regions = self._parse_result_with_text(raw_result, image_path)
        return [r.contour for r in regions]
    
    def unload(self) -> None:
        """Unload the PaddleOCR model and free memory."""
        if self._model is not None:
            logger.debug("Unloading PaddleOCR model...")
            del self._model
            self._model = None
            
            import gc
            gc.collect()
            logger.debug("PaddleOCR model unloaded")
    
    def get_model_info(self) -> dict:
        """Get information about the configured model.
        
        Returns:
            Dictionary with model configuration
        """
        return {
            "name": "PaddleOCR-VL",
            "version": self.pipeline_version,
            "tensorrt": self.use_tensorrt,
            "precision": self.precision,
            "device": self.device,
        }


# Backwards compatibility alias
TextSpotter = PaddleOCRTextSpotter
