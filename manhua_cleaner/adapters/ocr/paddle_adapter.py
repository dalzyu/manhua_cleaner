"""PaddleOCR adapter - implements OCREngine port."""

from __future__ import annotations

import logging
from pathlib import Path

from ...application.ports.ocr_engine import OCREngine, TextDetectionResult
from ...domain.entities.image import Image
from ...domain.entities.text_region import TextRegion
from ...domain.value_objects.geometry import Contour, Point

logger = logging.getLogger(__name__)


class PaddleOCRAdapter(OCREngine):
    """Adapter for PaddleOCR engine.
    
    Implements OCREngine port using PaddleOCR-VL.
    """
    
    def __init__(
        self,
        pipeline_version: str = "v1.5",
        use_tensorrt: bool = True,
        precision: str = "fp16",
        device: str = "auto"
    ):
        self._pipeline_version = pipeline_version
        self._use_tensorrt = use_tensorrt
        self._precision = precision
        self._device = device
        self._model = None
    
    @property
    def name(self) -> str:
        return "PaddleOCR-VL"
    
    @property
    def is_available(self) -> bool:
        """Check if PaddleOCR is installed."""
        try:
            import paddleocr
            return True
        except ImportError:
            return False
    
    def load(self) -> None:
        """Load PaddleOCR model."""
        if self._model is not None:
            return
        
        try:
            from paddleocr import PaddleOCRVL
            self._model = PaddleOCRVL()
            logger.info("PaddleOCR model loaded")
        except ImportError as e:
            raise RuntimeError(
                "PaddleOCR not installed. Install with: pip install paddleocr"
            ) from e
    
    def unload(self) -> None:
        """Unload model."""
        if self._model is not None:
            del self._model
            self._model = None
            import gc
            gc.collect()
            logger.info("PaddleOCR model unloaded")
    
    def detect(self, image: Image | Path) -> TextDetectionResult:
        """Detect text regions."""
        regions = self.detect_with_text(image)
        return TextDetectionResult(regions=regions)
    
    def detect_with_text(self, image: Image | Path) -> list[TextRegion]:
        """Detect text with content."""
        if self._model is None:
            self.load()
        
        if isinstance(image, Image):
            # Save to temp file if needed
            path = image.source_path
            if path is None:
                raise ValueError("Image must have source_path or be a Path")
        else:
            path = image
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        
        logger.debug(f"Running PaddleOCR on {path.name}")
        
        # Run detection
        result = self._model.predict(
            str(path),
            pipeline_version=self._pipeline_version,
            use_layout_detection=False,
            prompt_label="spotting",
            use_tensorrt=self._use_tensorrt,
            precision=self._precision,
            vlm_extra_args={'ocr_max_pixels': 2048 * 28 * 28}
        )
        
        # Parse results
        return self._parse_result(result[0])
    
    def _parse_result(self, raw_result: object) -> list[TextRegion]:
        """Parse PaddleOCR result."""
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
            
            points = polygons[i]
            contour = Contour(
                tuple(Point(float(p[0]), float(p[1])) for p in points)
            )
            confidence = scores[i] if i < len(scores) else 0.0
            
            regions.append(TextRegion(
                contour=contour,
                text=str(text),
                confidence=float(confidence)
            ))
        
        return regions
