"""OCR functionality for text detection in images."""

import logging
from typing import Optional
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

from ..config import IMAGE_CONFIG

logger = logging.getLogger(__name__)

# Type aliases
Contour = npt.NDArray[np.int32]  # Shape (N, 1, 2) for cv2 contours
TextRegions = list[Contour]  # List of detected text region contours


class TextSpotter:
    """Text detection using PaddleOCR."""
    
    def __init__(
        self,
        pipeline_version: str = "v1.5",
        use_tensorrt: bool = True,
        precision: str = "fp16"
    ):
        """Initialize the text spotter.
        
        Args:
            pipeline_version: PaddleOCR-VL pipeline version (v1.5 or v1.0)
            use_tensorrt: Use TensorRT for acceleration if available
            precision: Model precision (fp16 or fp32)
        """
        self._pipeline: Optional[object] = None
        self.pipeline_version = pipeline_version
        self.use_tensorrt = use_tensorrt
        self.precision = precision
    
    def _ensure_loaded(self) -> None:
        """Lazy load the OCR model."""
        if self._pipeline is None:
            logger.debug(f"Loading PaddleOCR model (version={self.pipeline_version}, "
                        f"tensorrt={self.use_tensorrt}, precision={self.precision})...")
            try:
                from paddleocr import PaddleOCRVL
                self._pipeline = PaddleOCRVL()
                logger.debug("PaddleOCR model loaded")
            except ImportError as e:
                from ..exceptions import OCRError
                raise OCRError(
                    "PaddleOCR not installed. "
                    "Install PaddlePaddle referring to https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/"
                    "Then run 'pip install paddleocr' to install PaddleOCR toolkit."
                ) from e
    
    def detect(
        self,
        image_path: str | Path,
        use_layout: bool = False
    ) -> TextRegions:
        """Detect text regions in an image.
        
        Args:
            image_path: Path to the image file
            use_layout: Whether to use layout detection
            
        Returns:
            List of contour points for detected text regions
        """
        self._ensure_loaded()
        
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        logger.debug(f"Running OCR on {image_path.name}")
        
        # Run OCR
        result = self._pipeline.predict(
            str(image_path),
            pipeline_version=self.pipeline_version,
            use_layout_detection=use_layout,
            prompt_label="spotting",
            use_tensorrt=self.use_tensorrt,
            precision=self.precision,
            vlm_extra_args={'ocr_max_pixels': IMAGE_CONFIG.ocr_max_pixels}
        )
        
        # Parse results
        parsed = self._parse_result(result[0], image_path)
        logger.debug(f"Detected {len(parsed)} text regions")
        
        return parsed
    
    def _parse_result(
        self,
        raw_result: object,
        image_path: Path
    ) -> TextRegions:
        """Parse OCR result into structured format.
        
        Args:
            raw_result: Raw result from PaddleOCR
            image_path: Path to the image (for dimensions)
            
        Returns:
            List of contour points for detected text regions
        """
        result_dict = dict(raw_result)
        spotting_result = result_dict.get('spotting_res', {})
        
        texts = list(spotting_result.get("rec_texts", []))
        polygons = list(spotting_result.get("rec_polys", []))
        
        regions: TextRegions = []
        # Ensure the image exists and is readable (dimensions may be needed later)
        with Image.open(image_path):
            pass
        
        for i, text in enumerate(texts):
            if i >= len(polygons):
                logger.warning(f"Mismatch between texts and polygons at index {i}")
                break
            
            points = np.array(polygons[i], dtype=np.int32)
            
            regions.append(points.reshape(-1, 1, 2))
        
        return regions
    
    def unload(self) -> None:
        """Unload the model and free memory."""
        if self._pipeline is not None:
            logger.debug("Unloading OCR model...")
            del self._pipeline
            self._pipeline = None
            
            import gc
            gc.collect()


# Backwards compatibility alias
PaddleOCRSpotter = TextSpotter
