"""Core image processing pipeline."""

import logging
import os
import warnings
from collections import OrderedDict
from functools import cached_property
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image

# Try to import Pydantic, fall back to dataclasses if not available
from dataclasses import dataclass, field as dataclass_field

try:
    from pydantic import BaseModel, Field, field_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

from ..config import IMAGE_CONFIG, ModelType, WHITELIST_PRESETS
from ..exceptions import (
    ConfigurationError,
    ImageProcessingError,
    ModelError,
    OCRError,
    ValidationError,
)
from ..models import ModelFactory
from ..models.base import BaseImageModel
from .image_ops import (
    Contour,
    Quadrilateral,
    contour_to_rect,
    crop_to_contour,
    create_blend_mask,
    ensure_minimum_size,
    expand_quadrilateral,
    expand_quadrilaterals,
    get_edge_average,
    get_edge_variance,
    merge_intersecting_boxes,
)
from .ocr_base import BaseOCR
from .ocr_factory import OCRFactory
from .text_region import TextRegion
from .textbox_grouper import TextboxGrouper
from .whitelist_filter import WhitelistFilter, create_filter_from_config, WhitelistConfig as _WhitelistConfig

logger = logging.getLogger(__name__)


def _validate_path(path: Path, base_path: Optional[Path] = None) -> None:
    """Validate that a path doesn't contain path traversal attempts.
    
    Args:
        path: Path to validate
        base_path: Optional base directory that path must be within
        
    Raises:
        ValidationError: If path contains traversal sequences or is outside base_path
    """
    try:
        # Resolve to absolute path (follows symlinks)
        resolved = path.resolve()
        
        # Check for path traversal - path should not escape intended directory
        if base_path is not None:
            resolved_base = base_path.resolve()
            try:
                # Check if resolved path is within base_path
                resolved.relative_to(resolved_base)
            except ValueError:
                raise ValidationError(
                    f"Path escapes base directory: {path}",
                    field="image_path"
                )
        
        # Check for null bytes and other dangerous characters
        path_str = str(path)
        if '\x00' in path_str:
            raise ValidationError(
                "Path contains null bytes",
                field="image_path"
            )
            
    except (OSError, ValueError) as e:
        raise ValidationError(f"Invalid path: {path}", field="image_path") from e


def _diagnose_image_load_failure(image_path: Path) -> str:
    """Diagnose why an image failed to load.
    
    Args:
        image_path: Path to the image that failed to load
        
    Returns:
        Human-readable error message with diagnosis
    """
    if not image_path.exists():
        return f"Image file not found: {image_path}"
    
    if not image_path.is_file():
        return f"Path is not a file: {image_path}"
    
    # Check file size
    try:
        size = image_path.stat().st_size
        if size == 0:
            return f"Image file is empty (0 bytes): {image_path}"
    except OSError as e:
        return f"Cannot access image file: {image_path} ({e})"
    
    # Check extension support
    ext = image_path.suffix.lower()
    from ..config import SUPPORTED_IMAGE_EXTENSIONS
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        return (f"Unsupported image format: {ext}. "
                f"Supported formats: {', '.join(SUPPORTED_IMAGE_EXTENSIONS)}")
    
    # Likely codec issue
    return (f"Could not decode image: {image_path}. "
            "The file may be corrupted or OpenCV may not support this codec.")


@dataclass
class ProcessingResult:
    """Result of processing an image."""
    success: bool
    image: Optional[Image.Image] = None
    error_message: Optional[str] = None
    boxes_processed: int = 0
    boxes_smart_filled: int = 0


if PYDANTIC_AVAILABLE:
    class ProcessingConfig(BaseModel):
        """Configuration for image processing with Pydantic validation."""
        
        model_config = {"arbitrary_types_allowed": True, "validate_assignment": True}
        
        model_type: ModelType = ModelType.FLUX_2_KLEIN_4B
        device: str = "auto"
        steps: Optional[int] = Field(default=None, ge=1, le=100)
        expand_pixels: int = Field(default=IMAGE_CONFIG.default_expand_pixels, ge=0, le=500)
        color_correct: bool = True
        edge_blend: bool = True
        # Smart fill settings
        smart_fill: bool = True
        smart_fill_expand_pixels: int = Field(default=5, ge=0, le=100)
        smart_fill_threshold: float = IMAGE_CONFIG.default_color_variance_threshold
        # GPU memory management
        gpu_memory_threshold_gb: float = 6.0
        gpu_cleanup_interval: int = IMAGE_CONFIG.gpu_cleanup_interval
        # OCR settings
        ocr_model: str = "paddleocr"
        ocr_pipeline_version: str = "v1.5"
        ocr_use_tensorrt: bool = True
        ocr_precision: str = "fp16"
        ocr_workers: int = Field(default=1, ge=1, le=8)
        # Prompts
        prompt: str = "remove all text"
        extra_pass_prompt: Optional[str] = None
        # Whitelist settings
        whitelist_enabled: bool = False
        whitelist_preset: str = "none"
        whitelist_patterns: list[str] = Field(default_factory=list)
        whitelist_group_distance: int = Field(default=50, ge=10, le=200)
        # Extra pass upscaling settings
        extra_pass_upscale: bool = False
        extra_pass_upscale_factor: float = Field(default=2.0, ge=1.0, le=8.0)
        extra_pass_upscale_method: str = "lanczos"
        
        @field_validator('smart_fill_expand_pixels')
        @classmethod
        def validate_smart_fill_expand(cls, v: int, info) -> int:
            """Warn if smart fill expansion is larger than AI expansion."""
            expand_pixels = info.data.get('expand_pixels', IMAGE_CONFIG.default_expand_pixels)
            if v > expand_pixels:
                warnings.warn(
                    f"smart_fill_expand_pixels ({v}) > expand_pixels ({expand_pixels}); "
                    "smart fill will run on a larger region than AI inpainting."
                )
            return v
        
        def model_post_init(self, __context) -> None:
            """Initialize whitelist patterns from preset."""
            if not self.whitelist_patterns:
                preset_patterns = WHITELIST_PRESETS.get(self.whitelist_preset, [])
                # Use object.__setattr__ to avoid triggering validation again
                object.__setattr__(self, 'whitelist_patterns', preset_patterns.copy())
            else:
                # Combine preset with custom
                preset_patterns = WHITELIST_PRESETS.get(self.whitelist_preset, [])
                existing = set(self.whitelist_patterns)
                combined = list(self.whitelist_patterns)
                for p in preset_patterns:
                    if p not in existing:
                        combined.append(p)
                object.__setattr__(self, 'whitelist_patterns', combined)
else:
    # Fallback to dataclasses if Pydantic not available
    @dataclass
    class ProcessingConfig:
        """Configuration for image processing."""
        model_type: ModelType = ModelType.FLUX_2_KLEIN_4B
        device: str = "auto"
        steps: Optional[int] = None
        expand_pixels: int = dataclass_field(default=IMAGE_CONFIG.default_expand_pixels)
        color_correct: bool = True
        edge_blend: bool = True
        smart_fill: bool = True
        smart_fill_expand_pixels: int = 5
        smart_fill_threshold: float = IMAGE_CONFIG.default_color_variance_threshold
        gpu_memory_threshold_gb: float = 6.0
        gpu_cleanup_interval: int = IMAGE_CONFIG.gpu_cleanup_interval
        ocr_model: str = "paddleocr"
        ocr_pipeline_version: str = "v1.5"
        ocr_use_tensorrt: bool = True
        ocr_precision: str = "fp16"
        ocr_workers: int = 1
        prompt: str = "remove all text"
        extra_pass_prompt: Optional[str] = None
        whitelist_enabled: bool = False
        whitelist_preset: str = "none"
        whitelist_patterns: list[str] = dataclass_field(default_factory=list)
        whitelist_group_distance: int = 50
        extra_pass_upscale: bool = False
        extra_pass_upscale_factor: float = 2.0
        extra_pass_upscale_method: str = "lanczos"
        
        def __post_init__(self) -> None:
            """Validate and normalize configuration values."""
            if self.expand_pixels < 0:
                raise ValidationError("expand_pixels must be >= 0", field="expand_pixels")
            if self.smart_fill_expand_pixels < 0:
                raise ValidationError(
                    "smart_fill_expand_pixels must be >= 0", field="smart_fill_expand_pixels"
                )
            if self.smart_fill_expand_pixels > self.expand_pixels:
                logger.warning(
                    "smart_fill_expand_pixels (%d) > expand_pixels (%d); "
                    "smart fill will run on a larger region than AI inpainting.",
                    self.smart_fill_expand_pixels,
                    self.expand_pixels,
                )
            # Initialize whitelist_patterns - combine preset + custom
            preset_patterns = WHITELIST_PRESETS.get(self.whitelist_preset, [])
            
            if not self.whitelist_patterns:
                self.whitelist_patterns = preset_patterns.copy()
            else:
                existing = set(self.whitelist_patterns)
                for p in preset_patterns:
                    if p not in existing:
                        self.whitelist_patterns.append(p)


class _LRUCache:
    """Simple LRU cache using OrderedDict."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache: OrderedDict = OrderedDict()
    
    def get(self, key):
        """Get value from cache, moving it to most recently used."""
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None
    
    def put(self, key, value):
        """Put value into cache, evicting oldest if at capacity."""
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)  # Remove oldest (first item)
        self._cache[key] = value
    
    def clear(self):
        """Clear all entries from cache."""
        self._cache.clear()
    
    def __contains__(self, key):
        return key in self._cache
    
    def __len__(self):
        return len(self._cache)


class BatchProcessor:
    """Process multiple images for text removal."""
    
    # Maximum number of images to cache OCR results for
    OCR_CACHE_SIZE = 100
    
    def __init__(self, config: ProcessingConfig):
        """Initialize processor.
        
        Args:
            config: Processing configuration
        """
        self.config = config
        self._ocr_cache = _LRUCache(max_size=self.OCR_CACHE_SIZE)
    
    @cached_property
    def ocr(self) -> BaseOCR:
        """Lazy-load OCR model using factory."""
        ocr = OCRFactory.create(
            model_type=self.config.ocr_model,
            device=self.config.device,
            pipeline_version=self.config.ocr_pipeline_version,
            use_tensorrt=self.config.ocr_use_tensorrt,
            precision=self.config.ocr_precision
        )
        ocr.load()
        return ocr
    
    @cached_property
    def image_model(self) -> BaseImageModel:
        """Lazy-load image model."""
        model = ModelFactory.create(
            self.config.model_type,
            self.config.device
        )
        model.load()
        return model
    
    def detect_text_regions(self, image_path: Path, progress_callback: Optional[Callable[[str], None]] = None) -> list[Contour]:
        """Detect text regions in an image.
        
        Args:
            image_path: Path to image
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of contours for text regions
            
        Raises:
            OCRError: If OCR detection fails
        """
        # Check cache
        cached = self._ocr_cache.get(image_path)
        cache_info = f"Cache keys: {len(self._ocr_cache)} items"
        logger.debug(f"Looking for {image_path} in cache. {cache_info}")
        
        if cached is not None:
            logger.debug(f"Using cached OCR result for {image_path.name}")
            if progress_callback:
                progress_callback(f"Using cached OCR result for {image_path.name}")
            return cached
        
        logger.debug(f"Cache miss for {image_path.name}, running OCR")
        
        # Run OCR
        logger.info(f"Running OCR on {image_path.name}")
        if progress_callback:
            progress_callback(f"Running OCR on {image_path.name}...")
        try:
            text_regions = self.ocr.detect_with_text(image_path)
        except Exception as e:
            raise OCRError(f"OCR detection failed for {image_path.name}: {e}") from e
        
        # Apply whitelist filtering if enabled
        if self.config.whitelist_enabled and text_regions:
            if progress_callback:
                progress_callback(f"Applying whitelist filter to {len(text_regions)} regions...")
            text_regions = self._filter_by_whitelist(text_regions, progress_callback)
        
        # Convert to contours
        try:
            contours = [r.contour for r in text_regions]
        except Exception as e:
            raise ImageProcessingError(
                f"Failed to convert OCR regions to contours: {e}",
                str(image_path)
            ) from e
        
        # Cache result
        self._ocr_cache.put(image_path, contours)
        
        return contours
    
    def _filter_by_whitelist(self, regions: list[TextRegion], progress_callback: Optional[Callable[[str], None]] = None) -> list[TextRegion]:
        """Filter text regions based on whitelist configuration.
        
        Groups nearby regions into textboxes and filters based on combined text.
        
        Args:
            regions: List of detected text regions.
            progress_callback: Optional callback for logging whitelist activity.
            
        Returns:
            List of regions from non-whitelisted textboxes.
        """
        if not regions:
            return []
        
        # Create whitelist filter
        whitelist_config = WhitelistConfig(
            enabled=True,
            patterns=self.config.whitelist_patterns,
            group_distance=self.config.whitelist_group_distance
        )
        filter_ = create_filter_from_config(whitelist_config)
        
        if filter_ is None:
            if progress_callback:
                progress_callback("Whitelist: No patterns configured, skipping filter")
            return regions
        
        # Group regions into textboxes
        grouper = TextboxGrouper(max_distance=self.config.whitelist_group_distance)
        textboxes = grouper.group_regions(regions)
        
        # Get statistics before filtering
        stats = filter_.get_stats(textboxes)
        
        # Log whitelist activity
        if progress_callback:
            if stats['whitelisted_textboxes'] > 0:
                progress_callback(
                    f"Whitelist: Preserving {stats['whitelisted_textboxes']} textboxes "
                    f"({stats['whitelisted_regions']} regions)"
                )
            else:
                progress_callback(
                    f"Whitelist: Checked {stats['total_textboxes']} textboxes, "
                    f"none matched patterns"
                )
        
        if stats['whitelisted_textboxes'] > 0:
            logger.info(
                f"Whitelist: {stats['whitelisted_textboxes']}/{stats['total_textboxes']} "
                f"textboxes kept ({stats['whitelisted_regions']} regions preserved)"
            )
        
        # Filter regions
        return filter_.filter_regions(textboxes)
    
    def process_image(
        self,
        image_path: Path | str,
        progress_callback: Optional[Callable[[str], None]] = None,
        preview_callback: Optional[Callable[[np.ndarray, str, bool], None]] = None
    ) -> ProcessingResult:
        """Process a single image.
        
        New Pipeline:
        1. Run OCR on full image → raw_boxes
        2. Expand boxes by smart_fill_expand (NO MERGE) → smart_boxes
        3. Run smart fill → remaining_boxes
        4. Expand remaining boxes by (ai_expand - smart_fill_expand) → ai_pre_boxes
        5. Merge intersecting boxes → merged_boxes
        6. Run OCR on merged boxes → final_boxes
        7. AI inpainting on final_boxes
        
        Args:
            image_path: Path to input image (str or Path)
            progress_callback: Optional callback for progress updates
            preview_callback: Optional callback for preview images (image, stage, is_pre_ai)
                is_pre_ai=True means this is the consolidated pre-AI preview
            
        Returns:
            Processing result
            
        Raises:
            ValidationError: If inputs are invalid
            ImageProcessingError: If image processing fails
            OCRError: If OCR fails
            ModelError: If AI model fails
        """
        # Validate inputs
        if image_path is None:
            raise ValidationError("image_path cannot be None")
        
        # Convert to Path if string
        if isinstance(image_path, str):
            image_path = Path(image_path)
        elif not isinstance(image_path, Path):
            raise ValidationError(f"image_path must be str or Path, got {type(image_path).__name__}")
        
        # Validate path for traversal attempts
        _validate_path(image_path)
        
        # Validate callbacks
        if progress_callback is not None and not callable(progress_callback):
            raise ValidationError("progress_callback must be callable")
        
        if preview_callback is not None and not callable(preview_callback):
            raise ValidationError("preview_callback must be callable")
        
        # Check file exists
        if not image_path.exists():
            raise ImageProcessingError(f"Image file not found: {image_path}", str(image_path))
        
        # Check it's a file (not directory)
        if not image_path.is_file():
            raise ImageProcessingError(f"Path is not a file: {image_path}", str(image_path))
        
        try:
            if progress_callback:
                progress_callback(f"Loading {image_path.name}")
            
            # Load image for processing
            img_cv = cv2.imread(str(image_path))
            if img_cv is None:
                diagnosis = _diagnose_image_load_failure(image_path)
                raise ImageProcessingError(diagnosis, str(image_path))
            
            # Keep original for pre-AI preview
            preview_base = img_cv.copy()
            
            img_h, img_w = img_cv.shape[:2]
            ai_expand = self.config.expand_pixels
            smart_expand = self.config.smart_fill_expand_pixels
            delta = ai_expand - smart_expand
            
            # Step 1: Run OCR on full image
            if progress_callback:
                progress_callback("Running OCR...")
            
            raw_boxes = self.detect_text_regions(image_path, progress_callback)
            
            if not raw_boxes:
                logger.info(f"No text detected in {image_path.name}, copying original")
                img = Image.open(image_path).convert("RGB")
                return ProcessingResult(success=True, image=img, boxes_processed=0)
            
            if progress_callback:
                progress_callback(f"Detected {len(raw_boxes)} text regions")
            
            # Step 2: Expand boxes by smart_fill_expand (NO MERGE YET)
            if progress_callback:
                progress_callback(f"Expanding boxes by {smart_expand}px for smart fill...")
            
            smart_boxes = expand_quadrilaterals(raw_boxes, smart_expand)
            
            # Clamp to image bounds
            for box in smart_boxes:
                box[:, 0] = np.clip(box[:, 0], 0, img_w - 1)
                box[:, 1] = np.clip(box[:, 1], 0, img_h - 1)
            
            # Step 3: Run smart fill on smart_boxes
            smart_filled_count = 0
            remaining_boxes: list[Quadrilateral] = []
            
            if self.config.smart_fill and smart_boxes:
                if progress_callback:
                    progress_callback(f"Running smart fill on {len(smart_boxes)} regions...")
                
                img_cv, remaining_boxes, smart_filled_count = self._apply_smart_fill(
                    img_cv, smart_boxes
                )
                
                if smart_filled_count > 0:
                    logger.info(
                        f"Smart filled {smart_filled_count} regions, "
                        f"{len(remaining_boxes)} remaining for AI"
                    )
            else:
                remaining_boxes = smart_boxes
            
            # Step 4: Expand remaining boxes by (ai_expand - smart_fill_expand)
            ai_pre_boxes: list[Quadrilateral] = remaining_boxes
            if remaining_boxes and delta > 0:
                if progress_callback:
                    progress_callback(f"Expanding remaining boxes by {delta}px for AI...")
                
                ai_pre_boxes = expand_quadrilaterals(remaining_boxes, delta)
                
                # Clamp to image bounds
                for box in ai_pre_boxes:
                    box[:, 0] = np.clip(box[:, 0], 0, img_w - 1)
                    box[:, 1] = np.clip(box[:, 1], 0, img_h - 1)
            
            # Step 5: Merge intersecting boxes
            if progress_callback:
                progress_callback(f"Merging {len(ai_pre_boxes)} intersecting boxes...")
            
            merged_boxes = merge_intersecting_boxes(ai_pre_boxes, use_bounding_rect=True)
            
            # Step 6: Convert merged boxes to contours for AI inpainting
            final_boxes: list[Contour] = []
            if merged_boxes:
                if progress_callback:
                    progress_callback(f"Processing {len(merged_boxes)} merged regions...")
                
                # Convert Quadrilaterals to Contours for AI inpainting
                final_boxes = [box.reshape(-1, 1, 2) for box in merged_boxes]
            
            # Build consolidated pre-AI preview showing all stages
            if preview_callback:
                pre_ai_preview = self._build_pre_ai_preview_v2(
                    preview_base,
                    raw_boxes,
                    smart_boxes,
                    remaining_boxes,
                    ai_pre_boxes,
                    merged_boxes,
                    final_boxes,
                    smart_filled_count
                )
                preview_callback(pre_ai_preview, "pre_ai", True)
            
            # Step 7: AI inpainting on final_boxes
            if final_boxes:
                if progress_callback:
                    progress_callback(f"AI inpainting {len(final_boxes)} regions...")
                
                img_cv = self._apply_ai_inpainting(img_cv, final_boxes, progress_callback, preview_callback)
            
            # Extra pass if configured
            if self.config.extra_pass_prompt is not None:
                if progress_callback:
                    progress_callback("Running extra quality pass")
                
                # Upscale before extra pass if enabled
                if self.config.extra_pass_upscale and self.config.extra_pass_upscale_factor > 1.0:
                    upscale_factor = self.config.extra_pass_upscale_factor
                    if progress_callback:
                        progress_callback(f"Upscaling {upscale_factor}x before extra pass...")
                    
                    # Get current dimensions
                    h, w = img_cv.shape[:2]
                    new_w, new_h = int(w * upscale_factor), int(h * upscale_factor)
                    
                    # Select interpolation method
                    if self.config.extra_pass_upscale_method == "lanczos":
                        interp = cv2.INTER_LANCZOS4
                    elif self.config.extra_pass_upscale_method == "bicubic":
                        interp = cv2.INTER_CUBIC
                    else:
                        interp = cv2.INTER_LINEAR
                    
                    # Resize image
                    img_cv = cv2.resize(img_cv, (new_w, new_h), interpolation=interp)
                    
                    if progress_callback:
                        progress_callback(f"Upscaled from {w}x{h} to {new_w}x{new_h}")
                
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                
                result = self.image_model.run(
                    prompt=self.config.extra_pass_prompt,
                    image=img_pil,
                    steps=self.config.steps
                )
            else:
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                result = Image.fromarray(img_rgb)
            
            return ProcessingResult(
                success=True,
                image=result,
                boxes_processed=len(final_boxes),
                boxes_smart_filled=smart_filled_count
            )
            
        except (ValidationError, ImageProcessingError, OCRError, ModelError):
            # Re-raise our custom exceptions as-is
            raise
        except Exception as e:
            # Wrap unexpected exceptions in ImageProcessingError with full context
            logger.exception(f"Failed to process {image_path.name}")
            raise ImageProcessingError(
                f"Unexpected error processing image: {type(e).__name__}: {e}",
                str(image_path)
            ) from e
    
    def _build_pre_ai_preview(
        self,
        base_img: np.ndarray,
        raw_boxes: list[Contour],
        smart_boxes: list[Quadrilateral],
        remaining_boxes: list[Quadrilateral],
        ai_pre_boxes: list[Quadrilateral],
        merged_boxes: list[Quadrilateral],
        final_boxes: list[Contour],
        smart_filled_count: int
    ) -> np.ndarray:
        """Build a consolidated preview showing new pipeline stages.
        
        New Pipeline Visualization:
        1. Raw OCR boxes
        2. Smart expand (smart_fill_expand)
        3. After smart fill (remaining)
        4. AI pre-expand (ai_expand - smart_fill_expand)
        5. Merged boxes (final for AI)
        
        Args:
            base_img: Original image
            raw_boxes: Raw OCR boxes (initial detection)
            smart_boxes: After expanding by smart_fill_expand
            remaining_boxes: After smart fill (remaining for AI)
            ai_pre_boxes: After expanding remaining by (ai_expand - smart_fill_expand)
            merged_boxes: After merging intersecting boxes (final for AI)
            final_boxes: Final boxes for AI inpainting (same as merged_boxes)
            smart_filled_count: Number of boxes that were smart filled
            
        Returns:
            Image with all pre-AI stages overlaid
        """
        result = base_img.copy()
        
        # Draw all stages with increasing opacity
        # 1. Raw OCR boxes (faintest - initial detection)
        if raw_boxes:
            result = self._draw_boxes_on_image(
                result, raw_boxes, (0, 255, 0),  # Green
                fill_alpha=0.05, border_alpha=1.0,
                label=f"1. Raw OCR: {len(raw_boxes)}"
            )
        
        # 2. Smart expand boxes (smart_fill_expand)
        if smart_boxes:
            result = self._draw_boxes_on_image(
                result, smart_boxes, (0, 255, 255),  # Cyan
                fill_alpha=0.08, border_alpha=1.0,
                label=f"2. Smart Expand: {len(smart_boxes)}"
            )
        
        # 3. Remaining after smart fill
        if remaining_boxes:
            result = self._draw_boxes_on_image(
                result, remaining_boxes, (0, 128, 255),  # Orange
                fill_alpha=0.12, border_alpha=1.0,
                label=f"3. After Smart Fill: {len(remaining_boxes)}"
            )
        
        # 4. AI pre-expand (before merge)
        if ai_pre_boxes:
            result = self._draw_boxes_on_image(
                result, ai_pre_boxes, (255, 255, 0),  # Yellow
                fill_alpha=0.15, border_alpha=1.0,
                label=f"4. AI Pre-Expand: {len(ai_pre_boxes)}"
            )
        
        # 5. Merged boxes (final for AI)
        if merged_boxes:
            result = self._draw_boxes_on_image(
                result, merged_boxes, (255, 0, 0),  # Blue
                fill_alpha=0.25, border_alpha=1.0,
                label=f"5. Merged (Final): {len(merged_boxes)}"
            )
        
        # Add summary text at top
        summary = (
            f"Pipeline: {len(raw_boxes)} OCR -> {len(smart_boxes)} SmartExp -> "
            f"{smart_filled_count} Filled -> {len(remaining_boxes)} Remain -> "
            f"{len(ai_pre_boxes)} AIpre -> {len(merged_boxes)} Merge -> AI"
        )
        cv2.putText(result, summary, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result
    
    def _draw_boxes_on_image(
        self,
        img: np.ndarray,
        boxes: list[Contour] | list[Quadrilateral],
        color: tuple[int, int, int],
        fill_alpha: float = 0.25,
        border_alpha: float = 1.0,
        label: str = ""
    ) -> np.ndarray:
        """Draw bounding boxes on image.
        
        Args:
            img: Input image
            boxes: List of boxes (contours or quadrilaterals)
            color: BGR color tuple
            fill_alpha: Transparency for fill (0-1)
            border_alpha: Transparency for border (0-1)
            label: Optional label to display
            
        Returns:
            Image with drawn boxes
        """
        result = img.copy()
        if not boxes:
            return result
        
        # Create overlay for fill
        overlay = result.copy()
        
        for i, box in enumerate(boxes):
            if isinstance(box, np.ndarray):
                if box.ndim == 3:
                    pts = box.reshape(-1, 2)
                else:
                    pts = box
                
                pts_int = pts.astype(np.int32)
                
                # Draw fill
                cv2.fillPoly(overlay, [pts_int], color)
                
                # Draw border
                cv2.polylines(result, [pts_int], True, color, 2)
                
                # Draw number for first few boxes
                if i < 10:
                    x, y = int(pts[0][0]), int(pts[0][1])
                    cv2.putText(result, str(i+1), (x+3, y-2),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Blend fill
        if fill_alpha > 0:
            result = cv2.addWeighted(overlay, fill_alpha, result, 1 - fill_alpha, 0)
        
        return result
    
    def _apply_smart_fill(
        self,
        img: np.ndarray,
        boxes: list[Quadrilateral],
        preview_callback: Optional[Callable[[np.ndarray, str, bool], None]] = None
    ) -> tuple[np.ndarray, list[Quadrilateral], int]:
        """Apply smart fill to simple regions.
        
        The blend margin equals smart_fill_expand_pixels since that expanded
        area theoretically contains no text (just context around the text).
        This creates a smooth transition from original to filled region.
        
        Args:
            img: Input image
            boxes: List of quadrilaterals
            preview_callback: Optional callback for preview updates
            
        Returns:
            Tuple of (modified image, remaining boxes for AI, count filled)
        """
        img = img.copy()  # Work on copy to avoid side effects
        remaining: list[Quadrilateral] = []
        filled_count = 0
        
        img_h, img_w = img.shape[:2]
        
        # Blend margin equals the expand value (the expanded area has no text)
        blend_margin = self.config.smart_fill_expand_pixels
        
        for box in boxes:
            # Clamp to image bounds
            box[:, 0] = np.clip(box[:, 0], 0, img_w - 1)
            box[:, 1] = np.clip(box[:, 1], 0, img_h - 1)
            
            # Get bounding rect
            contour = box.reshape(-1, 1, 2)
            x, y, w, h = cv2.boundingRect(contour)
            
            if w <= 0 or h <= 0:
                continue
            
            cropped = img[y:y+h, x:x+w]
            if cropped.size == 0:
                continue
            
            # Check variance
            variance = get_edge_variance(cropped)
            max_variance = np.max(variance)
            
            if max_variance < self.config.smart_fill_threshold:
                # Fill with average color
                avg = get_edge_average(cropped)
                fill = np.full((h, w, 3), avg, dtype=np.uint8)
                
                # Blend across the expanded margin for smooth transition
                if self.config.edge_blend and blend_margin > 0 and blend_margin < min(w, h) // 2:
                    mask = create_blend_mask(h, w, blend_margin)
                    mask = mask[:, :, np.newaxis]
                    original = cropped.astype(np.float32)
                    fill_float = fill.astype(np.float32)
                    blended = (mask * fill_float + (1 - mask) * original)
                    img[y:y+h, x:x+w] = blended.astype(np.uint8)
                else:
                    img[y:y+h, x:x+w] = fill
                
                filled_count += 1
            else:
                remaining.append(box)
        
        return img, remaining, filled_count
    
    def _apply_ai_inpainting(
        self,
        img: np.ndarray,
        boxes: list[Contour],
        progress_callback: Optional[Callable[[str], None]] = None,
        preview_callback: Optional[Callable[[np.ndarray, str, bool], None]] = None
    ) -> np.ndarray:
        """Apply AI inpainting to text regions with side-by-side preview.
        
        Args:
            img: Input image (modified in place)
            boxes: List of contours to inpaint
            progress_callback: Optional progress callback
            preview_callback: Optional callback for preview updates
            
        Returns:
            Modified image
            
        Raises:
            ModelError: If AI model inference fails
        """
        import gc
        
        # Try to import torch for GPU memory management
        try:
            import torch
            _torch_available = True
        except ImportError:
            _torch_available = False
            torch = None  # type: ignore
        
        # GPU memory management configuration
        gpu_threshold = self.config.gpu_memory_threshold_gb
        gpu_interval = self.config.gpu_cleanup_interval
        
        def check_gpu_memory() -> bool:
            """Check if GPU memory cleanup is needed based on pressure."""
            if not _torch_available or not torch.cuda.is_available():
                return False
            allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
            return allocated > gpu_threshold
        
        # Clear GPU memory before starting
        if _torch_available and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        for i, box in enumerate(boxes):
            if progress_callback:
                progress_callback(f"Inpainting region {i+1}/{len(boxes)}")
            
            cropped = crop_to_contour(img, box)
            x, y, w, h = cv2.boundingRect(box)
            orig_h, orig_w = cropped.shape[:2]
            
            # Downscale oversized regions for safety, then upscale back after AI
            ai_input = cropped
            scale = 1.0
            if max(orig_h, orig_w) > IMAGE_CONFIG.max_size:
                scale = IMAGE_CONFIG.max_size / max(orig_h, orig_w)
                ai_input = cv2.resize(
                    cropped, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                )
            
            # Show BEFORE state (left side) - copy current image before modification
            if preview_callback:
                before_img = img.copy()
                # Highlight the box being processed
                cv2.polylines(before_img, [box.astype(np.int32)], True, (0, 255, 255), 3)
                cv2.putText(before_img, f"Before AI {i+1}/{len(boxes)}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Run AI inpainting
            try:
                padded, padding = ensure_minimum_size(ai_input)
                pad_top, pad_bottom, pad_left, pad_right = padding
                input_h, input_w = ai_input.shape[:2]
                
                # CRITICAL: Use .copy() to prevent memory sharing issues with PIL
                padded_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).copy()
                padded_pil = Image.fromarray(padded_rgb)
                
                # Clear temporary references before AI inference
                del padded, padded_rgb
                
                result = self.image_model.run(
                    prompt=self.config.prompt,
                    image=padded_pil,
                    steps=self.config.steps
                )
                
                # Clear PIL image reference
                del padded_pil
                
                # CRITICAL: Use .copy() to prevent memory sharing
                result_np = np.array(result).copy()
                result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
                
                # Clear result PIL image
                del result, result_np
                
            except Exception as e:
                raise ModelError(
                    f"AI inpainting failed for region {i+1}/{len(boxes)}: {type(e).__name__}: {e}"
                ) from e
            
            if any(padding):
                result_bgr = result_bgr[
                    pad_top:pad_top + input_h,
                    pad_left:pad_left + input_w
                ]
            
            # Ensure result_bgr matches AI input size (AI model may output slightly different dims)
            if result_bgr.shape[:2] != (input_h, input_w):
                logger.warning(
                    f"AI output shape mismatch: got {result_bgr.shape[:2]}, "
                    f"expected ({input_h}, {input_w}). Resizing."
                )
                result_bgr = cv2.resize(
                    result_bgr,
                    (input_w, input_h),
                    interpolation=cv2.INTER_LANCZOS4
                )
            
            # Restore to original crop size if downscaled
            if scale != 1.0:
                result_bgr = cv2.resize(
                    result_bgr,
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_LANCZOS4
                )
            
            actual_h, actual_w = orig_h, orig_w
            
            if self.config.color_correct:
                diff = get_edge_average(cropped) - get_edge_average(result_bgr)
                result_bgr = np.clip(
                    result_bgr.astype(np.float32) + diff, 0, 255
                ).astype(np.uint8)
            
            # Apply blending across the expanded margin for smooth transition
            # The expand_pixels area theoretically contains no text (just context)
            blend_margin = self.config.expand_pixels
            if self.config.edge_blend and blend_margin > 0 and blend_margin < min(actual_h, actual_w) // 2:
                # Use actual shape of cropped to ensure consistency
                mask = create_blend_mask(actual_h, actual_w, blend_margin)
                mask = mask[:, :, np.newaxis]
                original = cropped.astype(np.float32)
                inpainted = result_bgr.astype(np.float32)
                
                # Final safety check - should be same size now
                if inpainted.shape != original.shape:
                    logger.error(
                        f"Unexpected shape mismatch after resize: "
                        f"inpainted {inpainted.shape} vs original {original.shape}"
                    )
                    inpainted = cv2.resize(
                        inpainted, 
                        (original.shape[1], original.shape[0]),
                        interpolation=cv2.INTER_LANCZOS4
                    )
                
                blended = (mask * inpainted + (1 - mask) * original)
                img[y:y+actual_h, x:x+actual_w] = blended.astype(np.uint8)
                # Clear large temporaries
                del mask, original, inpainted, blended
            else:
                img[y:y+actual_h, x:x+actual_w] = result_bgr
            
            # Clear cropped reference
            del cropped
            
            # Show AFTER state (right side) with side-by-side comparison
            if preview_callback:
                after_img = img.copy()
                cv2.polylines(after_img, [box.astype(np.int32)], True, (0, 255, 0), 2)
                cv2.putText(after_img, f"After AI {i+1}/{len(boxes)}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Create side-by-side comparison
                comparison = np.hstack([before_img, after_img])
                preview_callback(comparison, f"ai_region_{i+1}", False)
                del after_img, comparison
            
            # Smart GPU memory cleanup: check interval OR memory pressure
            should_cleanup = (
                (i + 1) % gpu_interval == 0 or  # Regular interval
                check_gpu_memory()  # Or memory pressure
            )
            if should_cleanup and _torch_available and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        # Final cleanup
        if _torch_available and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        return img
    
    def unload_models(self) -> None:
        """Unload all models and free memory."""
        # Clear cached properties by deleting from __dict__
        if 'ocr' in self.__dict__:
            self.ocr.unload()
            del self.__dict__['ocr']
        
        if 'image_model' in self.__dict__:
            self.image_model.unload()
            del self.__dict__['image_model']
        
        self._ocr_cache.clear()
        logger.info("All models unloaded")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.unload_models()
        return False  # Don't suppress exceptions
