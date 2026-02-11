"""Text removal service - orchestrates the processing pipeline."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from ...domain.entities.image import Image, ProcessingResult
from ...domain.entities.text_region import Textbox
from ...domain.services.box_merging import merge_intersecting_boxes
from ...domain.services.smart_fill import apply_smart_fill
from ...domain.value_objects.config import ProcessingConfig
from ...domain.value_objects.geometry import Quadrilateral, BoundingBox
from ..ports.cache import Cache, MemoryCache
from ..ports.event_publisher import EventPublisher, ProcessingEvent, SimpleEventPublisher
from ..ports.image_model import ImageModel
from ..ports.ocr_engine import OCREngine

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Context passed through pipeline steps."""
    image: Image
    config: ProcessingConfig
    textboxes: list[Textbox] = field(default_factory=list)
    smart_filled_boxes: list[Quadrilateral] = field(default_factory=list)
    ai_boxes: list[Quadrilateral] = field(default_factory=list)
    result: ProcessingResult | None = None


class PipelineStep:
    """Base class for pipeline steps."""
    
    def __init__(self, name: str):
        self.name = name
    
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute this step and return updated context."""
        raise NotImplementedError


class DetectTextStep(PipelineStep):
    """Step 1: Detect text regions."""
    
    def __init__(self, ocr: OCREngine, cache: Cache | None = None):
        super().__init__("detect_text")
        self._ocr = ocr
        self._cache = cache
    
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        cache_key = None
        if ctx.image.source_path and self._cache:
            cache_key = f"ocr:{ctx.image.source_path}"
            cached = self._cache.get(cache_key)
            if cached:
                logger.debug("Using cached OCR result")
                ctx.textboxes = cached
                return ctx
        
        result = self._ocr.detect(ctx.image)
        
        # Group regions into textboxes
        from ...domain.services.text_grouping import group_regions
        ctx.textboxes = group_regions(
            result.regions,
            max_distance=ctx.config.whitelist_group_distance
        )
        
        if cache_key:
            self._cache.set(cache_key, ctx.textboxes)
        
        logger.info(f"Detected {len(ctx.textboxes)} text boxes")
        return ctx


class ApplyWhitelistStep(PipelineStep):
    """Step 2: Filter whitelisted textboxes."""
    
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.config.whitelist_enabled or not ctx.config.whitelist_patterns:
            return ctx
        
        filtered = []
        whitelisted = []
        
        for tb in ctx.textboxes:
            if tb.is_whitelisted(ctx.config.whitelist_patterns):
                whitelisted.append(tb)
            else:
                filtered.append(tb)
        
        ctx.textboxes = filtered
        logger.info(f"Whitelisted {len(whitelisted)} boxes, {len(filtered)} remaining")
        return ctx


class SmartFillStep(PipelineStep):
    """Step 3: Smart fill simple regions."""
    
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.config.smart_fill:
            ctx.ai_boxes = [tb.regions[0].contour.to_quadrilateral() for tb in ctx.textboxes]
            return ctx
        
        # Get quadrilaterals for all textboxes
        boxes = [tb.regions[0].contour.to_quadrilateral() for tb in ctx.textboxes]
        
        # Expand for smart fill
        expanded = [box.expand(ctx.config.smart_fill_expand_pixels) for box in boxes]
        
        # Apply smart fill
        ctx.image, remaining, filled = apply_smart_fill(
            ctx.image,
            expanded,
            threshold=ctx.config.smart_fill_threshold
        )
        
        ctx.smart_filled_boxes = [expanded[i] for i in filled]
        ctx.ai_boxes = remaining
        
        logger.info(f"Smart filled {len(filled)} boxes, {len(remaining)} for AI")
        return ctx


class MergeBoxesStep(PipelineStep):
    """Step 4: Merge intersecting boxes."""
    
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        if len(ctx.ai_boxes) <= 1:
            return ctx
        
        delta = ctx.config.expand_pixels - ctx.config.smart_fill_expand_pixels
        if delta > 0:
            ctx.ai_boxes = [box.expand(delta) for box in ctx.ai_boxes]
        
        ctx.ai_boxes = merge_intersecting_boxes(ctx.ai_boxes)
        logger.info(f"Merged to {len(ctx.ai_boxes)} boxes for AI")
        return ctx


class AIInpaintStep(PipelineStep):
    """Step 5: AI inpainting."""
    
    def __init__(self, model: ImageModel):
        super().__init__("ai_inpaint")
        self._model = model
    
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.ai_boxes:
            return ctx
        
        # TODO: Implement AI inpainting loop
        # For now, just log
        logger.info(f"AI inpainting {len(ctx.ai_boxes)} regions")
        
        return ctx


class TextRemovalService:
    """Service for removing text from images."""
    
    def __init__(
        self,
        ocr: OCREngine,
        image_model: ImageModel,
        config: ProcessingConfig,
        cache: Cache | None = None,
        events: EventPublisher | None = None
    ):
        self._ocr = ocr
        self._image_model = image_model
        self._config = config
        self._cache = cache or MemoryCache()
        self._events = events or SimpleEventPublisher()
        self._pipeline = self._build_pipeline()
    
    def _build_pipeline(self) -> list[PipelineStep]:
        """Build processing pipeline."""
        return [
            DetectTextStep(self._ocr, self._cache),
            ApplyWhitelistStep(),
            SmartFillStep(),
            MergeBoxesStep(),
            AIInpaintStep(self._image_model),
        ]
    
    def remove_text(self, image: Image | Path) -> ProcessingResult:
        """Remove text from image.
        
        Args:
            image: Image to process
            
        Returns:
            Processing result
        """
        start_time = time.time()
        
        if isinstance(image, Path):
            image = Image.from_file(image)
        
        self._events.publish(ProcessingEvent(
            stage="start",
            message="Starting text removal",
            image_path=image.source_path
        ))
        
        ctx = PipelineContext(image=image, config=self._config)
        
        try:
            for step in self._pipeline:
                self._events.publish(ProcessingEvent(
                    stage=step.name,
                    message=f"Executing {step.name}"
                ))
                ctx = step.execute(ctx)
            
            elapsed = (time.time() - start_time) * 1000
            
            result = ProcessingResult.success_result(
                image=ctx.image,
                boxes_processed=len(ctx.ai_boxes),
                boxes_smart_filled=len(ctx.smart_filled_boxes),
                processing_time_ms=elapsed
            )
            
            self._events.publish(ProcessingEvent(
                stage="complete",
                message="Processing complete",
                progress=1.0
            ))
            
            return result
            
        except Exception as e:
            logger.exception("Processing failed")
            return ProcessingResult.failure(str(e))
    
    def subscribe_to_events(self, callback: callable) -> None:
        """Subscribe to processing events."""
        self._events.subscribe(callback)
    
    def __enter__(self) -> TextRemovalService:
        """Context manager entry."""
        self._ocr.load()
        self._image_model.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - cleanup resources."""
        self._ocr.unload()
        self._image_model.unload()
        return False  # Don't suppress exceptions
