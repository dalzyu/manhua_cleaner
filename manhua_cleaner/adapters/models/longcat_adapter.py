"""LongCat model adapter - implements ImageModel port."""

from __future__ import annotations

import logging

from ...application.ports.image_model import ImageModel, InpaintResult
from ...domain.entities.image import Image
from ...domain.value_objects.config import ModelType
from ...domain.value_objects.geometry import Mask

logger = logging.getLogger(__name__)


class LongCatAdapter(ImageModel):
    """Adapter for LongCat models."""
    
    def __init__(
        self,
        model_type: ModelType = ModelType.LONGCAT,
        device: str = "auto"
    ):
        self._model_type = model_type
        self._device = device
        self._pipe = None
    
    @property
    def name(self) -> str:
        return self._model_type.value
    
    @property
    def is_available(self) -> bool:
        """Check if required dependencies are installed."""
        try:
            import torch
            from diffusers import LongCatImageEditPipeline
            return True
        except ImportError:
            return False
    
    @property
    def default_steps(self) -> int:
        if "Turbo" in self._model_type.value:
            return 8
        return 50
    
    @property
    def max_steps(self) -> int:
        return 100
    
    def load(self) -> None:
        """Load LongCat model."""
        if self._pipe is not None:
            return
        
        try:
            import torch
            from diffusers import LongCatImageEditPipeline
        except ImportError as e:
            raise RuntimeError(
                "LongCat requires torch and diffusers. Install with: pip install torch diffusers"
            ) from e
        
        model_id = self._get_model_id()
        logger.info(f"Loading {model_id}...")
        
        self._pipe = LongCatImageEditPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16
        )
        
        self._apply_device()
        logger.info("LongCat model loaded")
    
    def unload(self) -> None:
        """Unload model."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            import gc
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("LongCat model unloaded")
    
    def inpaint(
        self,
        image: Image,
        mask: Mask,
        prompt: str,
        steps: int | None = None
    ) -> InpaintResult:
        """Inpaint masked region."""
        if self._pipe is None:
            self.load()
        
        steps = steps or self.default_steps
        result_image = self._run_inference(image, prompt, steps)
        
        return InpaintResult(
            image=result_image,
            steps_used=steps
        )
    
    def edit(
        self,
        image: Image,
        prompt: str,
        steps: int | None = None
    ) -> InpaintResult:
        """Edit entire image."""
        if self._pipe is None:
            self.load()
        
        steps = steps or self.default_steps
        result_image = self._run_inference(image, prompt, steps)
        
        return InpaintResult(
            image=result_image,
            steps_used=steps
        )
    
    def _run_inference(self, image: Image, prompt: str, steps: int) -> Image:
        """Run model inference."""
        import torch
        
        pil_image = image._data.convert('RGB')
        
        guidance_scale = 1.0 if "Turbo" in self._model_type.value else 4.5
        
        result = self._pipe(
            pil_image,
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=torch.Generator().manual_seed(43)
        ).images[0]
        
        return Image(_data=result, source_path=image.source_path)
    
    def _get_model_id(self) -> str:
        """Get HuggingFace model ID."""
        model_ids = {
            ModelType.LONGCAT: "meituan-longcat/LongCat-Image-Edit",
            ModelType.LONGCAT_TURBO: "meituan-longcat/LongCat-Image-Edit-Turbo",
            ModelType.LONGCAT_DF11: "mingyi456/LongCat-Image-Edit-DF11",
            ModelType.LONGCAT_TURBO_DF11: "mingyi456/LongCat-Image-Edit-Turbo-DF11",
        }
        return model_ids.get(self._model_type, "meituan-longcat/LongCat-Image-Edit")
    
    def _apply_device(self) -> None:
        """Apply device settings."""
        if self._pipe is None:
            return
        
        if self._device == "cuda":
            try:
                self._pipe.enable_model_cpu_offload()
                return
            except Exception as e:
                logger.warning(f"CPU offload failed: {e}")
        
        try:
            import torch
            self._pipe.to(self._device)
        except Exception as e:
            logger.warning(f"Failed to move to device {self._device}: {e}")
