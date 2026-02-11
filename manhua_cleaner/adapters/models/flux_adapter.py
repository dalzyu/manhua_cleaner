"""FLUX model adapter - implements ImageModel port."""

from __future__ import annotations

import logging
from pathlib import Path

from ...application.ports.image_model import ImageModel, InpaintResult
from ...domain.entities.image import Image
from ...domain.value_objects.config import ModelType
from ...domain.value_objects.geometry import Mask

logger = logging.getLogger(__name__)


class FluxAdapter(ImageModel):
    """Adapter for FLUX.2-klein models."""
    
    def __init__(
        self,
        model_type: ModelType = ModelType.FLUX_2_KLEIN_4B,
        device: str = "auto",
        speedup: bool = False
    ):
        self._model_type = model_type
        self._device = device
        self._speedup = speedup
        self._pipe = None
    
    @property
    def name(self) -> str:
        return self._model_type.value
    
    @property
    def is_available(self) -> bool:
        """Check if required dependencies are installed."""
        try:
            import torch
            from diffusers import Flux2KleinPipeline
            return True
        except ImportError:
            return False
    
    @property
    def default_steps(self) -> int:
        return 4
    
    @property
    def max_steps(self) -> int:
        return 24
    
    def load(self) -> None:
        """Load FLUX model."""
        if self._pipe is not None:
            return
        
        try:
            import torch
            from diffusers import Flux2KleinPipeline
        except ImportError as e:
            raise RuntimeError(
                "FLUX requires torch and diffusers. Install with: pip install torch diffusers"
            ) from e
        
        model_id = self._get_model_id()
        logger.info(f"Loading {model_id}...")
        
        self._pipe = Flux2KleinPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16
        )
        
        self._apply_device()
        logger.info("FLUX model loaded")
    
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
            logger.info("FLUX model unloaded")
    
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
        
        # TODO: Implement actual inpainting with mask
        # For now, use edit mode on cropped region
        logger.debug(f"Inpainting with prompt: {prompt}")
        
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
        
        # Prepare image
        pil_image = image._data.convert('RGB')
        
        # Run inference
        result = self._pipe(
            image=pil_image,
            prompt=prompt,
            height=pil_image.height,
            width=pil_image.width,
            guidance_scale=1.0,
            num_inference_steps=steps,
            generator=torch.Generator(device=self._device).manual_seed(0)
        ).images[0]
        
        return Image(_data=result, source_path=image.source_path)
    
    def _get_model_id(self) -> str:
        """Get HuggingFace model ID."""
        model_ids = {
            ModelType.FLUX_2_KLEIN_9B: "black-forest-labs/FLUX.2-klein-9B",
            ModelType.FLUX_2_KLEIN_4B: "black-forest-labs/FLUX.2-klein-4B",
            ModelType.FLUX_2_KLEIN_9B_SNDQ: "Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32",
            ModelType.FLUX_2_KLEIN_4B_SNDQ: "Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic",
        }
        return model_ids.get(self._model_type, "black-forest-labs/FLUX.2-klein-4B")
    
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
