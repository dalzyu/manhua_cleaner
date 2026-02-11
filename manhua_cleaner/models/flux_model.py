"""FLUX.2-klein model implementation."""

import logging
from typing import Optional

import torch
from diffusers import Flux2KleinPipeline
from PIL import Image

from ..config import ModelConfig, ModelType, MODEL_CONFIGS
from .base import BaseImageModel

logger = logging.getLogger(__name__)


class Flux2KleinModel(BaseImageModel):
    """FLUX.2-klein image editing model."""
    
    def __init__(
        self,
        model_type: ModelType = ModelType.FLUX_2_KLEIN_4B,
        device: str = "auto",
        speedup: bool = True
    ):
        """Initialize FLUX.2-klein model.
        
        Args:
            model_type: Which FLUX variant to use
            device: Compute device
            speedup: Enable torch.compile for faster inference (quantized only)
        """
        super().__init__(device)
        self.model_type = model_type
        self.config = MODEL_CONFIGS[model_type]
        self.speedup = speedup if self.config.quant is not None else False
        self._pipe: Optional[Flux2KleinPipeline] = None
        
        if speedup and self.config.quant is None:
            logger.warning("Speedup only available for quantized models")
    
    def load(self) -> None:
        """Load the model."""
        if self.is_loaded:
            logger.debug("Model already loaded")
            return
        
        logger.info(f"Loading {self.model_type.value}...")
        
        if self.config.quant and "SDNQ-4bit-dynamic" in self.config.quant:
            self._load_sndq()
        else:
            self._load_full_precision()
        
        self._apply_device()
        logger.info(f"Model loaded on {self.device}")
    
    def _load_full_precision(self) -> None:
        """Load full precision model."""
        self._pipe = Flux2KleinPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.bfloat16
        )
    
    def _load_sndq(self) -> None:
        """Load sndq model with optional speedup."""
        from sdnq import SDNQConfig  # noqa: F401
        from sdnq.common import use_torch_compile as triton_is_available
        from sdnq.loader import apply_sdnq_options_to_model
        
        self._pipe = Flux2KleinPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.bfloat16
        )
        
        # Enable INT8 MatMul for supported GPUs
        if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
            self._pipe.transformer = apply_sdnq_options_to_model(
                self._pipe.transformer, use_quantized_matmul=True
            )
            self._pipe.text_encoder = apply_sdnq_options_to_model(
                self._pipe.text_encoder, use_quantized_matmul=True
            )
            
            if self.speedup:
                logger.info("Enabling torch.compile for faster inference")
                self._pipe.transformer = torch.compile(self._pipe.transformer)
    
    def run(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        steps: Optional[int] = None,
        **kwargs
    ) -> Image.Image:
        """Run inference.
        
        Args:
            prompt: Text prompt
            image: Input image
            steps: Number of inference steps
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Generated image
        """
        if not self.is_loaded:
            self.load()
        
        steps = self.validate_steps(steps)
        
        if image is None:
            raise ValueError("FLUX.2-klein requires an input image")
        
        # Prepare dimensions using base class helper
        image, (orig_width, orig_height) = self._prepare_image_dimensions(image)
        
        # Run inference
        result = self._pipe(
            image=image.convert('RGB'),
            prompt=prompt,
            height=image.height,
            width=image.width,
            guidance_scale=1.0,
            num_inference_steps=steps,
            generator=self._get_generator(seed=0)
        ).images[0]
        
        # Restore original dimensions using base class helper
        return self._restore_image_dimensions(result, (orig_width, orig_height))
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._pipe is not None
    
    @property
    def default_steps(self) -> int:
        """Get default steps."""
        return self.config.default_steps
    
    @property
    def max_steps(self) -> int:
        """Get max steps."""
        return self.config.max_steps
