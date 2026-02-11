"""LongCat model implementation."""

import logging
from typing import Optional

import torch
from diffusers import LongCatImageEditPipeline, LongCatImageTransformer2DModel
from PIL import Image
from transformers.initialization import no_init_weights

from ..config import ModelConfig, ModelType, MODEL_CONFIGS
from .base import BaseImageModel

logger = logging.getLogger(__name__)


class LongCatModel(BaseImageModel):
    """LongCat image editing model."""
    
    def __init__(
        self,
        model_type: ModelType = ModelType.LONGCAT,
        device: str = "auto"
    ):
        """Initialize LongCat model.
        
        Args:
            model_type: Which LongCat variant to use
            device: Compute device
        """
        super().__init__(device)
        self.model_type = model_type
        self.config = MODEL_CONFIGS[model_type]
        self._pipe: Optional[LongCatImageEditPipeline] = None
    
    def load(self) -> None:
        """Load the model."""
        if self.is_loaded:
            logger.debug("Model already loaded")
            return
        
        logger.info(f"Loading {self.model_type.value}...")
        
        if self.config.quant == "DF11":
            self._load_df11()
        elif self.config.turbo:
            self._load_turbo()
        else:
            self._load_standard()
        
        self._apply_device()
        logger.info(f"Model loaded on {self.device}")
    
    def _load_standard(self) -> None:
        """Load official LongCat models from Meituan."""
        self._pipe = LongCatImageEditPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.bfloat16
        )
    
    def _load_turbo(self) -> None:
        """Load turbo variant."""
        self._pipe = LongCatImageEditPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.bfloat16
        )
    
    def _load_df11(self) -> None:
        """Load quantized model with DFloat11."""
        try:
            from dfloat11 import DFloat11Model
        except ImportError as e:
            raise ImportError(
                "cupy and dfloat11 are required for quantized LongCat. "
                "Install with: pip install cupy dfloat11"
            ) from e
        
        if self.config.turbo:
            base_model_id = "meituan-longcat/LongCat-Image-Edit-Turbo"
        else:
            base_model_id = "meituan-longcat/LongCat-Image-Edit"
        
        # Load transformer with DFloat11 quantization
        with no_init_weights():
            transformer = LongCatImageTransformer2DModel.from_config(
                LongCatImageTransformer2DModel.load_config(
                    base_model_id,
                    subfolder="transformer"
                ),
                torch_dtype=torch.bfloat16
            ).to(torch.bfloat16)
        
        DFloat11Model.from_pretrained(
            self.config.model_id,
            device="cpu",
            bfloat16_model=transformer,
        )
        
        self._pipe = LongCatImageEditPipeline.from_pretrained(
            self.config.model_id,
            transformer=transformer,
            torch_dtype=torch.bfloat16
        )
        
        # Also quantize text encoder
        DFloat11Model.from_pretrained(
            "mingyi456/Qwen2.5-VL-7B-Instruct-DF11",
            device="cpu",
            bfloat16_model=self._pipe.text_encoder,
        )
    
    def run(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        steps: Optional[int] = None,
        negative_prompt: str = "",
        **kwargs
    ) -> Image.Image:
        """Run inference.
        
        Args:
            prompt: Text prompt
            image: Input image
            steps: Number of inference steps
            negative_prompt: Negative prompt for guidance
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Generated image
        """
        if not self.is_loaded:
            self.load()
        
        steps = self.validate_steps(steps)
        
        if image is None:
            raise ValueError("LongCat requires an input image")
        
        # Prepare dimensions using base class helper
        image, (orig_width, orig_height) = self._prepare_image_dimensions(image)
        
        # Determine guidance scale based on turbo mode
        guidance_scale = 1.0 if self.config.turbo else 4.5
        
        # Run inference
        result = self._pipe(
            image.convert('RGB'),
            prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            num_images_per_prompt=1,
            generator=self._get_generator(seed=43)
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
