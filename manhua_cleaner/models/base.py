"""Abstract base class for AI image generation models."""

import gc
import logging
import math
from abc import ABC, abstractmethod
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class BaseImageModel(ABC):
    """Abstract base class for image inpainting models."""
    
    # Padding multiple for model input dimensions (must be multiple of 16 for most models)
    PADDING_MULTIPLE = 16
    
    def __init__(self, device: str = "auto"):
        """Initialize the model.
        
        Args:
            device: Compute device ('cuda', 'mps', 'cpu', or 'auto')
        """
        self.device = self._resolve_device(device)
        self._pipe = None
    
    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve device string to actual device.
        
        Args:
            device: Device string ('cuda', 'mps', 'cpu', or 'auto')
            
        Returns:
            Resolved device string
        """
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        if device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS requested but not available, falling back to CPU")
            return "cpu"
        
        return device
    
    def _prepare_image_dimensions(self, image: Image.Image) -> tuple[Image.Image, tuple[int, int]]:
        """Prepare image dimensions for model input.
        
        Ensures dimensions are multiples of PADDING_MULTIPLE (usually 16).
        
        Args:
            image: Input PIL image
            
        Returns:
            Tuple of (resized_image, (orig_width, orig_height))
        """
        orig_width, orig_height = image.size
        
        # Ensure dimensions are multiples of PADDING_MULTIPLE
        upscaled_width = math.ceil(orig_width / self.PADDING_MULTIPLE) * self.PADDING_MULTIPLE
        upscaled_height = math.ceil(orig_height / self.PADDING_MULTIPLE) * self.PADDING_MULTIPLE
        
        if (upscaled_width, upscaled_height) != (orig_width, orig_height):
            image = image.resize(
                (upscaled_width, upscaled_height),
                Image.Resampling.LANCZOS
            )
            logger.debug(f"Resized image from ({orig_width}, {orig_height}) to "
                        f"({upscaled_width}, {upscaled_height})")
        
        return image, (orig_width, orig_height)
    
    def _restore_image_dimensions(self, image: Image.Image, orig_size: tuple[int, int]) -> Image.Image:
        """Restore image to original dimensions after processing.
        
        Args:
            image: Processed PIL image
            orig_size: Original (width, height) tuple
            
        Returns:
            Resized image
        """
        if image.size != orig_size:
            image = image.resize(orig_size, Image.Resampling.LANCZOS)
            logger.debug(f"Restored image to original size: {orig_size}")
        return image
    
    def _cleanup_gpu_memory(self) -> None:
        """Clean up GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _apply_device(self) -> None:
        """Apply the configured device to the pipeline if possible."""
        if self._pipe is None:
            return
        if self.device == "cuda":
            # Prefer CPU offload for CUDA to reduce VRAM pressure
            try:
                self._pipe.enable_model_cpu_offload()
                return
            except Exception as e:
                logger.warning(f"CPU offload failed, falling back to .to('cuda'): {e}")
        try:
            self._pipe.to(self.device)
        except Exception as e:
            logger.warning(f"Failed to move model to {self.device}: {e}")

    def _get_generator(self, seed: Optional[int] = None) -> torch.Generator:
        """Create a torch.Generator on the configured device with a seed."""
        try:
            gen = torch.Generator(device=self.device)
        except Exception:
            gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        return gen
    
    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""
        pass
    
    def unload(self) -> None:
        """Unload the model and free memory.
        
        This method can be overridden by subclasses, but the default
        implementation handles the common case of deleting the pipeline
        and cleaning up GPU memory.
        """
        if self._pipe is not None:
            logger.info(f"Unloading {self.__class__.__name__}...")
            del self._pipe
            self._pipe = None
            self._cleanup_gpu_memory()
            logger.info(f"{self.__class__.__name__} unloaded")
    
    @abstractmethod
    def run(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        steps: Optional[int] = None,
        **kwargs
    ) -> Image.Image:
        """Run inference on an image.
        
        Args:
            prompt: Text prompt for the model
            image: Input image (if None, must provide dimensions)
            steps: Number of inference steps (None for model default)
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated image
        """
        pass
    
    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        pass
    
    @property
    @abstractmethod
    def default_steps(self) -> int:
        """Get default number of inference steps."""
        pass
    
    @property
    @abstractmethod
    def max_steps(self) -> int:
        """Get maximum allowed inference steps."""
        pass
    
    def validate_steps(self, steps: Optional[int]) -> int:
        """Validate and return steps count.
        
        Args:
            steps: Requested steps (None for default)
            
        Returns:
            Validated steps count
            
        Raises:
            ValueError: If steps is out of valid range
        """
        if steps is None:
            return self.default_steps
        
        if steps < 1:
            raise ValueError(f"Steps must be at least 1, got {steps}")
        if steps > self.max_steps:
            raise ValueError(
                f"Steps cannot exceed {self.max_steps} for this model, got {steps}"
            )
        return steps
