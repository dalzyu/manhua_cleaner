"""Image Model port - interface for AI image processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from ...domain.entities.image import Image
from ...domain.value_objects.geometry import Mask


@dataclass(frozen=True, slots=True)
class InpaintResult:
    """Result of inpainting operation."""
    image: Image
    processing_time_ms: float = 0.0
    steps_used: int = 0


@runtime_checkable
class ImageModel(Protocol):
    """Port for AI image processing models.
    
    Implementations: FLUX, LongCat, Stable Diffusion, etc.
    """
    
    @property
    def name(self) -> str:
        """Model name."""
        ...
    
    @property
    def is_available(self) -> bool:
        """Check if model dependencies are installed."""
        ...
    
    @property
    def default_steps(self) -> int:
        """Default inference steps."""
        ...
    
    @property
    def max_steps(self) -> int:
        """Maximum allowed steps."""
        ...
    
    def load(self) -> None:
        """Load model into memory."""
        ...
    
    def unload(self) -> None:
        """Unload model and free memory."""
        ...
    
    def inpaint(
        self,
        image: Image,
        mask: Mask,
        prompt: str,
        steps: int | None = None
    ) -> InpaintResult:
        """Inpaint masked region.
        
        Args:
            image: Source image
            mask: Binary mask of region to inpaint
            prompt: Text prompt for generation
            steps: Number of inference steps (None for default)
            
        Returns:
            Inpainted image result
        """
        ...
    
    def edit(
        self,
        image: Image,
        prompt: str,
        steps: int | None = None
    ) -> InpaintResult:
        """Edit entire image.
        
        Args:
            image: Source image
            prompt: Text prompt for editing
            steps: Number of inference steps
            
        Returns:
            Edited image result
        """
        ...
