"""AI models for image generation."""

from typing import Union

from ..config import ModelType
from .base import BaseImageModel
from .flux_model import Flux2KleinModel
from .longcat_model import LongCatModel


# Model type to class mapping
_MODEL_MAP: dict[ModelType, tuple[type[BaseImageModel], dict]] = {
    ModelType.FLUX_2_KLEIN_9B: (Flux2KleinModel, {}),
    ModelType.FLUX_2_KLEIN_4B: (Flux2KleinModel, {}),
    ModelType.FLUX_2_KLEIN_9B_SNDQ: (Flux2KleinModel, {}),
    ModelType.FLUX_2_KLEIN_4B_SNDQ: (Flux2KleinModel, {}),
    ModelType.LONGCAT: (LongCatModel, {}),
    ModelType.LONGCAT_TURBO: (LongCatModel, {}),
    ModelType.LONGCAT_DF11: (LongCatModel, {}),
    ModelType.LONGCAT_TURBO_DF11: (LongCatModel, {}),
}


class ModelFactory:
    """Factory for creating model instances."""
    
    @staticmethod
    def create(
        model_type: ModelType | str,
        device: str = "auto",
        speedup: bool = False
    ) -> BaseImageModel:
        """Create a model instance.
        
        Args:
            model_type: Model type enum or string
            device: Compute device
            speedup: Enable speedup for quantized models
            
        Returns:
            Configured model instance
            
        Raises:
            ValueError: If model type is not supported
        """
        if isinstance(model_type, str):
            model_type = ModelType(model_type)
        
        if model_type not in _MODEL_MAP:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model_class, extra_kwargs = _MODEL_MAP[model_type]
        return model_class(
            model_type=model_type,
            device=device,
            **extra_kwargs
        )


def get_available_models() -> list[str]:
    """Get list of available model names."""
    return [m.value for m in ModelType]


__all__ = [
    'BaseImageModel',
    'Flux2KleinModel',
    'LongCatModel',
    'ModelFactory',
    'get_available_models',
]
