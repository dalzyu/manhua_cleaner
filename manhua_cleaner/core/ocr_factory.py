"""Factory for creating OCR model instances."""

import logging
from enum import Enum
from typing import Type

from ..exceptions import ConfigurationError
from .ocr_base import BaseOCR

logger = logging.getLogger(__name__)


class OCRModelType(str, Enum):
    """Supported OCR model types."""
    PADDLEOCR = "paddleocr"
    EASYOCR = "easyocr"


# Lazy-loaded implementation registry
_IMPLEMENTATIONS: dict[OCRModelType, Type[BaseOCR]] | None = None


def _load_implementations() -> dict[OCRModelType, Type[BaseOCR]]:
    """Load and return available OCR implementations."""
    global _IMPLEMENTATIONS
    if _IMPLEMENTATIONS is not None:
        return _IMPLEMENTATIONS
    
    implementations: dict[OCRModelType, Type[BaseOCR]] = {}
    
    # PaddleOCR - always available (required dependency)
    from .ocr_paddle import PaddleOCRTextSpotter
    implementations[OCRModelType.PADDLEOCR] = PaddleOCRTextSpotter
    
    # EasyOCR - optional (lazy import)
    try:
        from .ocr_easyocr import EasyOCRTextSpotter
        implementations[OCRModelType.EASYOCR] = EasyOCRTextSpotter
        logger.debug("EasyOCR available")
    except ImportError:
        logger.debug("EasyOCR not installed (optional)")
    
    _IMPLEMENTATIONS = implementations
    return implementations


class OCRFactory:
    """Factory for creating OCR model instances.
    
    This factory creates the appropriate OCR implementation based on
    the requested model type. It handles lazy imports to avoid
    loading unnecessary dependencies.
    
    Example:
        >>> from manhua_cleaner.core.ocr_factory import OCRFactory, OCRModelType
        >>> ocr = OCRFactory.create(OCRModelType.EASYOCR, device='cuda')
        >>> ocr.load()
        >>> regions = ocr.detect('image.jpg')
    """
    
    @classmethod
    def create(
        cls,
        model_type: OCRModelType | str,
        device: str = "auto",
        **kwargs
    ) -> BaseOCR:
        """Create an OCR model instance.
        
        Args:
            model_type: OCR model type (enum or string)
            device: Compute device ('cuda', 'cpu', 'auto')
            **kwargs: Additional model-specific arguments
            
        Returns:
            Configured OCR instance
            
        Raises:
            ConfigurationError: If model type is not supported
        """
        # Convert string to enum if needed
        if isinstance(model_type, str):
            try:
                model_type = OCRModelType(model_type)
            except ValueError as e:
                available = [m.value for m in OCRModelType]
                raise ConfigurationError(
                    f"Unknown OCR model: '{model_type}'. "
                    f"Available: {', '.join(available)}",
                    config_key="ocr_model"
                ) from e
        
        implementations = _load_implementations()
        impl_class = implementations.get(model_type)
        
        if impl_class is None:
            available = [m.value for m in implementations.keys()]
            raise ConfigurationError(
                f"OCR model '{model_type.value}' not available. "
                f"Available: {', '.join(available)}",
                config_key="ocr_model"
            )
        
        logger.debug(f"Creating OCR model: {model_type.value}")
        return impl_class(device=device, **kwargs)
    
    @classmethod
    def list_available(cls) -> list[OCRModelType]:
        """List all available OCR model types.
        
        Returns:
            List of available model types
        """
        return list(_load_implementations().keys())
    
    @classmethod
    def is_available(cls, model_type: OCRModelType | str) -> bool:
        """Check if an OCR model type is available.
        
        Args:
            model_type: Model type to check
            
        Returns:
            True if available, False otherwise
        """
        if isinstance(model_type, str):
            try:
                model_type = OCRModelType(model_type)
            except ValueError:
                return False
        
        return model_type in _load_implementations()


def get_ocr_model(model_type: OCRModelType | str, **kwargs) -> BaseOCR:
    """Convenience function to create an OCR model.
    
    Args:
        model_type: OCR model type
        **kwargs: Additional arguments passed to the model
        
    Returns:
        Configured OCR instance
    """
    return OCRFactory.create(model_type, **kwargs)
