"""Plugin registry - discovers and loads plugins via entry points."""

from __future__ import annotations

import importlib
import logging
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..application.ports.image_model import ImageModel
    from ..application.ports.ocr_engine import OCREngine

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Registry for discovering and loading plugins.
    
    Uses setuptools entry points for plugin discovery:
    - manhua_cleaner.ocr: OCR engine implementations
    - manhua_cleaner.models: Image model implementations
    
    Third-party packages can register plugins:
    
    [project.entry-points."manhua_cleaner.ocr"]
    my_ocr = "my_package:MyOCREngine"
    """
    
    OCR_GROUP = "manhua_cleaner.ocr"
    MODEL_GROUP = "manhua_cleaner.models"
    
    @classmethod
    @lru_cache(maxsize=1)
    def discover_ocr_engines(cls) -> dict[str, type]:
        """Discover all available OCR engines.
        
        Returns:
            Dict mapping engine names to classes
        """
        engines = {}
        
        try:
            from importlib.metadata import entry_points
            
            eps = entry_points()
            if hasattr(eps, 'select'):
                # Python 3.10+
                group = eps.select(group=cls.OCR_GROUP)
            else:
                # Python 3.9
                group = eps.get(cls.OCR_GROUP, [])
            
            for ep in group:
                try:
                    engine_class = ep.load()
                    engines[ep.name] = engine_class
                    logger.debug(f"Discovered OCR engine: {ep.name}")
                except Exception as e:
                    logger.warning(f"Failed to load OCR engine {ep.name}: {e}")
                    
        except ImportError:
            logger.debug("importlib.metadata not available")
        
        # Always include built-in engines
        from ..adapters.ocr.paddle_adapter import PaddleOCRAdapter
        engines["paddleocr"] = PaddleOCRAdapter
        
        try:
            from ..adapters.ocr.easyocr_adapter import EasyOCRAdapter
            engines["easyocr"] = EasyOCRAdapter
        except ImportError:
            pass
        
        return engines
    
    @classmethod
    @lru_cache(maxsize=1)
    def discover_models(cls) -> dict[str, type]:
        """Discover all available image models.
        
        Returns:
            Dict mapping model names to classes
        """
        models = {}
        
        try:
            from importlib.metadata import entry_points
            
            eps = entry_points()
            if hasattr(eps, 'select'):
                group = eps.select(group=cls.MODEL_GROUP)
            else:
                group = eps.get(cls.MODEL_GROUP, [])
            
            for ep in group:
                try:
                    model_class = ep.load()
                    models[ep.name] = model_class
                    logger.debug(f"Discovered model: {ep.name}")
                except Exception as e:
                    logger.warning(f"Failed to load model {ep.name}: {e}")
                    
        except ImportError:
            logger.debug("importlib.metadata not available")
        
        # Always include built-in models
        from ..adapters.models.flux_adapter import FluxAdapter
        from ..adapters.models.longcat_adapter import LongCatAdapter
        
        models["flux"] = FluxAdapter
        models["longcat"] = LongCatAdapter
        
        return models
    
    @classmethod
    def create_ocr_engine(
        cls,
        name: str,
        **kwargs
    ) -> "OCREngine":
        """Create OCR engine instance by name.
        
        Args:
            name: Engine name (e.g., 'paddleocr', 'easyocr')
            **kwargs: Constructor arguments
            
        Returns:
            OCREngine instance
            
        Raises:
            ValueError: If engine not found
        """
        engines = cls.discover_ocr_engines()
        
        if name not in engines:
            available = ", ".join(engines.keys())
            raise ValueError(f"Unknown OCR engine: {name}. Available: {available}")
        
        return engines[name](**kwargs)
    
    @classmethod
    def create_model(
        cls,
        name: str,
        **kwargs
    ) -> "ImageModel":
        """Create image model instance by name.
        
        Args:
            name: Model name (e.g., 'flux', 'longcat')
            **kwargs: Constructor arguments
            
        Returns:
            ImageModel instance
            
        Raises:
            ValueError: If model not found
        """
        models = cls.discover_models()
        
        if name not in models:
            available = ", ".join(models.keys())
            raise ValueError(f"Unknown model: {name}. Available: {available}")
        
        return models[name](**kwargs)
    
    @classmethod
    def list_available_ocr(cls) -> list[str]:
        """List available OCR engine names."""
        return list(cls.discover_ocr_engines().keys())
    
    @classmethod
    def list_available_models(cls) -> list[str]:
        """List available model names."""
        return list(cls.discover_models().keys())
