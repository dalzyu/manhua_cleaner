"""Tests for the processing pipeline."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from ..config import ModelType
from ..core.processor import BatchProcessor, ProcessingConfig, ProcessingResult
from ..core.image_ops import Contour


class TestProcessingConfig:
    """Test processing configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ProcessingConfig()
        
        assert config.model_type == ModelType.FLUX_2_KLEIN_4B
        assert config.device == "auto"
        assert config.color_correct is True
        assert config.smart_fill is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ProcessingConfig(
            model_type=ModelType.LONGCAT_TURBO,
            steps=8,
            prompt="custom prompt"
        )
        
        assert config.model_type == ModelType.LONGCAT_TURBO
        assert config.steps == 8
        assert config.prompt == "custom prompt"

    def test_smart_fill_expand_not_clamped(self):
        """Smart fill expand can exceed AI expand without clamping."""
        config = ProcessingConfig(expand_pixels=10, smart_fill_expand_pixels=20)
        assert config.smart_fill_expand_pixels == 20


class TestBatchProcessor:
    """Test batch processor."""
    
    def test_initialization(self):
        """Test processor initialization."""
        config = ProcessingConfig()
        processor = BatchProcessor(config)
        
        assert processor.config == config
        assert processor._ocr is None
        assert processor._image_model is None
        assert len(processor._ocr_cache) == 0  # Empty cache
    
    def test_ocr_lazy_loading(self):
        """Test OCR is loaded on demand."""
        config = ProcessingConfig()
        processor = BatchProcessor(config)
        
        # Should start as None
        assert processor._ocr is None
        
        # Access should trigger lazy loading (if paddleocr available)
        # Note: If paddleocr is not installed, OCRError will be raised
        # which wraps the ImportError
    
    def test_ocr_cache(self):
        """Test OCR result caching."""
        config = ProcessingConfig()
        processor = BatchProcessor(config)
        
        # Add mock cache entry
        test_path = Path("/test/image.jpg")
        mock_boxes = [np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]], dtype=np.int32)]
        processor._ocr_cache.put(test_path, mock_boxes)
        
        # Cache should return same boxes
        assert processor._ocr_cache.get(test_path) is mock_boxes


class TestProcessingResult:
    """Test processing result dataclass."""
    
    def test_success_result(self):
        """Test successful result."""
        from PIL import Image
        
        img = Image.new('RGB', (100, 100))
        result = ProcessingResult(
            success=True,
            image=img,
            boxes_processed=5
        )
        
        assert result.success is True
        assert result.image is img
        assert result.boxes_processed == 5
        assert result.error_message is None
    
    def test_failure_result(self):
        """Test failure result."""
        result = ProcessingResult(
            success=False,
            error_message="Something went wrong"
        )
        
        assert result.success is False
        assert result.image is None
        assert result.error_message == "Something went wrong"


class TestModelIntegration:
    """Integration tests with mocked models."""
    
    @patch('manhua_cleaner.core.processor.ModelFactory')
    def test_processor_model_loading(self, mock_factory):
        """Test that processor loads model correctly."""
        mock_model = Mock()
        mock_model.is_loaded = True
        mock_model.default_steps = 4
        mock_model.max_steps = 24
        
        mock_factory.create.return_value = mock_model
        
        config = ProcessingConfig(model_type=ModelType.FLUX_2_KLEIN_4B)
        processor = BatchProcessor(config)
        
        # Access image_model property
        model = processor.image_model
        
        mock_factory.create.assert_called_once_with(
            ModelType.FLUX_2_KLEIN_4B,
            "auto"
        )
        mock_model.load.assert_called_once()
        assert model is mock_model
