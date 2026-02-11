"""Tests for configuration module."""

import pytest

from ..config import (
    ModelType,
    Backend,
    ModelConfig,
    MODEL_CONFIGS,
    IMAGE_CONFIG,
)


class TestModelType:
    """Test ModelType enum."""
    
    def test_model_type_values(self):
        """Test that all model types have correct values."""
        assert ModelType.FLUX_2_KLEIN_4B.value == "FLUX.2-klein-4B"
        assert ModelType.FLUX_2_KLEIN_9B.value == "FLUX.2-klein-9B"
        assert ModelType.LONGCAT_TURBO.value == "LongCat-Image-Edit-Turbo"
    
    def test_model_type_from_string(self):
        """Test creating ModelType from string."""
        model = ModelType("FLUX.2-klein-4B")
        assert model == ModelType.FLUX_2_KLEIN_4B
    
    def test_invalid_model_type(self):
        """Test that invalid model type raises error."""
        with pytest.raises(ValueError):
            ModelType("invalid-model")


class TestBackend:
    """Test Backend enum."""
    
    def test_backend_values(self):
        """Test backend enum values."""
        assert Backend.AUTO.value == "auto"
        assert Backend.CUDA.value == "cuda"
        assert Backend.CPU.value == "cpu"


class TestModelConfigs:
    """Test model configurations."""
    
    def test_all_models_have_config(self):
        """Test that every ModelType has a config."""
        for model_type in ModelType:
            assert model_type in MODEL_CONFIGS
    
    def test_flux_config(self):
        """Test FLUX model configuration."""
        config = MODEL_CONFIGS[ModelType.FLUX_2_KLEIN_4B]
        assert config.model_id.startswith("black-forest-labs/")
        assert config.default_steps == 4
        assert config.max_steps == 24
        assert not config.quantized
    
    def test_longcat_config(self):
        """Test LongCat model configuration."""
        config = MODEL_CONFIGS[ModelType.LONGCAT]
        assert config.model_id.startswith("meituan-longcat/")
        assert config.default_steps == 50
        assert config.max_steps == 100
    
    def test_turbo_config(self):
        """Test turbo variant has lower default steps."""
        turbo = MODEL_CONFIGS[ModelType.LONGCAT_TURBO]
        normal = MODEL_CONFIGS[ModelType.LONGCAT]
        assert turbo.default_steps < normal.default_steps
        assert turbo.turbo


class TestImageConfig:
    """Test image processing configuration."""
    
    def test_min_size(self):
        """Test minimum size is reasonable."""
        assert IMAGE_CONFIG.min_size > 0
        assert IMAGE_CONFIG.min_size <= 128
    
    def test_max_size(self):
        """Test maximum size is reasonable."""
        assert IMAGE_CONFIG.max_size >= 1024
    
    def test_padding_multiple(self):
        """Test padding multiple is power of 2."""
        assert IMAGE_CONFIG.padding_multiple in [8, 16, 32, 64]
