"""Unit tests for configuration."""

import pytest
from manhua_cleaner.domain.value_objects.config import (
    ProcessingConfig, ModelType, Backend, WHITELIST_PRESETS
)


class TestProcessingConfig:
    """Tests for ProcessingConfig."""
    
    def test_default_values(self):
        config = ProcessingConfig()
        assert config.model_type == ModelType.FLUX_2_KLEIN_4B
        assert config.device == "auto"
        assert config.expand_pixels == 25
        assert config.smart_fill is True
    
    def test_custom_values(self):
        config = ProcessingConfig(
            model_type=ModelType.LONGCAT_TURBO,
            expand_pixels=50,
            steps=8
        )
        assert config.model_type == ModelType.LONGCAT_TURBO
        assert config.expand_pixels == 50
        assert config.steps == 8
    
    def test_whitelist_patterns_from_preset(self):
        config = ProcessingConfig(
            whitelist_enabled=True,
            whitelist_preset="hearts"
        )
        assert len(config.whitelist_patterns) == 1
        # Should have the hearts pattern
        assert "♡♥❤" in config.whitelist_patterns[0]
    
    def test_validation_expand_pixels_range(self):
        # Should validate range
        with pytest.raises(Exception):  # ValidationError or ValueError
            ProcessingConfig(expand_pixels=-1)
    
    def test_validation_ocr_workers_range(self):
        with pytest.raises(Exception):
            ProcessingConfig(ocr_workers=0)
        
        with pytest.raises(Exception):
            ProcessingConfig(ocr_workers=10)


class TestModelType:
    """Tests for ModelType enum."""
    
    def test_string_values(self):
        assert ModelType.FLUX_2_KLEIN_4B.value == "FLUX.2-klein-4B"
        assert ModelType.LONGCAT.value == "LongCat-Image-Edit"
    
    def test_from_string(self):
        model = ModelType("FLUX.2-klein-4B")
        assert model == ModelType.FLUX_2_KLEIN_4B


class TestWhitelistPresets:
    """Tests for whitelist presets."""
    
    def test_preset_exists(self):
        assert "none" in WHITELIST_PRESETS
        assert "hearts" in WHITELIST_PRESETS
        assert "sfx_only" in WHITELIST_PRESETS
    
    def test_none_preset_empty(self):
        assert WHITELIST_PRESETS["none"] == []
    
    def test_hearts_preset(self):
        hearts = WHITELIST_PRESETS["hearts"]
        assert len(hearts) == 1
        assert "♡♥❤" in hearts[0]
