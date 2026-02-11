"""Configuration value objects with validation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Try to use Pydantic, fall back to dataclasses
try:
    from pydantic import BaseModel, Field, field_validator, model_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False


class ModelType(str, Enum):
    """Supported AI model types."""
    FLUX_2_KLEIN_9B = "FLUX.2-klein-9B"
    FLUX_2_KLEIN_4B = "FLUX.2-klein-4B"
    FLUX_2_KLEIN_9B_SNDQ = "FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32"
    FLUX_2_KLEIN_4B_SNDQ = "FLUX.2-klein-4B-SDNQ-4bit-dynamic"
    LONGCAT = "LongCat-Image-Edit"
    LONGCAT_TURBO = "LongCat-Image-Edit-Turbo"
    LONGCAT_DF11 = "LongCat-Image-Edit-DF11"
    LONGCAT_TURBO_DF11 = "LongCat-Image-Edit-Turbo-DF11"


class Backend(str, Enum):
    """Compute backend options."""
    AUTO = "auto"
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


class OCRModelType(str, Enum):
    """Supported OCR model types."""
    PADDLEOCR = "paddleocr"
    EASYOCR = "easyocr"


# Whitelist presets for character filtering
WHITELIST_PRESETS: dict[str, list[str]] = {
    "none": [],
    "sfx_only": [
        r"^[\s!?.…〜・\-*/\\+=♡♥❤]*$",
    ],
    "punctuation": [
        r"^[\s!?…\.。．!‼?⁇~～〜\-ー―♡♥❤]*$",
    ],
    "hearts": [
        r"^[\s♡♥❤💕💗💓💞💖💘]*$",
    ],
    "symbols": [
        r"^[\s♪♫♬♩⚡★☆✦✧]*$",
    ],
    "japanese_sfx": [
        r"^[\sドガバキポンビシツズゾガギグゲゴザジズゼゾダヂヅデド!?！？…〜・\-*ー―]*$",
    ],
}


if PYDANTIC_AVAILABLE:
    class ProcessingConfig(BaseModel):
        """Processing configuration with validation."""
        
        model_config = {"validate_assignment": False}
        
        # Model settings
        model_type: ModelType = ModelType.FLUX_2_KLEIN_4B
        device: str = "auto"
        steps: int | None = Field(default=None, ge=1, le=100)
        
        # Box expansion
        expand_pixels: int = Field(default=25, ge=0, le=500)
        
        # Processing options
        color_correct: bool = True
        edge_blend: bool = True
        smart_fill: bool = True
        smart_fill_expand_pixels: int = Field(default=5, ge=0, le=100)
        smart_fill_threshold: float = 10.0
        
        # GPU memory
        gpu_memory_threshold_gb: float = 6.0
        gpu_cleanup_interval: int = 5
        
        # OCR settings
        ocr_model: str = "paddleocr"
        ocr_pipeline_version: str = "v1.5"
        ocr_use_tensorrt: bool = True
        ocr_precision: str = "fp16"
        ocr_workers: int = Field(default=1, ge=1, le=8)
        
        # Prompts
        prompt: str = "remove all text"
        extra_pass_prompt: str | None = None
        
        # Whitelist settings
        whitelist_enabled: bool = False
        whitelist_preset: str = "none"
        whitelist_patterns: list[str] = Field(default_factory=list)
        whitelist_group_distance: int = Field(default=50, ge=10, le=200)
        
        # Upscaling
        extra_pass_upscale: bool = False
        extra_pass_upscale_factor: float = Field(default=2.0, ge=1.0, le=8.0)
        extra_pass_upscale_method: str = "lanczos"
        
        @field_validator('smart_fill_expand_pixels')
        @classmethod
        def validate_smart_fill(cls, v: int, info: Any) -> int:
            """Warn if smart fill expansion is larger than AI expansion."""
            expand_pixels = info.data.get('expand_pixels', 25)
            if v > expand_pixels:
                warnings.warn(
                    f"smart_fill_expand_pixels ({v}) > expand_pixels ({expand_pixels})"
                )
            return v
        
        @model_validator(mode='after')
        def init_whitelist_patterns(self) -> ProcessingConfig:
            """Initialize whitelist patterns from preset."""
            if not self.whitelist_patterns:
                preset = WHITELIST_PRESETS.get(self.whitelist_preset, [])
                self.whitelist_patterns = preset.copy()
            else:
                # Combine preset with custom
                preset = WHITELIST_PRESETS.get(self.whitelist_preset, [])
                existing = set(self.whitelist_patterns)
                for p in preset:
                    if p not in existing:
                        self.whitelist_patterns.append(p)
            return self

else:
    @dataclass
    class ProcessingConfig:
        """Processing configuration (dataclass fallback)."""
        model_type: ModelType = ModelType.FLUX_2_KLEIN_4B
        device: str = "auto"
        steps: int | None = None
        expand_pixels: int = 25
        color_correct: bool = True
        edge_blend: bool = True
        smart_fill: bool = True
        smart_fill_expand_pixels: int = 5
        smart_fill_threshold: float = 10.0
        gpu_memory_threshold_gb: float = 6.0
        gpu_cleanup_interval: int = 5
        ocr_model: str = "paddleocr"
        ocr_pipeline_version: str = "v1.5"
        ocr_use_tensorrt: bool = True
        ocr_precision: str = "fp16"
        ocr_workers: int = 1
        prompt: str = "remove all text"
        extra_pass_prompt: str | None = None
        whitelist_enabled: bool = False
        whitelist_preset: str = "none"
        whitelist_patterns: list[str] = field(default_factory=list)
        whitelist_group_distance: int = 50
        extra_pass_upscale: bool = False
        extra_pass_upscale_factor: float = 2.0
        extra_pass_upscale_method: str = "lanczos"
        
        def __post_init__(self) -> None:
            if self.expand_pixels < 0:
                raise ValueError("expand_pixels must be >= 0")
            if self.smart_fill_expand_pixels > self.expand_pixels:
                warnings.warn("smart_fill_expand_pixels > expand_pixels")
            
            if not self.whitelist_patterns:
                self.whitelist_patterns = WHITELIST_PRESETS.get(
                    self.whitelist_preset, []
                ).copy()
            else:
                preset = WHITELIST_PRESETS.get(self.whitelist_preset, [])
                existing = set(self.whitelist_patterns)
                for p in preset:
                    if p not in existing:
                        self.whitelist_patterns.append(p)


# Backward compatibility
__all__ = [
    'ModelType',
    'Backend',
    'OCRModelType',
    'ProcessingConfig',
    'WHITELIST_PRESETS',
]
