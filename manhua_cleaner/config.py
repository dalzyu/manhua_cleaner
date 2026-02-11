"""Configuration and constants for the Manhua Cleaner project."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# Whitelist presets for character filtering
# Patterns use [\s...]*$ to allow text containing ONLY these characters (plus whitespace)
# This ensures text with any letters/words is NOT whitelisted
WHITELIST_PRESETS: dict[str, list[str]] = {
    "none": [],
    "sfx_only": [
        # Matches text with only SFX symbols and whitespace (no letters/numbers from ANY language)
        r"^[\s!?.…〜・\-*/\\+=♡♥❤]*$",
    ],
    "punctuation": [
        # Matches text with only punctuation and whitespace (no letters/numbers from ANY language)
        r"^[\s!?…\.。．!‼?⁇~～〜\-ー―♡♥❤]*$",
    ],
    "hearts": [
        # Matches text with only hearts and whitespace (no letters/numbers)
        r"^[\s♡♥❤💕💗💓💞💖💘]*$",
    ],
    "symbols": [
        # Matches text with only symbols and whitespace (no letters/numbers)
        r"^[\s♪♫♬♩⚡★☆✦✧]*$",
    ],
    "japanese_sfx": [
        # Matches text with only Japanese SFX chars and punctuation
        r"^[\sドガバキポンビシツズゾガギグゲゴザジズゼゾダヂヅデド!?！？…〜・\-*ー―]*$",
    ],
}


@dataclass
class WhitelistConfig:
    """Configuration for character whitelist filtering."""
    
    enabled: bool = False
    preset: str = "none"  # Key from WHITELIST_PRESETS
    custom_patterns: list[str] = field(default_factory=list)
    group_distance: int = 50  # Pixels for spatial grouping
    
    @property
    def patterns(self) -> list[str]:
        """Get active patterns (preset + custom)."""
        preset_patterns = WHITELIST_PRESETS.get(self.preset, [])
        return preset_patterns + self.custom_patterns


class OCRModelType(str, Enum):
    """Supported OCR model types."""
    PADDLEOCR = "paddleocr"
    EASYOCR = "easyocr"


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


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for an AI model."""
    model_id: str
    default_steps: int
    max_steps: int
    min_steps: int
    requires_auth: bool = False
    quant: Optional[str] = None
    turbo: bool = False
    
    @property
    def quantized(self) -> bool:
        """Check if this model uses quantization."""
        return self.quant is not None


# Model configurations
MODEL_CONFIGS: dict[ModelType, ModelConfig] = {
    ModelType.FLUX_2_KLEIN_9B: ModelConfig(
        model_id="black-forest-labs/FLUX.2-klein-9B",
        default_steps=4,
        max_steps=24,
        min_steps=1,
        requires_auth=True
    ),
    ModelType.FLUX_2_KLEIN_4B: ModelConfig(
        model_id="black-forest-labs/FLUX.2-klein-4B",
        default_steps=4,
        max_steps=24,
        min_steps=1,
    ),
    ModelType.FLUX_2_KLEIN_9B_SNDQ: ModelConfig(
        model_id="Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32",
        default_steps=4,
        max_steps=24,
        min_steps=1,
        quant="SDNQ-4bit-dynamic-svd-r32",
    ),
    ModelType.FLUX_2_KLEIN_4B_SNDQ: ModelConfig(
        model_id="Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic",
        default_steps=4,
        max_steps=24,
        min_steps=1,
        quant="SDNQ-4bit-dynamic",
    ),
    ModelType.LONGCAT: ModelConfig(
        model_id="meituan-longcat/LongCat-Image-Edit",
        default_steps=50,
        max_steps=100,
        min_steps=1,
    ),
    ModelType.LONGCAT_TURBO: ModelConfig(
        model_id="meituan-longcat/LongCat-Image-Edit-Turbo",
        default_steps=8,
        max_steps=50,
        min_steps=1,
        turbo=True,
    ),
    ModelType.LONGCAT_DF11: ModelConfig(
        model_id="mingyi456/LongCat-Image-Edit-DF11",
        default_steps=50,
        max_steps=100,
        min_steps=1,
        quant="DF11",
    ),
    ModelType.LONGCAT_TURBO_DF11: ModelConfig(
        model_id="mingyi456/LongCat-Image-Edit-Turbo-DF11",
        default_steps=8,
        max_steps=100,
        min_steps=1,
        turbo=True,
        quant="DF11",
    ),
}


# Image processing constants
@dataclass(frozen=True)
class ImageProcessingConfig:
    """Configuration for image processing."""
    min_size: int = 64  # Minimum dimension for AI model input
    max_size: int = 4096  # Maximum dimension
    padding_multiple: int = 16  # Images must be multiples of this
    
    # OCR constants
    pipeline_version: str = "v1.5"  # PaddleOCR-VL pipeline version(v1.5 or v1.0)
    ocr_max_pixels: int = 2048 * 28 * 28
    use_tensorrt: bool = True  # Use TensorRT for OCR if available
    precision: str = "fp16"  # Precision for OCR model (fp16 or fp32)
    
    # Box expansion
    default_expand_pixels: int = 25
    max_expand_pixels: int = 500
    
    # Edge blending
    default_blend_margin: int = 0
    
    # Smart Fill
    default_color_variance_threshold: float = 10.0
    smart_fill_blend_max: int = 3  # Maximum blend margin for smart fill
    smart_fill_blend_divisor: int = 4  # Divisor for calculating blend from w/h

    # Color correction
    default_border_width: int = 1
    
    # Box expansion algorithm
    min_cos_half_threshold: float = 0.01  # Minimum cos_half for angle bisector calc
    
    # GPU memory management
    gpu_cleanup_interval: int = 5  # Clear GPU memory every N AI regions


IMAGE_CONFIG = ImageProcessingConfig()


# File handling - Based on OpenCV imread/imwrite support
# See: https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html
SUPPORTED_IMAGE_EXTENSIONS: tuple[str, ...] = (
    # JPEG formats
    '.jpg', '.jpeg', '.jpe', '.jp2',
    # PNG
    '.png',
    # BMP
    '.bmp', '.dib',
    # TIFF
    '.tiff', '.tif',
    # WebP
    '.webp',
    # Portable formats
    '.pbm', '.pgm', '.ppm', '.pxm', '.pnm',
    # Sun rasters
    '.sr', '.ras',
    # OpenEXR
    '.exr',
    # Radiance HDR
    '.hdr', '.pic',
)

# Environment
ENV_FILE = ".env"
HF_TOKEN_KEY = "HF_TOKEN"


# Logging
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"
