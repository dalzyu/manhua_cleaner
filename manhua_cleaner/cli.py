"""Command-line interface for Manhua Image Cleaner."""

import argparse
import logging
import sys
from pathlib import Path

from .config import ModelType, Backend, IMAGE_CONFIG, WHITELIST_PRESETS
from .core import BatchProcessor, ProcessingConfig
from .exceptions import (
    ManhuaCleanerError,
    ImageProcessingError,
    OCRError,
    ModelError,
    ValidationError,
)
from .utils.env import load_hf_token, setup_logging


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="manhua-cleaner",
        description="AI-powered text removal from manga/manhua images"
    )
    
    parser.add_argument("input", help="Input image or folder")
    parser.add_argument("-o", "--output", required=True, help="Output folder")
    
    parser.add_argument(
        "-m", "--model",
        choices=[m.value for m in ModelType],
        default=ModelType.FLUX_2_KLEIN_4B.value,
        help="AI model to use"
    )
    
    parser.add_argument(
        "-d", "--device",
        choices=[b.value for b in Backend],
        default=Backend.AUTO.value,
        help="Compute device"
    )
    
    parser.add_argument(
        "-s", "--steps",
        type=int,
        help="Number of inference steps"
    )
    
    parser.add_argument(
        "-e", "--expand",
        type=int,
        default=IMAGE_CONFIG.default_expand_pixels,
        help="Pixels to expand text boxes for AI"
    )
    
    parser.add_argument(
        "--no-color-correct",
        action="store_true",
        help="Disable color correction"
    )
    
    parser.add_argument(
        "--no-smart-fill",
        action="store_true",
        help="Disable smart fill"
    )
    
    parser.add_argument(
        "--smart-fill-expand",
        type=int,
        default=5,
        help="Expansion pixels for smart fill (default: 5)"
    )
    
    parser.add_argument(
        "-p", "--prompt",
        default="remove all text",
        help="Inpainting prompt"
    )
    
    parser.add_argument(
        "--extra-pass",
        help="Extra pass prompt (enables extra pass if set)"
    )
    
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining images if one fails"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    # OCR settings
    ocr_group = parser.add_argument_group("OCR options")
    ocr_group.add_argument(
        "--ocr-model",
        choices=["paddleocr", "easyocr"],
        default="paddleocr",
        help="OCR backend to use (default: paddleocr). "
             "'paddleocr' is recommended for Asian text. "
             "'easyocr' is easier to install but may be slower."
    )
    ocr_group.add_argument(
        "--ocr-version",
        choices=["v1.5", "v1.0"],
        default="v1.5",
        help="PaddleOCR pipeline version (default: v1.5)"
    )
    ocr_group.add_argument(
        "--ocr-precision",
        choices=["fp16", "fp32"],
        default="fp16",
        help="OCR model precision (default: fp16)"
    )
    ocr_group.add_argument(
        "--no-ocr-tensorrt",
        action="store_true",
        help="Disable TensorRT for OCR"
    )
    ocr_group.add_argument(
        "--ocr-workers",
        type=int,
        default=1,
        metavar="N",
        help="Number of OCR worker processes (default: 1). Each worker uses ~16GB VRAM, increase with caution."
    )
    
    # Whitelist settings
    whitelist_group = parser.add_argument_group("Character whitelist options")
    whitelist_group.add_argument(
        "--whitelist",
        action="store_true",
        help="Enable character whitelist filtering"
    )
    whitelist_group.add_argument(
        "--whitelist-preset",
        choices=list(WHITELIST_PRESETS.keys()),
        default="none",
        help="Use a built-in whitelist preset"
    )
    whitelist_group.add_argument(
        "--whitelist-file",
        type=Path,
        help="Path to whitelist file (JSON or text, one pattern per line)"
    )
    whitelist_group.add_argument(
        "--whitelist-patterns",
        nargs="+",
        help="Additional regex patterns to whitelist (can be combined with preset)"
    )
    whitelist_group.add_argument(
        "--whitelist-distance",
        type=int,
        default=50,
        metavar="PIXELS",
        help="Maximum distance for grouping text regions into textboxes (default: 50)"
    )
    
    return parser


def main(args: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    parsed = parser.parse_args(args)
    
    # Setup logging
    setup_logging(logging.DEBUG if parsed.verbose else logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Validate paths
    input_path = Path(parsed.input)
    output_path = Path(parsed.output)
    
    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        return 1
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check token
    token = load_hf_token()
    if not token:
        logger.warning("No HuggingFace token found. Set HF_TOKEN environment variable or use GUI.")
    
    # Load whitelist configuration
    whitelist_enabled = parsed.whitelist
    whitelist_patterns = []
    whitelist_preset = parsed.whitelist_preset
    
    if whitelist_enabled or whitelist_preset != "none":
        whitelist_enabled = True
        # Get preset patterns
        whitelist_patterns = WHITELIST_PRESETS.get(whitelist_preset, []).copy()
        
        # Load from file if specified
        if parsed.whitelist_file:
            from .core.whitelist_filter import WhitelistConfig
            if parsed.whitelist_file.suffix.lower() == '.json':
                file_config = WhitelistConfig.from_file(parsed.whitelist_file)
                whitelist_patterns.extend(file_config.patterns)
            else:
                file_config = WhitelistConfig.from_text_file(parsed.whitelist_file)
                whitelist_patterns.extend(file_config.patterns)
        
        # Add command-line patterns
        if parsed.whitelist_patterns:
            whitelist_patterns.extend(parsed.whitelist_patterns)
    
    # Create config
    config = ProcessingConfig(
        model_type=ModelType(parsed.model),
        device=parsed.device,
        steps=parsed.steps,
        expand_pixels=parsed.expand,
        color_correct=not parsed.no_color_correct,
        smart_fill=not parsed.no_smart_fill,
        smart_fill_expand_pixels=parsed.smart_fill_expand,
        ocr_model=parsed.ocr_model,
        ocr_pipeline_version=parsed.ocr_version,
        ocr_use_tensorrt=not parsed.no_ocr_tensorrt,
        ocr_precision=parsed.ocr_precision,
        ocr_workers=parsed.ocr_workers,
        prompt=parsed.prompt,
        extra_pass_prompt=parsed.extra_pass,
        whitelist_enabled=whitelist_enabled,
        whitelist_preset=whitelist_preset,
        whitelist_patterns=whitelist_patterns,
        whitelist_group_distance=parsed.whitelist_distance
    )
    
    # Get files to process
    if input_path.is_file():
        files = [input_path]
    else:
        from .config import SUPPORTED_IMAGE_EXTENSIONS
        files = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        files.sort()
    
    if not files:
        logger.error("No image files found")
        return 1
    
    logger.info(f"Processing {len(files)} image(s)...")
    
    # Process
    processor = BatchProcessor(config)
    success_count = 0
    failed = []
    total_smart_filled = 0
    total_ai_processed = 0
    
    def progress(msg: str):
        if parsed.verbose:
            logger.debug(msg)
    
    try:
        for i, file_path in enumerate(files, 1):
            logger.info(f"[{i}/{len(files)}] Processing {file_path.name}...")
            
            try:
                result = processor.process_image(file_path, progress)
                
                output_file = output_path / f"cleaned_{file_path.name}"
                result.image.save(output_file)
                
                # Log stats
                if result.boxes_smart_filled > 0 or result.boxes_processed > 0:
                    total_smart_filled += result.boxes_smart_filled
                    total_ai_processed += result.boxes_processed
                    logger.info(
                        f"  Saved: {output_file.name} "
                        f"(smart: {result.boxes_smart_filled}, AI: {result.boxes_processed})"
                    )
                else:
                    logger.info(f"  Saved: {output_file.name}")
                
                success_count += 1
                
            except (ValidationError, ImageProcessingError, OCRError, ModelError) as e:
                # Expected errors - log and continue or exit based on flag
                error_msg = str(e)
                if hasattr(e, 'image_path'):
                    logger.error(f"  Failed to process {file_path.name}: {error_msg}")
                else:
                    logger.error(f"  Failed: {error_msg}")
                failed.append((file_path.name, error_msg))
                
                if not parsed.continue_on_error:
                    logger.error("Use --continue-on-error to process remaining images")
                    break
                    
            except Exception as e:
                # Unexpected errors
                logger.exception(f"  Unexpected error processing {file_path.name}")
                failed.append((file_path.name, f"Unexpected: {type(e).__name__}: {e}"))
                
                if not parsed.continue_on_error:
                    logger.error("Use --continue-on-error to process remaining images")
                    break
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        processor.unload_models()
    
    # Summary
    logger.info("=" * 50)
    total_boxes = total_smart_filled + total_ai_processed
    if total_boxes > 0:
        optimization = 100 * total_smart_filled / total_boxes
        logger.info(
            f"Stats: {total_smart_filled} smart filled, "
            f"{total_ai_processed} AI processed ({optimization:.0f}% optimized)"
        )
    
    if failed:
        logger.warning(f"Completed: {success_count}/{len(files)} succeeded")
        for name, error in failed:
            logger.error(f"  - {name}: {error}")
        return 1
    else:
        logger.info(f"Completed: All {len(files)} images processed successfully")
        return 0


if __name__ == "__main__":
    sys.exit(main())
