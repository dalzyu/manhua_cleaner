"""Modern CLI interface using the new architecture."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ...domain.entities.image import Image
from ...domain.value_objects.config import (
    ProcessingConfig, ModelType, Backend, WHITELIST_PRESETS
)
from ...application.services.text_removal import TextRemovalService
from ...infrastructure.plugin_registry import PluginRegistry

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="manhua-cleaner",
        description="AI-powered text removal from manga/manhua images"
    )
    
    parser.add_argument("input", help="Input image or folder")
    parser.add_argument("-o", "--output", required=True, help="Output folder")
    
    # Model settings
    parser.add_argument(
        "-m", "--model",
        choices=[m.value for m in ModelType],
        default=ModelType.FLUX_2_KLEIN_4B.value,
        help="AI model to use (default: FLUX.2-klein-4B)"
    )
    
    parser.add_argument(
        "-d", "--device",
        choices=[b.value for b in Backend],
        default="auto",
        help="Compute device (default: auto)"
    )
    
    parser.add_argument(
        "-s", "--steps",
        type=int,
        help="Number of inference steps (model default if not specified)"
    )
    
    # Box expansion
    parser.add_argument(
        "-e", "--expand",
        type=int,
        default=25,
        help="Pixels to expand text boxes for AI (default: 25)"
    )
    
    parser.add_argument(
        "--smart-fill-expand",
        type=int,
        default=5,
        help="Expansion pixels for smart fill, smaller than AI expand (default: 5)"
    )
    
    # Processing options
    parser.add_argument(
        "--no-color-correct",
        action="store_true",
        help="Disable color correction"
    )
    
    parser.add_argument(
        "--no-smart-fill",
        action="store_true",
        help="Disable smart fill optimization"
    )
    
    parser.add_argument(
        "--no-edge-blend",
        action="store_true",
        help="Disable edge blending"
    )
    
    # Prompts
    parser.add_argument(
        "-p", "--prompt",
        default="remove all text",
        help="Inpainting prompt (default: 'remove all text')"
    )
    
    parser.add_argument(
        "--extra-pass",
        help="Extra pass prompt for quality improvement (enables extra pass if set)"
    )
    
    # Error handling
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
    
    # OCR settings group
    ocr_group = parser.add_argument_group("OCR options")
    ocr_group.add_argument(
        "--ocr-model",
        choices=PluginRegistry.list_available_ocr(),
        default="paddleocr",
        help="OCR backend to use (default: paddleocr)"
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
        help="Disable TensorRT acceleration for OCR"
    )
    ocr_group.add_argument(
        "--ocr-workers",
        type=int,
        default=1,
        metavar="N",
        help="Number of OCR worker processes, each uses ~16GB VRAM (default: 1)"
    )
    
    # Whitelist settings group
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
        help="Use a built-in whitelist preset (default: none)"
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
        help="Maximum distance for grouping text regions (default: 50)"
    )
    
    return parser


def load_whitelist_patterns(parsed_args) -> list[str]:
    """Load whitelist patterns from various sources."""
    patterns = []
    
    # Get preset patterns
    if parsed_args.whitelist_preset != "none":
        patterns = WHITELIST_PRESETS.get(parsed_args.whitelist_preset, []).copy()
    
    # Load from file if specified
    if parsed_args.whitelist_file:
        if parsed_args.whitelist_file.suffix.lower() == '.json':
            import json
            with open(parsed_args.whitelist_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                file_patterns = data.get('patterns', [])
        else:
            with open(parsed_args.whitelist_file, 'r', encoding='utf-8') as f:
                file_patterns = [
                    line.strip() for line in f
                    if line.strip() and not line.startswith('#')
                ]
        patterns.extend(file_patterns)
    
    # Add command-line patterns
    if parsed_args.whitelist_patterns:
        patterns.extend(parsed_args.whitelist_patterns)
    
    return patterns


def process_image(
    service: TextRemovalService,
    input_path: Path,
    output_path: Path,
    prefix: str = "cleaned_"
) -> tuple[bool, dict]:
    """Process a single image.
    
    Returns:
        Tuple of (success, stats)
    """
    logger.info(f"Processing {input_path.name}...")
    
    result = service.remove_text(input_path)
    
    if not result.success:
        logger.error(f"Failed: {result.error_message}")
        return False, {}
    
    if result.image:
        output_file = output_path / f"{prefix}{input_path.name}"
        result.image.save(output_file)
        logger.info(
            f"Saved: {output_file.name} "
            f"(smart: {result.boxes_smart_filled}, AI: {result.boxes_processed})"
        )
        return True, {
            'smart_filled': result.boxes_smart_filled,
            'ai_processed': result.boxes_processed
        }
    
    return True, {}


def main(args: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    parsed = parser.parse_args(args)
    
    setup_logging(parsed.verbose)
    
    input_path = Path(parsed.input)
    output_path = Path(parsed.output)
    
    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        return 1
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load whitelist patterns if enabled
    whitelist_patterns = []
    whitelist_enabled = parsed.whitelist
    if parsed.whitelist_preset != "none":
        whitelist_enabled = True
    
    if whitelist_enabled or parsed.whitelist_file or parsed.whitelist_patterns:
        whitelist_enabled = True
        whitelist_patterns = load_whitelist_patterns(parsed)
    
    # Build configuration
    config = ProcessingConfig(
        model_type=ModelType(parsed.model),
        device=parsed.device,
        steps=parsed.steps,
        expand_pixels=parsed.expand,
        smart_fill_expand_pixels=parsed.smart_fill_expand,
        color_correct=not parsed.no_color_correct,
        smart_fill=not parsed.no_smart_fill,
        edge_blend=not parsed.no_edge_blend,
        ocr_model=parsed.ocr_model,
        ocr_pipeline_version=parsed.ocr_version,
        ocr_use_tensorrt=not parsed.no_ocr_tensorrt,
        ocr_precision=parsed.ocr_precision,
        ocr_workers=parsed.ocr_workers,
        prompt=parsed.prompt,
        extra_pass_prompt=parsed.extra_pass,
        whitelist_enabled=whitelist_enabled,
        whitelist_preset=parsed.whitelist_preset,
        whitelist_patterns=whitelist_patterns,
        whitelist_group_distance=parsed.whitelist_distance
    )
    
    # Create engines via plugin system
    try:
        ocr = PluginRegistry.create_ocr_engine(parsed.ocr_model)
        model = PluginRegistry.create_model(parsed.model)
    except Exception as e:
        logger.error(f"Failed to initialize engines: {e}")
        return 1
    
    # Create service
    service = TextRemovalService(
        ocr=ocr,
        image_model=model,
        config=config
    )
    
    # Get files to process
    if input_path.is_file():
        files = [input_path]
    else:
        # Supported image extensions
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif')
        files = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in extensions
        ]
        files.sort()
    
    if not files:
        logger.error("No image files found")
        return 1
    
    logger.info(f"Processing {len(files)} image(s)...")
    logger.info(f"Model: {parsed.model}, OCR: {parsed.ocr_model}")
    if whitelist_enabled:
        logger.info(f"Whitelist enabled with {len(whitelist_patterns)} patterns")
    
    success_count = 0
    failed = []
    total_smart_filled = 0
    total_ai_processed = 0
    
    try:
        with service:
            for i, file_path in enumerate(files, 1):
                logger.info(f"[{i}/{len(files)}] {file_path.name}")
                
                try:
                    success, stats = process_image(service, file_path, output_path)
                    
                    if success:
                        success_count += 1
                        total_smart_filled += stats.get('smart_filled', 0)
                        total_ai_processed += stats.get('ai_processed', 0)
                    else:
                        failed.append((file_path.name, "Processing failed"))
                        if not parsed.continue_on_error:
                            logger.error("Use --continue-on-error to process remaining images")
                            break
                            
                except Exception as e:
                    logger.exception(f"Unexpected error processing {file_path.name}")
                    failed.append((file_path.name, f"{type(e).__name__}: {e}"))
                    
                    if not parsed.continue_on_error:
                        logger.error("Use --continue-on-error to process remaining images")
                        break
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
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
