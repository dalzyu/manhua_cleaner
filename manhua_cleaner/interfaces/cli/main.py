"""Modern CLI interface using the new architecture."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ...domain.entities.image import Image
from ...domain.value_objects.config import ProcessingConfig, ModelType, Backend
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
    
    parser.add_argument(
        "-m", "--model",
        choices=PluginRegistry.list_available_models(),
        default="flux",
        help="AI model to use"
    )
    
    parser.add_argument(
        "-d", "--device",
        choices=[b.value for b in Backend],
        default="auto",
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
        default=25,
        help="Pixels to expand text boxes"
    )
    
    parser.add_argument(
        "--ocr",
        choices=PluginRegistry.list_available_ocr(),
        default="paddleocr",
        help="OCR engine"
    )
    
    parser.add_argument(
        "--no-smart-fill",
        action="store_true",
        help="Disable smart fill"
    )
    
    parser.add_argument(
        "-p", "--prompt",
        default="remove all text",
        help="Inpainting prompt"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    return parser


def process_image(
    service: TextRemovalService,
    input_path: Path,
    output_path: Path
) -> bool:
    """Process a single image.
    
    Returns:
        True if successful
    """
    logger.info(f"Processing {input_path.name}...")
    
    result = service.remove_text(input_path)
    
    if not result.success:
        logger.error(f"Failed: {result.error_message}")
        return False
    
    if result.image:
        output_file = output_path / f"cleaned_{input_path.name}"
        result.image.save(output_file)
        logger.info(
            f"Saved: {output_file.name} "
            f"(smart: {result.boxes_smart_filled}, AI: {result.boxes_processed})"
        )
    
    return True


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
    
    # Build configuration
    config = ProcessingConfig(
        model_type=ModelType.FLUX_2_KLEIN_4B,  # Simplified for now
        device=parsed.device,
        steps=parsed.steps,
        expand_pixels=parsed.expand,
        smart_fill=not parsed.no_smart_fill,
        ocr_model=parsed.ocr,
        prompt=parsed.prompt
    )
    
    # Create engines via plugin system
    try:
        ocr = PluginRegistry.create_ocr_engine(parsed.ocr)
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
    
    # Process files
    if input_path.is_file():
        files = [input_path]
    else:
        files = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        ]
        files.sort()
    
    if not files:
        logger.error("No image files found")
        return 1
    
    logger.info(f"Processing {len(files)} image(s)...")
    
    success_count = 0
    
    with service:
        for i, file_path in enumerate(files, 1):
            logger.info(f"[{i}/{len(files)}] {file_path.name}")
            if process_image(service, file_path, output_path):
                success_count += 1
    
    logger.info(f"Completed: {success_count}/{len(files)} succeeded")
    return 0 if success_count == len(files) else 1


if __name__ == "__main__":
    sys.exit(main())
