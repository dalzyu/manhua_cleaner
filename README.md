# Manhua Image Cleaner

**AI-powered text removal from manga, manhua, and comic images.**

⚠️ **IMPORTANT DISCLAIMER: PROVIDED AS-IS**

This vibe-coded software is provided **as-is**, without warranty of any kind, express or implied. Use at your own risk. The authors and contributors are not responsible for any damage, data loss, or issues arising from the use of this software. Always backup your original images before processing.

---

## Features

- **OCR-based text detection** using PaddleOCR-VL
- **AI-powered inpainting** with FLUX.2-klein and LongCat-Image-Edit models
- **Smart fill** for simple backgrounds (runs before AI for speed)
- **Batch processing** for entire folders
- **Character whitelist** to preserve SFX, symbols, or specific text
- **Plugin system** for extensibility
- **Modern PyQt6 GUI** with live preview
- **Command-line interface** for automation(not fully tested)

## Quick Start

### Installation
WIP
```

### GUI Usage

```bash
manhua-cleaner-gui
```

### CLI Usage

```bash
# Process single image
manhua-cleaner image.jpg -o output/

# Process folder
manhua-cleaner ./manga_chapter/ -o ./cleaned/

# Use specific model
manhua-cleaner ./images/ -o ./out/ -m longcat -s 8

# Full options
manhua-cleaner ./images/ -o ./out/ \
    -m flux \
    -d cuda \
    -s 4 \
    -e 25 \
    -p "remove all text and speech bubbles" \
    --no-smart-fill \
    -v
```

## Python API

```python
from manhua_cleaner import BatchProcessor, ProcessingConfig, ModelType

config = ProcessingConfig(
    model_type=ModelType.FLUX_2_KLEIN_4B,
    steps=4,
    expand_pixels=25,
    smart_fill=True,
    prompt="remove all text"
)

processor = BatchProcessor(config)

# Process with automatic cleanup
with processor:
    result = processor.process_image("page_01.jpg")
    if result.success:
        result.image.save("cleaned_01.jpg")
```

## Available Models

| Model | VRAM | Speed | Quality |
|-------|------|-------|---------|
| FLUX.2-klein-4B | ~13GB | Fast | Better |
| FLUX.2-klein-4B-SDNQ-4bit-dynamic | ~6GB | Fast | Good |
| FLUX.2-klein-9B | ~29GB | Fast | Great |
| FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32 | ~15GB | Fast | Better |
| LongCat-Image-Edit | ~18GB | Slow | Subpar |
| LongCat-Image-Edit-Turbo | ~18GB | Fast | OK |

## Features and Parmeters

### Character Whitelist
- **Purpose**: Preserve specific text (SFX, symbols) while removing others.
- **Usage**: Add characters to the whitelist in the GUI or CLI.
- **Example**: `sfx_only` preset to keep special effects.

### Smart Fill
- **Purpose**: Automatically fill text that resides on monochromatic backgrounds. Can reduce number of text regions processed by AI by upto 80% in some cases.
- **Usage**: Enable in processing (default: ON).
- **Note**: May fill in unintended areas as it only checks pixels on the the edges of the text region.

### Colour Correction
- **Purpose**: Fix colour shifts from AI processing. Detects difference in colour of edges of original text region and new one and applies correction.
- **Usage**: Enable in processing (default: ON).

### Edge Blending
- **Purpose**: Blends Edges of processed text regions with original image to make them look more natural.
- **Usage**: Enable in settings (default: ON).

## Extra Pass
- **Purpose**: Run the AI model a second time on the completed image to improve results or upscale.
- **Usage**: Enable in processing (default: OFF).

### Live Preview(WIP)
- **Purpose**: See the image in real-time in the GUI as it's being processed.
- **Note**: Kind of janky when processing more than one image at a time.

## AI expand
- **Purpose**: Expands the text bounding polygon detected by OCR by a specified number of pixels to give more context to the AI and ensure the AI fills the entire text region.
- **Usage**: Set the number of pixels to expand in the GUI or CLI.
- **Note**: Higher values will increase processing time and may cause AI model to change unintended areas. Lower values may cause more colour shifting with certain AI models.

## Smart Fill expand
- **Purpose**: Expands the text bounding polygon detected by Smart Fill by a specified number of pixels to give more context to the AI and ensure the AI fills the entire text region.
- **Usage**: Set the number of pixels to expand in the GUI or CLI.
- **Note**: Higher values will decrease number of eligible regions as image borders contain non-background elements. Lower values may cause some text to not be fully filled.

## Architecture

This project uses **Clean Architecture** for maintainability:

```
manhua_cleaner/
├── domain/          # Pure business logic (zero dependencies)
├── application/     # Use cases and orchestration
├── adapters/        # External integrations (OCR, AI models)
├── infrastructure/  # Plugin system, utilities
└── interfaces/      # CLI, GUI
```

### Plugin System

Extend functionality via entry points:

```toml
# In your pyproject.toml
[project.entry-points."manhua_cleaner.ocr"]
my_ocr = "my_package:MyOCREngine"

[project.entry-points."manhua_cleaner.models"]
my_model = "my_package:MyImageModel"
```

## System Requirements

- Python 3.10+
- CUDA-capable GPU recommended (16GB VRAM for best results)
- 32GB+ RAM recommended for batch processing

## Limitations & Known Issues

⚠️ **Please Read Before Use:**

1. **Quality Variability**: AI inpainting quality varies based on image complexity, prompting, text size, and background detail.

2. **Processing Time**: AI models are computationally intensive. A single image may take 5-60 seconds depending on settings and setup.

3. **VRAM Usage**: AI models require significant GPU memory. Close other GPU applications before use.

4. **OCR Accuracy**: Text detection depends on image quality. Low-resolution or stylized text may be missed.

5. **Backup Your Data**: Always keep originals. The software may occasionally produce unsatisfactory results.

6. **No Guarantee**: Results are not guaranteed. Manual touch-ups may be required.

## License

GPLv3 - See LICENSE file for details.
