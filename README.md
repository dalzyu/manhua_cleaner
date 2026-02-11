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
- **Modern PyQt6 GUI** with live preview(janky, but works)
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
| FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32 | ~15GB | Fast | Good |
| LongCat-Image-Edit | ~18GB | Slow | Subpar |
| LongCat-Image-Edit-Turbo | ~18GB | Fast | OK |

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
- CUDA-capable GPU recommended (16GB VRAM for best results, 6GB minimum)
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
