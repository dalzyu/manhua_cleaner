# Manhua Image Cleaner

**AI-powered text removal from manga, manhua, and comic images.**

> ⚠️ **IMPORTANT DISCLAIMER: PROVIDED AS-IS**
>
> This software is provided **as-is**, without warranty of any kind, express or implied. Use at your own risk. The authors and contributors are not responsible for any damage, data loss, or issues arising from the use of this software. Always backup your original images before processing.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [GUI](#gui)
  - [CLI](#cli)
  - [Python API](#python-api)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Plugin System](#plugin-system)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)

---

## Features

- **OCR-based text detection** using PaddleOCR or EasyOCR
- **AI-powered inpainting** with FLUX.2-klein and LongCat models
- **Smart fill optimization** for simple backgrounds (reduces AI workload)
- **Batch processing** for entire folders
- **Character whitelist** to preserve SFX, symbols, or specific text
- **Plugin system** for extensibility
- **Modern PyQt6 GUI** with live preview
- **Command-line interface** for automation
- **Clean Architecture** - modular, testable, maintainable

---

## Installation

### Requirements

- Python 3.10 or higher
- CUDA-capable GPU recommended (8GB+ VRAM for AI models)
- 16GB+ RAM recommended for batch processing

### Installation Options

The package uses optional dependencies to keep the core lightweight:

```bash
# Minimal install - core only (50MB, no GPU needed)
# Use this if you only need the domain logic and utilities
pip install manhua-cleaner

# With OCR support - adds PaddleOCR (~500MB)
pip install manhua-cleaner[ocr]

# With AI models - adds PyTorch + Diffusers (~5GB, requires GPU)
pip install manhua-cleaner[models]

# With GUI - adds PyQt6 (~200MB)
pip install manhua-cleaner[gui]

# With quantization support - adds SDNQ/DFloat11
pip install manhua-cleaner[quantization]

# Everything included (full installation ~6GB)
pip install manhua-cleaner[all]

# CPU-only install (OCR only, no AI models)
pip install manhua-cleaner[ocr,gui]
```

### Verify Installation

```bash
# Check available plugins
python -c "from manhua_cleaner.infrastructure import PluginRegistry; \
           print('OCR:', PluginRegistry.list_available_ocr()); \
           print('Models:', PluginRegistry.list_available_models())"
```

---

## Quick Start

### GUI Mode

```bash
# Launch the PyQt6 graphical interface
manhua-cleaner-gui
```

### CLI Mode

```bash
# Process a single image
manhua-cleaner image.jpg -o output/

# Process an entire folder
manhua-cleaner ./manga_chapter/ -o ./cleaned/

# Use LongCat model with 8 steps
manhua-cleaner ./images/ -o ./out/ -m LongCat-Image-Edit-Turbo -s 8

# Full options example
manhua-cleaner ./images/ -o ./out/ \
    --model FLUX.2-klein-4B \
    --device cuda \
    --steps 4 \
    --expand 25 \
    --smart-fill-expand 5 \
    --prompt "remove all text and speech bubbles" \
    --whitelist --whitelist-preset sfx_only \
    --verbose
```

---

## Usage

### GUI

The GUI provides a visual interface with:
- **Input/Output selection** - File or folder browsing
- **Model selection** - Choose AI model and device
- **Parameters panel** - Steps, expansion, smart fill settings
- **OCR settings** - Backend selection, precision, workers
- **Whitelist configuration** - Preserve specific text patterns
- **Live preview** - See results before/after processing
- **Progress tracking** - Real-time processing status

```bash
manhua-cleaner-gui
```

### CLI

#### Basic Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `input` | - | - | Input image file or folder |
| `--output` | `-o` | - | Output folder (required) |
| `--model` | `-m` | FLUX.2-klein-4B | AI model to use |
| `--device` | `-d` | auto | Compute device (auto/cuda/mps/cpu) |
| `--steps` | `-s` | model default | Number of inference steps |
| `--expand` | `-e` | 25 | Pixels to expand text boxes |
| `--smart-fill-expand` | - | 5 | Smart fill expansion (smaller than AI) |
| `--prompt` | `-p` | "remove all text" | Inpainting prompt |
| `--verbose` | `-v` | - | Enable verbose output |

#### Processing Options

| Argument | Description |
|----------|-------------|
| `--no-color-correct` | Disable color correction |
| `--no-smart-fill` | Disable smart fill optimization |
| `--no-edge-blend` | Disable edge blending |
| `--extra-pass` | Enable extra quality pass (provide prompt) |
| `--continue-on-error` | Continue batch processing on failure |

#### OCR Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--ocr-model` | paddleocr | OCR backend (paddleocr/easyocr) |
| `--ocr-version` | v1.5 | PaddleOCR pipeline version |
| `--ocr-precision` | fp16 | OCR precision (fp16/fp32) |
| `--no-ocr-tensorrt` | - | Disable TensorRT acceleration |
| `--ocr-workers` | 1 | Parallel OCR workers (VRAM intensive) |

#### Whitelist Options

| Argument | Description |
|----------|-------------|
| `--whitelist` | Enable whitelist filtering |
| `--whitelist-preset` | Use built-in preset (none/sfx_only/hearts/symbols/japanese_sfx) |
| `--whitelist-file` | Load patterns from JSON/text file |
| `--whitelist-patterns` | Custom regex patterns (space-separated) |
| `--whitelist-distance` | Max distance for text grouping (default: 50px) |

#### CLI Examples

```bash
# Basic usage
manhua-cleaner manga_page.jpg -o cleaned/

# Batch processing with specific model
manhua-cleaner ./chapter_01/ -o ./cleaned/ \
    --model LongCat-Image-Edit-Turbo \
    --steps 8 \
    --device cuda

# Preserve SFX symbols using whitelist
manhua-cleaner manga.jpg -o out/ \
    --whitelist \
    --whitelist-preset sfx_only \
    --whitelist-distance 30

# Custom whitelist patterns
manhua-cleaner manga.jpg -o out/ \
    --whitelist \
    --whitelist-patterns "^[!?]+$" "^[♡♥❤]+$"

# Load whitelist from file
manhua-cleaner manga.jpg -o out/ \
    --whitelist \
    --whitelist-file patterns.txt

# High-quality processing with extra pass
manhua-cleaner cover.jpg -o out/ \
    --steps 12 \
    --extra-pass "enhance quality, sharpen details"

# Batch processing with error tolerance
manhua-cleaner ./batch/ -o ./out/ \
    --continue-on-error \
    --verbose

# CPU-only processing (no GPU)
manhua-cleaner manga.jpg -o out/ \
    --device cpu \
    --ocr-model easyocr
```

### Python API

#### Basic Usage

```python
from manhua_cleaner import BatchProcessor, ProcessingConfig, ModelType

# Create configuration
config = ProcessingConfig(
    model_type=ModelType.FLUX_2_KLEIN_4B,
    steps=4,
    expand_pixels=25,
    smart_fill=True,
    smart_fill_expand_pixels=5,
    prompt="remove all text"
)

# Process with automatic cleanup
processor = BatchProcessor(config)

with processor:
    result = processor.process_image("page_01.jpg")
    if result.success:
        print(f"Smart filled: {result.boxes_smart_filled}")
        print(f"AI processed: {result.boxes_processed}")
        result.image.save("cleaned_01.jpg")
```

#### Advanced Usage with New Architecture

```python
from manhua_cleaner.domain.entities.image import Image
from manhua_cleaner.domain.value_objects.config import ProcessingConfig, ModelType
from manhua_cleaner.application.services.text_removal import TextRemovalService
from manhua_cleaner.infrastructure.plugin_registry import PluginRegistry

# Create engines via plugin system
ocr = PluginRegistry.create_ocr_engine('paddleocr')
model = PluginRegistry.create_model('flux')

# Configure
config = ProcessingConfig(
    model_type=ModelType.FLUX_2_KLEIN_4B,
    steps=4,
    expand_pixels=25,
    whitelist_enabled=True,
    whitelist_preset='hearts'
)

# Create service
service = TextRemovalService(
    ocr=ocr,
    image_model=model,
    config=config
)

# Process with context manager
with service:
    result = service.remove_text("manga_page.jpg")
    if result.success:
        print(f"Processing time: {result.processing_time_ms}ms")
        result.image.save("cleaned.jpg")
```

#### Event Subscription

```python
from manhua_cleaner.application.ports.event_publisher import ProcessingEvent

# Subscribe to processing events
def on_event(event: ProcessingEvent):
    print(f"[{event.stage}] {event.message}")
    if event.progress:
        print(f"Progress: {event.progress:.0%}")

service.subscribe_to_events(on_event)
```

---

## Configuration

### Available Models

| Model | Parameters | Default Steps | VRAM | Speed | Quality |
|-------|------------|---------------|------|-------|---------|
| FLUX.2-klein-4B | 4B | 4 | ~8GB | Fast | Good |
| FLUX.2-klein-9B | 9B | 4 | ~16GB | Fast | Better |
| FLUX.2-klein-9B-SDNQ-4bit | 9B (quantized) | 4 | ~6GB | Fast | Good |
| LongCat-Image-Edit | - | 50 | ~12GB | Medium | Excellent |
| LongCat-Image-Edit-Turbo | - | 8 | ~12GB | Fast | Good |
| LongCat-Image-Edit-DF11 | - (quantized) | 50 | ~8GB | Medium | Good |

### Whitelist Presets

| Preset | Description |
|--------|-------------|
| `none` | No whitelisting |
| `sfx_only` | SFX symbols (!?…〜) |
| `punctuation` | Ending punctuation |
| `hearts` | Heart symbols (♡♥❤) |
| `symbols` | Music notes, stars (♪♫★☆) |
| `japanese_sfx` | Japanese SFX characters |

---

## Architecture

This project uses **Clean Architecture** with clear separation of concerns:

```
manhua_cleaner/
├── domain/              # Pure business logic, zero dependencies
│   ├── entities/        # Image, TextRegion, ProcessingResult
│   ├── value_objects/   # Config, Geometry types
│   └── services/        # Box merging, smart fill, text grouping
├── application/         # Use cases and orchestration
│   ├── ports/           # Interfaces (OCREngine, ImageModel)
│   └── services/        # TextRemovalService, BatchProcessor
├── adapters/            # External integrations
│   ├── models/          # FLUX, LongCat implementations
│   ├── ocr/             # PaddleOCR, EasyOCR implementations
│   └── persistence/     # File I/O, caching
├── infrastructure/      # Plugin system, utilities
└── interfaces/          # CLI, GUI, API
```

### Key Design Principles

1. **Dependency Inversion** - Domain depends on nothing
2. **Plugin System** - Extensible via entry points
3. **Lazy Loading** - Heavy deps only loaded when needed
4. **Protocol-based** - Structural typing with Python protocols

---

## Plugin System

The plugin system allows third-party developers to add new OCR engines or AI models without modifying the core codebase.

### Creating a Plugin

1. Implement the port interface:

```python
# my_ocr_plugin.py
from manhua_cleaner.application.ports.ocr_engine import OCREngine, TextDetectionResult

class MyOCR(OCREngine):
    @property
    def name(self) -> str:
        return "MyCustomOCR"
    
    def detect(self, image):
        # Your implementation
        return TextDetectionResult(regions=[...])
```

2. Register via entry points in your `pyproject.toml`:

```toml
[project.entry-points."manhua_cleaner.ocr"]
my_ocr = "my_ocr_plugin:MyOCR"
```

3. Users can now use your plugin:

```bash
manhua-cleaner image.jpg -o out/ --ocr-model my_ocr
```

### Available Plugin Hooks

- `manhua_cleaner.ocr` - OCR engines
- `manhua_cleaner.models` - AI image models

---

## Feature Explanations

### Character Whitelist
- **Purpose**: Preserve specific text (SFX, symbols) while removing other text
- **How it works**: Groups nearby text regions and checks if combined text matches whitelist patterns
- **Usage**: Enable with `--whitelist` and choose a preset or custom patterns
- **Presets**: 
  - `sfx_only` - Keeps punctuation and SFX symbols (!?…〜)
  - `hearts` - Keeps heart symbols (♡♥❤)
  - `symbols` - Keeps music notes, stars (♪♫★☆)
  - `japanese_sfx` - Keeps Japanese SFX characters
- **Example**: `--whitelist --whitelist-preset sfx_only`

### Smart Fill
- **Purpose**: Automatically fill text on simple/monochromatic backgrounds without using AI
- **How it works**: Checks color variance around text region edges; if low variance, fills with average color
- **Benefit**: Can reduce AI workload by up to 80% in images with simple backgrounds
- **Usage**: Enabled by default (`--smart-fill-expand` controls expansion)
- **Note**: May occasionally fill unintended areas if background near text is complex

### Smart Fill Expand
- **Purpose**: Expands text bounding boxes before smart fill analysis
- **How it works**: Adds padding around detected text regions to include context
- **Default**: 5 pixels (smaller than AI expand)
- **Trade-off**: Higher values reduce eligible regions (more goes to AI); lower values may miss text edges
- **Usage**: `--smart-fill-expand 10`

### AI Expand
- **Purpose**: Expands text bounding boxes before AI inpainting
- **How it works**: Grows the polygon around detected text to give AI context
- **Default**: 25 pixels
- **Trade-offs**:
  - Higher values: More context for AI, but slower and may affect unintended areas
  - Lower values: Faster, but may cause color shifting with some models
- **Usage**: `--expand 30`

### Color Correction
- **Purpose**: Fixes color shifts from AI processing
- **How it works**: Detects color difference between edges of original and processed regions, applies correction
- **Usage**: Enabled by default; disable with `--no-color-correct`
- **When to disable**: If you notice unwanted color changes in processed areas

### Edge Blending
- **Purpose**: Blends processed regions with original image for natural transitions
- **How it works**: Applies gradient fade at boundaries of filled regions
- **Usage**: Enabled by default; disable with `--no-edge-blend`
- **Note**: Helps hide boundaries between original and AI-generated content

### Extra Pass
- **Purpose**: Run AI model a second time on the entire image for quality improvement
- **Use cases**:
  - Upscale the image
  - Apply overall quality enhancement
  - Fix remaining artifacts from first pass
- **Usage**: `--extra-pass "improve image quality"`
- **Note**: Balloons processing time; best for low quality images or images with undetecatable text

### Extra Pass Upscaling
- **Purpose**: Increase image resolution before extra pass
- **How it works**: Upscales image using Lanczos/bicubic/bilinear, then AI adds detail
- **Options**: 1.5x, 2x, 3x, 4x, 8x
- **Methods**: Lanczos (best quality), Bicubic (balanced), Bilinear (fastest)
- **Usage**: Available in GUI; for CLI use Python API

### OCR Workers
- **Purpose**: Parallel OCR processing for batch operations
- **How it works**: Each worker loads a separate OCR model instance
- **Default**: 1 worker
- **Warning**: Each worker uses ~16GB VRAM - increase with caution
- **Usage**: `--ocr-workers 2` (only if you have 32GB+ VRAM)

### OCR Pipeline Version (PaddleOCR-VL)
- **Options**: v1.5 (default), v1.0
- **Usage**: `--ocr-version v1.5`

### OCR Precision
- **Options**: fp16 (default), fp32
- **fp16**: Half precision, faster, uses less VRAM
- **fp32**: Full precision, more accurate, slower
- **Usage**: `--ocr-precision fp32` (if OCR misses text)

### Continue on Error
- **Purpose**: Don't stop batch processing if one image fails
- **Usage**: `--continue-on-error`
- **Behavior**: Logs errors, continues with remaining images, reports failures at end

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Use quantized models
manhua-cleaner image.jpg -o out/ --model FLUX.2-klein-9B-SDNQ-4bit

# Reduce batch size (process fewer at once)
# Close other GPU applications
```

### OCR Missing Text

```bash
# Increase expansion
manhua-cleaner image.jpg -o out/ --expand 40

# Try different OCR backend
manhua-cleaner image.jpg -o out/ --ocr-model easyocr

# Use FP32 precision for better accuracy
manhua-cleaner image.jpg -o out/ --ocr-precision fp32
```

### Slow Processing

```bash
# Enable smart fill
manhua-cleaner image.jpg -o out/ --smart-fill-expand 10

# Use Turbo models
manhua-cleaner image.jpg -o out/ --model LongCat-Image-Edit-Turbo --steps 8

# Reduce inference steps
manhua-cleaner image.jpg -o out/ --steps 4
```

### Import Errors

```bash
# If you get "No module named torch"
pip install manhua-cleaner[models]

# If you get "No module named paddleocr"
pip install manhua-cleaner[ocr]

# If you get "No module named PyQt6"
pip install manhua-cleaner[gui]
```

---

## Limitations

⚠️ **Please read before using:**

1. **Quality Variability**: AI inpainting quality varies based on image complexity, text size, and background detail. Results are not guaranteed.

2. **Processing Time**: AI models are computationally intensive. A single image may take 10-60 seconds depending on settings and hardware.

3. **VRAM Requirements**: AI models require significant GPU memory. Ensure you have sufficient VRAM or use quantized models.

4. **OCR Accuracy**: Text detection depends on image quality. Low-resolution, blurry, or highly stylized text may be missed.

5. **Backup Your Data**: Always keep originals. The software may occasionally produce unsatisfactory results that require manual touch-ups.

6. **As-Is Software**: This software is provided without warranty. Use at your own risk.

---

## Development

### Running Tests

```bash
# Install dev dependencies
pip install manhua-cleaner[dev]

# Run tests
pytest manhua_cleaner/tests/ -v

# Run with coverage
pytest --cov=manhua_cleaner --cov-report=html
```

### Project Structure

```
manhua_cleaner/
├── domain/              # Zero-dependency business logic
│   ├── entities/        # Core domain objects
│   ├── value_objects/   # Immutable config/geometry
│   └── services/        # Pure business logic
├── application/         # Use cases
│   ├── ports/           # Abstract interfaces
│   └── services/        # Orchestration
├── adapters/            # External implementations
├── infrastructure/      # Plugin system
└── interfaces/          # CLI/GUI
```

---

## License

<<<<<<< HEAD
GPLv3 - See LICENSE file for details.

=======
See LICENSE file for details.

---

## Disclaimer (Reiterated)

**THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.**

**IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.**

**YOU USE THIS SOFTWARE ENTIRELY AT YOUR OWN RISK. ALWAYS BACKUP YOUR ORIGINAL IMAGES.**
>>>>>>> f9d6c2b (changed readme)
