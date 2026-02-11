"""Image entity - abstraction over image data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class ImageData(Protocol):
    """Protocol for image data - allows different backends."""
    
    @property
    def width(self) -> int: ...
    
    @property
    def height(self) -> int: ...
    
    @property
    def mode(self) -> str: ...
    
    def crop(self, box: tuple[int, int, int, int]) -> ImageData: ...
    
    def paste(self, im: ImageData, box: tuple[int, int, int, int] | None = None) -> None: ...
    
    def convert(self, mode: str) -> ImageData: ...
    
    def save(self, path: Path | str, **kwargs) -> None: ...


@dataclass(frozen=True, slots=True)
class Image:
    """Domain entity representing an image.
    
    Wraps underlying image data without exposing implementation details.
    """
    _data: ImageData
    source_path: Path | None = None
    
    @property
    def width(self) -> int:
        return self._data.width
    
    @property
    def height(self) -> int:
        return self._data.height
    
    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)
    
    @property
    def mode(self) -> str:
        return self._data.mode
    
    def crop(self, x: int, y: int, w: int, h: int) -> Image:
        """Crop to region and return new Image."""
        return Image(
            _data=self._data.crop((x, y, x + w, y + h)),
            source_path=self.source_path
        )
    
    def paste(self, other: Image, x: int, y: int) -> None:
        """Paste another image at position."""
        self._data.paste(other._data, (x, y))
    
    def convert(self, mode: str) -> Image:
        """Convert to different color mode."""
        return Image(
            _data=self._data.convert(mode),
            source_path=self.source_path
        )
    
    def save(self, path: Path | str) -> None:
        """Save image to path."""
        self._data.save(path)
    
    @classmethod
    def from_file(cls, path: Path | str) -> Image:
        """Load image from file."""
        path = Path(path)
        # Lazy import - domain doesn't depend on PIL
        from PIL import Image as PILImage
        return cls(_data=PILImage.open(path), source_path=path)
    
    @classmethod
    def from_array(cls, data: object, source_path: Path | None = None) -> Image:
        """Create from numpy array or other data."""
        from PIL import Image as PILImage
        return cls(_data=PILImage.fromarray(data), source_path=source_path)
    
    def to_array(self) -> object:
        """Convert to numpy array."""
        import numpy as np
        return np.array(self._data)


@dataclass(frozen=True, slots=True)
class Mask:
    """Binary mask for image regions."""
    data: object  # numpy array or similar
    
    @property
    def width(self) -> int:
        return self.data.shape[1]
    
    @property
    def height(self) -> int:
        return self.data.shape[0]


@dataclass(frozen=True, slots=True)
class ProcessingResult:
    """Result of processing an image."""
    success: bool
    image: Image | None = None
    error_message: str | None = None
    boxes_processed: int = 0
    boxes_smart_filled: int = 0
    processing_time_ms: float = 0.0
    
    @classmethod
    def failure(cls, error: str) -> ProcessingResult:
        """Create a failure result."""
        return cls(success=False, error_message=error)
    
    @classmethod
    def success_result(
        cls,
        image: Image,
        boxes_processed: int = 0,
        boxes_smart_filled: int = 0,
        processing_time_ms: float = 0.0
    ) -> ProcessingResult:
        """Create a success result."""
        return cls(
            success=True,
            image=image,
            boxes_processed=boxes_processed,
            boxes_smart_filled=boxes_smart_filled,
            processing_time_ms=processing_time_ms
        )
