"""Text region representation for OCR results."""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import numpy.typing as npt

Contour = npt.NDArray[np.int32]


@dataclass
class TextRegion:
    """A single detected text region with its contour and recognized text."""
    
    contour: Contour
    text: str
    confidence: float = 0.0
    
    @property
    def center(self) -> tuple[float, float]:
        """Get center point of the region."""
        moments = cv2.moments(self.contour)
        if moments["m00"] == 0:
            # Fallback to bounding box center
            x, y, w, h = cv2.boundingRect(self.contour)
            return (x + w / 2, y + h / 2)
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        return (cx, cy)
    
    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Get bounding box as (x, y, w, h)."""
        return cv2.boundingRect(self.contour)
    
    @property
    def area(self) -> float:
        """Get contour area."""
        return cv2.contourArea(self.contour)


@dataclass
class Textbox:
    """A logical textbox containing one or more text regions."""
    
    regions: list[TextRegion]
    combined_text: str
    
    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Get union bounding box of all regions."""
        if not self.regions:
            return (0, 0, 0, 0)
        
        x1 = min(r.bbox[0] for r in self.regions)
        y1 = min(r.bbox[1] for r in self.regions)
        x2 = max(r.bbox[0] + r.bbox[2] for r in self.regions)
        y2 = max(r.bbox[1] + r.bbox[3] for r in self.regions)
        return (x1, y1, x2 - x1, y2 - y1)
    
    @property
    def center(self) -> tuple[float, float]:
        """Get center of the textbox."""
        bbox = self.bbox
        return (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
    
    @property
    def contours(self) -> list[Contour]:
        """Get all contours from regions."""
        return [r.contour for r in self.regions]


import cv2  # noqa: E402
