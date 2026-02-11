"""Text region entities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..value_objects.geometry import Contour, Point


@dataclass(frozen=True, slots=True)
class TextRegion:
    """A single detected text region with its content."""
    contour: 'Contour'
    text: str = ""
    confidence: float = 0.0
    
    @property
    def is_empty(self) -> bool:
        """Check if region has no text."""
        return not self.text.strip()
    
    @property
    def area(self) -> float:
        """Calculate area of the contour."""
        return self.contour.area


@dataclass(slots=True)
class Textbox:
    """A group of nearby text regions treated as one text box."""
    regions: list[TextRegion] = field(default_factory=list)
    
    @property
    def combined_text(self) -> str:
        """Concatenate all text from regions."""
        return " ".join(r.text for r in self.regions if r.text)
    
    @property
    def center(self) -> 'Point':
        """Calculate centroid of all regions."""
        from ..value_objects.geometry import Point
        if not self.regions:
            return Point(0, 0)
        
        # Average of region centroids weighted by area
        total_area = sum(r.area for r in self.regions)
        if total_area == 0:
            return Point(0, 0)
        
        x = sum(r.contour.centroid.x * r.area for r in self.regions) / total_area
        y = sum(r.contour.centroid.y * r.area for r in self.regions) / total_area
        return Point(x, y)
    
    def add_region(self, region: TextRegion) -> None:
        """Add a text region to this box."""
        self.regions.append(region)
    
    @property
    def is_whitelisted(self, patterns: list[str] | None = None) -> bool:
        """Check if this textbox matches whitelist patterns."""
        if not patterns:
            return False
        
        text = self.combined_text.strip()
        if not text:
            return False
        
        import re
        for pattern in patterns:
            try:
                if re.search(pattern, text):
                    return True
            except re.error:
                continue
        return False
