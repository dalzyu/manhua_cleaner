"""Text grouping service - group nearby text regions."""

from __future__ import annotations

from ...domain.entities.text_region import TextRegion, Textbox
from ...domain.value_objects.geometry import Point


def group_regions(
    regions: list[TextRegion],
    max_distance: float = 50.0
) -> list[Textbox]:
    """Group nearby text regions into textboxes.
    
    Uses distance-based clustering: regions within max_distance
    of each other are grouped.
    
    Args:
        regions: Detected text regions
        max_distance: Maximum distance for grouping (pixels)
        
    Returns:
        List of textboxes containing grouped regions
    """
    if not regions:
        return []
    
    if len(regions) == 1:
        return [Textbox(regions=[regions[0]])]
    
    # Simple greedy clustering
    remaining = list(regions)
    textboxes: list[Textbox] = []
    
    while remaining:
        # Start new textbox with first remaining region
        seed = remaining.pop(0)
        textbox = Textbox(regions=[seed])
        
        # Find all regions close to this textbox
        i = 0
        while i < len(remaining):
            region = remaining[i]
            if _is_close_to_textbox(region, textbox, max_distance):
                textbox.add_region(region)
                remaining.pop(i)
            else:
                i += 1
        
        textboxes.append(textbox)
    
    return textboxes


def _is_close_to_textbox(region: TextRegion, textbox: Textbox, max_distance: float) -> bool:
    """Check if region is close to any region in textbox."""
    region_center = region.contour.centroid
    
    for tb_region in textbox.regions:
        distance = region_center.distance_to(tb_region.contour.centroid)
        if distance <= max_distance:
            return True
    
    return False


def calculate_textbox_distance(tb1: Textbox, tb2: Textbox) -> float:
    """Calculate minimum distance between two textboxes."""
    min_distance = float('inf')
    
    for r1 in tb1.regions:
        for r2 in tb2.regions:
            dist = r1.contour.centroid.distance_to(r2.contour.centroid)
            min_distance = min(min_distance, dist)
    
    return min_distance
