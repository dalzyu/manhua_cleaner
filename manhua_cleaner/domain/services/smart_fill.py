"""Smart fill service - fill simple backgrounds without AI."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ...domain.entities.image import Image
from ...domain.value_objects.geometry import Quadrilateral

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SmartFillResult:
    """Result of smart fill operation."""
    filled_indices: list[int]
    remaining_indices: list[int]


def apply_smart_fill(
    image: Image,
    boxes: list[Quadrilateral],
    threshold: float = 10.0
) -> tuple[Image, list[Quadrilateral], list[int]]:
    """Apply smart fill to boxes with simple backgrounds.
    
    Args:
        image: Image to process
        boxes: Regions to potentially fill
        threshold: Color variance threshold (lower = stricter)
        
    Returns:
        Tuple of (modified_image, remaining_boxes, filled_indices)
    """
    remaining: list[Quadrilateral] = []
    filled: list[int] = []
    
    # Convert to numpy for processing
    img_array = image.to_array()
    
    for i, box in enumerate(boxes):
        bbox = box.bounding_box
        x = int(bbox.min_x)
        y = int(bbox.min_y)
        w = int(bbox.width)
        h = int(bbox.height)
        
        # Clamp to image bounds
        img_h, img_w = img_array.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        if w <= 0 or h <= 0:
            continue
        
        crop = img_array[y:y+h, x:x+w]
        
        # Check if background is simple
        variance = _calculate_edge_variance(crop)
        
        if variance < threshold:
            # Fill with average color
            avg_color = _calculate_edge_average(crop)
            img_array[y:y+h, x:x+w] = avg_color
            filled.append(i)
            logger.debug(f"Smart filled box {i}: variance={variance:.2f}")
        else:
            remaining.append(box)
    
    # Convert back to Image
    from ...domain.entities.image import Image as ImageEntity
    result_image = ImageEntity.from_array(img_array, image.source_path)
    
    return result_image, remaining, filled


def _calculate_edge_variance(crop: object) -> float:
    """Calculate variance along edges of crop region."""
    import numpy as np
    
    h, w = crop.shape[:2]
    if h < 2 or w < 2:
        return 0.0
    
    # Sample edges (top, bottom, left, right)
    edges = [
        crop[0, :],      # Top
        crop[-1, :],     # Bottom
        crop[:, 0],      # Left
        crop[:, -1],     # Right
    ]
    
    # Calculate variance
    edge_array = np.concatenate([e.flatten() for e in edges])
    return float(np.var(edge_array))


def _calculate_edge_average(crop: object) -> object:
    """Calculate average color along edges."""
    import numpy as np
    
    h, w = crop.shape[:2]
    if h < 2 or w < 2:
        return crop[0, 0] if crop.size > 0 else 0
    
    # Sample edges
    edges = [
        crop[0, :],
        crop[-1, :],
        crop[:, 0],
        crop[:, -1],
    ]
    
    edge_array = np.concatenate([e.flatten() for e in edges])
    
    # Return mean with same shape as original pixels
    if len(crop.shape) == 3:
        return np.mean(edge_array.reshape(-1, crop.shape[2]), axis=0).astype(crop.dtype)
    else:
        return np.mean(edge_array).astype(crop.dtype)
