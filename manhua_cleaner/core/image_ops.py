"""Image processing operations for text removal."""

import logging
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image
from shapely.geometry import Polygon

from ..config import IMAGE_CONFIG

logger = logging.getLogger(__name__)

# Type aliases
Contour = npt.NDArray[np.int32]  # Shape (N, 1, 2)
Quadrilateral = npt.NDArray[np.int32]  # Shape (4, 2)
ImageArray = npt.NDArray[np.uint8]  # HxW or HxWx3


def expand_quadrilateral(quad: Quadrilateral, px: int) -> Quadrilateral:
    """Expand quadrilateral outward by px pixels.
    
    Uses vector math to expand each vertex along the angle bisector
    of adjacent edges.
    
    Args:
        quad: (4, 2) array of points
        px: Number of pixels to expand (negative to shrink)
        
    Returns:
        Expanded quadrilateral as (4, 2) array
    """
    quad = np.array(quad, dtype=np.float32).reshape(4, 2)
    centroid = np.mean(quad, axis=0)
    
    # Sort points by angle from centroid for correct edge pairing
    angles = np.arctan2(quad[:, 1] - centroid[1], quad[:, 0] - centroid[0])
    quad = quad[np.argsort(angles)]
    
    expanded = np.zeros_like(quad)
    n = len(quad)
    
    for i in range(n):
        p1 = quad[i]
        p2 = quad[(i + 1) % n]
        p0 = quad[(i - 1) % n]
        
        # Current edge vector
        edge = p2 - p1
        edge_len = np.linalg.norm(edge)
        if edge_len == 0:
            expanded[i] = p1
            continue
        
        # Unit normal pointing outward
        normal = np.array([edge[1], -edge[0]]) / edge_len
        
        # Check if normal points outward (away from centroid)
        mid = (p1 + p2) / 2
        to_centroid = centroid - mid
        if np.dot(normal, to_centroid) > 0:
            normal = -normal
        
        # Calculate bisector of adjacent edges
        prev_edge = p1 - p0
        prev_len = np.linalg.norm(prev_edge)
        
        if prev_len > 0:
            prev_normal = np.array([prev_edge[1], -prev_edge[0]]) / prev_len
            if np.dot(prev_normal, centroid - (p0 + p1) / 2) > 0:
                prev_normal = -prev_normal
            
            # Move vertex along bisector
            bisector = normal + prev_normal
            bisector_len = np.linalg.norm(bisector)
            
            if bisector_len > 0:
                bisector = bisector / bisector_len
                cos_half = np.dot(normal, bisector)
                
                if abs(cos_half) > IMAGE_CONFIG.min_cos_half_threshold:
                    expanded[i] = p1 + (px / cos_half) * bisector
                else:
                    expanded[i] = p1 + px * normal
            else:
                expanded[i] = p1 + px * normal
        else:
            expanded[i] = p1 + px * normal
    
    return expanded.astype(np.int32)


def expand_quadrilaterals(
    quads: list[Quadrilateral],
    px: int
) -> list[Quadrilateral]:
    """Expand multiple quadrilaterals."""
    return [expand_quadrilateral(q, px) for q in quads]


def quads_intersect(q1: Quadrilateral, q2: Quadrilateral) -> Optional[Contour]:
    """Check if two quads intersect and return their union if they do.
    
    Args:
        q1: First quadrilateral
        q2: Second quadrilateral
        
    Returns:
        Union contour if they intersect, None otherwise
    """
    q1_arr = np.array(q1).reshape(-1, 2)
    q2_arr = np.array(q2).reshape(-1, 2)
    
    poly1 = Polygon(q1_arr)
    poly2 = Polygon(q2_arr)
    
    if poly1.intersects(poly2):
        union = np.array(list(poly1.union(poly2).exterior.coords))
        return union.reshape(-1, 1, 2).astype(np.int32)
    
    return None


def contour_to_bbox(contour: Contour) -> Quadrilateral:
    """Convert contour to bounding box quadrilateral.
    
    Args:
        contour: OpenCV contour
        
    Returns:
        Quadrilateral as (4, 2) array
    """
    x, y, w, h = cv2.boundingRect(contour)
    return np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h]
    ], dtype=np.int32)


def _get_bounding_box(quad: Quadrilateral) -> tuple[float, float, float, float]:
    """Get bounding box coordinates for a quadrilateral.
    
    Args:
        quad: Quadrilateral as (4, 2) array
        
    Returns:
        Tuple of (min_x, min_y, max_x, max_y)
    """
    return (float(quad[:, 0].min()), float(quad[:, 1].min()),
            float(quad[:, 0].max()), float(quad[:, 1].max()))


def _boxes_intersect_aabb(box1: tuple[float, float, float, float],
                          box2: tuple[float, float, float, float]) -> bool:
    """Check if two axis-aligned bounding boxes intersect.
    
    Args:
        box1: (min_x, min_y, max_x, max_y)
        box2: (min_x, min_y, max_x, max_y)
        
    Returns:
        True if boxes intersect
    """
    return not (box1[2] < box2[0] or box2[2] < box1[0] or
                box1[3] < box2[1] or box2[3] < box1[1])


def merge_intersecting_boxes(
    boxes: list[Quadrilateral],
    use_bounding_rect: bool = True
) -> list[Quadrilateral]:
    """Merge intersecting bounding boxes using spatial indexing.
    
    This implementation uses a sweep-line algorithm with AABB (axis-aligned
    bounding box) checks to reduce complexity from O(n^3) to O(n log n).
    
    Args:
        boxes: List of quadrilaterals
        use_bounding_rect: Use bounding rectangle for merged result
        
    Returns:
        List of merged boxes
    """
    if not boxes:
        return []
    
    n = len(boxes)
    if n == 1:
        return [boxes[0].copy()]
    
    # Compute bounding boxes for all quads
    bboxes = [_get_bounding_box(q) for q in boxes]
    
    # Union-Find data structure for connected components
    parent = list(range(n))
    
    def find(x: int) -> int:
        """Find root with path compression."""
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x: int, y: int) -> None:
        """Union two sets."""
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Sort by min_x for sweep-line algorithm
    sorted_indices = sorted(range(n), key=lambda i: bboxes[i][0])
    
    # Sweep-line: check nearby boxes for intersection
    for i, idx1 in enumerate(sorted_indices):
        box1 = bboxes[idx1]
        # Check all boxes that start before box1 ends (potential overlaps)
        for j in range(i + 1, len(sorted_indices)):
            idx2 = sorted_indices[j]
            box2 = bboxes[idx2]
            
            # If box2 starts after box1 ends, no overlap possible for remaining boxes
            if box2[0] > box1[2]:
                break
            
            # AABB check first (fast rejection)
            if _boxes_intersect_aabb(box1, box2):
                # Detailed intersection check using Shapely
                result = quads_intersect(boxes[idx1], boxes[idx2])
                if result is not None:
                    union(idx1, idx2)
    
    # Group boxes by their connected component
    components: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        if root not in components:
            components[root] = []
        components[root].append(i)
    
    # Merge each component
    result: list[Quadrilateral] = []
    for indices in components.values():
        if len(indices) == 1:
            result.append(boxes[indices[0]].copy())
        else:
            # Merge all boxes in this component
            merged_poly = None
            for idx in indices:
                quad = boxes[idx]
                quad_arr = np.array(quad).reshape(-1, 2)
                poly = Polygon(quad_arr)
                if merged_poly is None:
                    merged_poly = poly
                else:
                    merged_poly = merged_poly.union(poly)
            
            if merged_poly is not None:
                # Convert back to quadrilateral format
                if hasattr(merged_poly, 'exterior'):
                    if use_bounding_rect:
                        # Get bounding rectangle of merged shape
                        coords = np.array(list(merged_poly.exterior.coords)[:-1])
                        coords = coords.astype(np.int32)
                        coords = contour_to_bbox(coords.reshape(-1, 1, 2))
                    else:
                        # Use convex hull to get simplified polygon
                        # This preserves the general shape better than bounding rect
                        convex_hull = merged_poly.convex_hull
                        hull_coords = np.array(list(convex_hull.exterior.coords)[:-1])
                        # Ensure we have at most 4 points for quadrilateral
                        if len(hull_coords) > 4:
                            # Use bounding box as fallback for complex shapes
                            hull_coords = hull_coords.astype(np.int32)
                            coords = contour_to_bbox(hull_coords.reshape(-1, 1, 2))
                        else:
                            coords = hull_coords.astype(np.int32)
                    result.append(coords)
    
    return result


def clamp_contour_to_image(
    contour: Contour,
    img_width: int,
    img_height: int
) -> Contour:
    """Clamp contour points to image bounds.
    
    Args:
        contour: Contour to clamp
        img_width: Image width
        img_height: Image height
        
    Returns:
        Clamped contour
    """
    clamped = contour.copy()
    clamped[:, 0, 0] = np.clip(clamped[:, 0, 0], 0, img_width - 1)
    clamped[:, 0, 1] = np.clip(clamped[:, 0, 1], 0, img_height - 1)
    return clamped


def crop_to_contour(img: ImageArray, contour: Contour) -> ImageArray:
    """Crop image to contour's bounding box.
    
    Args:
        img: Input image
        contour: Contour defining region
        
    Returns:
        Cropped image region
    """
    x, y, w, h = cv2.boundingRect(contour)
    return img[y:y+h, x:x+w]


def get_edge_average(img: ImageArray, border_width: int = 1) -> np.ndarray:
    """Get average pixel value along image edges.
    
    Args:
        img: Input image
        border_width: Width of edge border to sample
        
    Returns:
        Average BGR values (or single value for grayscale)
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Create mask for borders
    mask[:border_width, :] = 255  # Top
    mask[-border_width:, :] = 255  # Bottom
    mask[:, :border_width] = 255  # Left
    mask[:, -border_width:] = 255  # Right
    
    mean_val = cv2.mean(img, mask=mask)
    
    # Return BGR for color, single value for grayscale
    if len(img.shape) == 3:
        return np.array(mean_val[:3])
    return np.array([mean_val[0]])


def get_edge_variance(img: ImageArray, border_width: int = 1) -> np.ndarray:
    """Get variance of pixel values along image edges.
    
    Args:
        img: Input image
        border_width: Width of edge border to sample
        
    Returns:
        Variance of BGR values (or single value for grayscale)
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Create mask for borders
    mask[:border_width, :] = 255
    mask[-border_width:, :] = 255
    mask[:, :border_width] = 255
    mask[:, -border_width:] = 255
    
    _, stddev = cv2.meanStdDev(img, mask=mask)
    
    variance = stddev ** 2
    if len(img.shape) == 3:
        return variance[:3].flatten()
    return np.array([variance[0][0]])


def create_blend_mask(
    height: int,
    width: int,
    margin: int
) -> npt.NDArray[np.float32]:
    """Create feathered alpha mask for edge blending.
    
    Creates a mask where edges fade from 0 (use original) 
    to 1 (use inpainted).
    
    Args:
        height: Mask height
        width: Mask width
        margin: Blend margin in pixels
        
    Returns:
        Float mask of shape (height, width)
    """
    if margin <= 0:
        return np.ones((height, width), dtype=np.float32)
    
    mask = np.ones((height, width), dtype=np.float32)
    
    for i in range(margin):
        fade = float(i) / float(margin)
        
        # Top edge
        if i < height:
            mask[i, :] = np.minimum(mask[i, :], fade)
        # Bottom edge
        if height - 1 - i >= 0:
            mask[height - 1 - i, :] = np.minimum(mask[height - 1 - i, :], fade)
        # Left edge
        if i < width:
            mask[:, i] = np.minimum(mask[:, i], fade)
        # Right edge
        if width - 1 - i >= 0:
            mask[:, width - 1 - i] = np.minimum(mask[:, width - 1 - i], fade)
    
    return mask


def ensure_minimum_size(
    img: ImageArray,
    min_size: int = IMAGE_CONFIG.min_size
) -> tuple[ImageArray, tuple[int, int, int, int]]:
    """Pad image to ensure minimum dimensions.
    
    Args:
        img: Input image
        min_size: Minimum dimension required
        
    Returns:
        Tuple of (padded_image, (pad_top, pad_bottom, pad_left, pad_right))
    """
    h, w = img.shape[:2]
    
    pad_w = max(0, min_size - w)
    pad_h = max(0, min_size - h)
    
    if pad_w == 0 and pad_h == 0:
        return img, (0, 0, 0, 0)
    
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    
    padded = cv2.copyMakeBorder(
        img,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_REPLICATE
    )
    
    return padded, (pad_top, pad_bottom, pad_left, pad_right)

def contour_to_rect(contour: Contour) -> Quadrilateral:
    """Convert any quadrilateral to a aligned rectangle using bounding box.
    
    Args:
        contour: Input contour
        
    Returns:
        Quadrilateral as (4, 2) array
    """
    x, y, w, h = cv2.boundingRect(contour)
    return np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h]
    ], dtype=np.int32)
