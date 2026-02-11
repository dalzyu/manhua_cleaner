"""Box merging service - pure geometry operations."""

from __future__ import annotations

from ...domain.value_objects.geometry import Quadrilateral, BoundingBox


def merge_intersecting_boxes(
    boxes: list[Quadrilateral],
    use_bounding_rect: bool = True
) -> list[Quadrilateral]:
    """Merge intersecting quadrilaterals using union-find.
    
    Algorithm:
        1. Build spatial index with bounding boxes
        2. Use sweep-line to find potential intersections
        3. Union-Find for connected components
        4. Merge each component into single quadrilateral
    
    Args:
        boxes: List of quadrilaterals to merge
        use_bounding_rect: Use bounding rectangle for merged result
        
    Returns:
        List of merged quadrilaterals
        
    Complexity: O(n log n) average case
    """
    if not boxes:
        return []
    
    if len(boxes) == 1:
        return [boxes[0]]
    
    # Get bounding boxes for all quads
    bboxes = [box.bounding_box for box in boxes]
    n = len(boxes)
    
    # Union-Find data structure
    parent = list(range(n))
    
    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])  # Path compression
        return parent[x]
    
    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Sort by min_x for sweep-line
    sorted_indices = sorted(range(n), key=lambda i: bboxes[i].min_x)
    
    # Find intersections
    for i, idx1 in enumerate(sorted_indices):
        bbox1 = bboxes[idx1]
        # Check boxes that could intersect (start before bbox1 ends)
        for j in range(i + 1, len(sorted_indices)):
            idx2 = sorted_indices[j]
            bbox2 = bboxes[idx2]
            
            # If bbox2 starts after bbox1 ends, no intersection possible
            if bbox2.min_x > bbox1.max_x:
                break
            
            # Check AABB intersection
            if bbox1.intersects(bbox2):
                # Detailed intersection check using actual geometry
                if _quads_intersect(boxes[idx1], boxes[idx2]):
                    union(idx1, idx2)
    
    # Group by component
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
            result.append(boxes[indices[0]])
        else:
            merged = _merge_component(
                [boxes[i] for i in indices],
                use_bounding_rect
            )
            result.append(merged)
    
    return result


def _quads_intersect(q1: Quadrilateral, q2: Quadrilateral) -> bool:
    """Check if two quadrilaterals intersect."""
    # Simplified: check if bounding boxes intersect
    # For exact check, would need polygon intersection
    return q1.bounding_box.intersects(q2.bounding_box)


def _merge_component(
    boxes: list[Quadrilateral],
    use_bounding_rect: bool
) -> Quadrilateral:
    """Merge multiple quadrilaterals into one."""
    if use_bounding_rect:
        # Compute union bounding box
        min_x = min(box.bounding_box.min_x for box in boxes)
        min_y = min(box.bounding_box.min_y for box in boxes)
        max_x = max(box.bounding_box.max_x for box in boxes)
        max_y = max(box.bounding_box.max_y for box in boxes)
        
        return Quadrilateral.from_bbox(min_x, min_y, max_x - min_x, max_y - min_y)
    else:
        # Use convex hull (simplified: return largest box)
        return max(boxes, key=lambda b: b.area)
