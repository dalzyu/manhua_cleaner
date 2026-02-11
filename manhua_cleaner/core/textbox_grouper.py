"""Spatial grouping of text regions into logical textboxes."""

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .text_region import TextRegion, Textbox

logger = logging.getLogger(__name__)


class TextboxGrouper:
    """Group nearby text regions into logical textboxes using spatial clustering."""
    
    def __init__(self, max_distance: float = 50.0):
        """
        Initialize the textbox grouper.
        
        Args:
            max_distance: Maximum distance (in pixels) between region centers
                         for them to be considered part of the same textbox.
        """
        self.max_distance = max_distance
    
    def group_regions(self, regions: list["TextRegion"]) -> list["Textbox"]:
        """
        Group text regions into logical textboxes.
        
        Uses DBSCAN clustering based on spatial distance between region centers.
        
        Args:
            regions: List of detected text regions.
            
        Returns:
            List of textboxes, each containing one or more regions.
        """
        from .text_region import Textbox
        
        if not regions:
            return []
        
        if len(regions) == 1:
            return [Textbox(
                regions=[regions[0]],
                combined_text=regions[0].text
            )]
        
        # Get centers of all regions
        centers = np.array([r.center for r in regions])
        
        try:
            from sklearn.cluster import DBSCAN
            
            # Use DBSCAN for spatial clustering
            clustering = DBSCAN(
                eps=self.max_distance,
                min_samples=1,  # Allow single-region clusters
                metric='euclidean'
            ).fit(centers)
            
            labels = clustering.labels_
            
        except ImportError:
            # Fallback: simple distance-based grouping without sklearn
            logger.debug("sklearn not available, using fallback grouping")
            labels = self._fallback_grouping(centers)
        
        # Group regions by cluster label
        textboxes = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            cluster_indices = [i for i, l in enumerate(labels) if l == label]
            cluster_regions = [regions[i] for i in cluster_indices]
            
            # Combine text from all regions (sorted by y position for reading order)
            cluster_regions.sort(key=lambda r: r.center[1])
            combined_text = " ".join(r.text.strip() for r in cluster_regions if r.text.strip())
            
            textboxes.append(Textbox(
                regions=cluster_regions,
                combined_text=combined_text
            ))
        
        logger.debug(f"Grouped {len(regions)} regions into {len(textboxes)} textboxes")
        return textboxes
    
    def _fallback_grouping(self, centers: np.ndarray) -> np.ndarray:
        """
        Simple distance-based grouping fallback when sklearn is not available.
        
        Uses a greedy approach: start with first point, group all points within
        max_distance, repeat for ungrouped points.
        """
        n = len(centers)
        labels = np.full(n, -1)
        current_label = 0
        
        for i in range(n):
            if labels[i] != -1:
                continue
            
            # Start new cluster
            labels[i] = current_label
            cluster_points = [i]
            
            # Find all points within max_distance
            changed = True
            while changed:
                changed = False
                for j in range(n):
                    if labels[j] != -1:
                        continue
                    
                    # Check distance to any point in cluster
                    for cp_idx in cluster_points:
                        dist = np.linalg.norm(centers[j] - centers[cp_idx])
                        if dist <= self.max_distance:
                            labels[j] = current_label
                            cluster_points.append(j)
                            changed = True
                            break
            
            current_label += 1
        
        return labels


def union_bbox(bboxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    """
    Compute the union of multiple bounding boxes.
    
    Args:
        bboxes: List of bounding boxes as (x, y, w, h).
        
    Returns:
        Union bounding box as (x, y, w, h).
    """
    if not bboxes:
        return (0, 0, 0, 0)
    
    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[0] + b[2] for b in bboxes)
    y2 = max(b[1] + b[3] for b in bboxes)
    
    return (x1, y1, x2 - x1, y2 - y1)
