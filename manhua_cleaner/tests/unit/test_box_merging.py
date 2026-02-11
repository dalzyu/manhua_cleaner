"""Unit tests for box merging service."""

import pytest
from manhua_cleaner.domain.services.box_merging import merge_intersecting_boxes
from manhua_cleaner.domain.value_objects.geometry import Quadrilateral, Point


class TestBoxMerging:
    """Tests for box merging algorithm."""
    
    def test_empty_list(self):
        result = merge_intersecting_boxes([])
        assert result == []
    
    def test_single_box(self):
        box = Quadrilateral.from_bbox(0, 0, 100, 100)
        result = merge_intersecting_boxes([box])
        assert len(result) == 1
        assert result[0] == box
    
    def test_non_intersecting_boxes(self):
        box1 = Quadrilateral.from_bbox(0, 0, 100, 100)
        box2 = Quadrilateral.from_bbox(200, 200, 100, 100)
        result = merge_intersecting_boxes([box1, box2])
        assert len(result) == 2
    
    def test_intersecting_boxes(self):
        box1 = Quadrilateral.from_bbox(0, 0, 100, 100)
        box2 = Quadrilateral.from_bbox(50, 50, 100, 100)  # Overlaps
        result = merge_intersecting_boxes([box1, box2])
        assert len(result) == 1
        # Result should cover both boxes
        assert result[0].area > box1.area
    
    def test_multiple_groups(self):
        # Group 1: boxes 1 and 2 intersect
        box1 = Quadrilateral.from_bbox(0, 0, 100, 100)
        box2 = Quadrilateral.from_bbox(50, 50, 100, 100)
        # Group 2: boxes 3 and 4 intersect
        box3 = Quadrilateral.from_bbox(500, 500, 100, 100)
        box4 = Quadrilateral.from_bbox(550, 550, 100, 100)
        # Not in any group
        box5 = Quadrilateral.from_bbox(1000, 1000, 100, 100)
        
        result = merge_intersecting_boxes([box1, box2, box3, box4, box5])
        assert len(result) == 3  # 2 merged groups + 1 standalone
    
    def test_chain_intersection(self):
        # A intersects B, B intersects C, but A doesn't directly intersect C
        # All should be merged into one
        box_a = Quadrilateral.from_bbox(0, 0, 100, 100)
        box_b = Quadrilateral.from_bbox(50, 50, 100, 100)
        box_c = Quadrilateral.from_bbox(100, 100, 100, 100)
        
        result = merge_intersecting_boxes([box_a, box_b, box_c])
        assert len(result) == 1
