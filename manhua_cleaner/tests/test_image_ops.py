"""Tests for image operations."""

import numpy as np
import pytest

from ..core.image_ops import (
    expand_quadrilateral,
    expand_quadrilaterals,
    contour_to_bbox,
    merge_intersecting_boxes,
    create_blend_mask,
    get_edge_average,
    get_edge_variance,
    ensure_minimum_size,
)


class TestExpandQuadrilateral:
    """Test quadrilateral expansion."""
    
    def test_expand_square(self):
        """Test expanding a square."""
        square = np.array([
            [10, 10],
            [20, 10],
            [20, 20],
            [10, 20]
        ], dtype=np.int32)
        
        expanded = expand_quadrilateral(square, 5)
        
        # Should have 4 points
        assert expanded.shape == (4, 2)
        # Area should increase
        original_area = 10 * 10
        # Calculate expanded area using shoelace
        x, y = expanded[:, 0], expanded[:, 1]
        expanded_area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        assert expanded_area > original_area
    
    def test_expand_zero_returns_same(self):
        """Test expanding by 0 pixels returns same shape."""
        quad = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.int32)
        expanded = expand_quadrilateral(quad, 0)
        assert expanded.shape == (4, 2)


class TestExpandQuadrilaterals:
    """Test batch quadrilateral expansion."""
    
    def test_expand_multiple(self):
        """Test expanding multiple quads."""
        quads = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.int32),
            np.array([[20, 20], [30, 20], [30, 30], [20, 30]], dtype=np.int32),
        ]
        
        expanded = expand_quadrilaterals(quads, 5)
        
        assert len(expanded) == 2
        for e in expanded:
            assert e.shape == (4, 2)


class TestContourToBBox:
    """Test contour to bounding box conversion."""
    
    def test_simple_contour(self):
        """Test converting simple contour to bbox."""
        # Create contour in OpenCV format (N, 1, 2)
        contour = np.array([
            [[5, 5]],
            [[15, 5]],
            [[15, 15]],
            [[5, 15]],
        ], dtype=np.int32)
        
        bbox = contour_to_bbox(contour)
        
        # Should be 4 corners
        assert bbox.shape == (4, 2)
        # Should match bounding rect (cv2.boundingRect returns (x, y, w, h))
        # So x_max = x + w, y_max = y + h
        assert bbox[0][0] == 5   # x min
        assert bbox[0][1] == 5   # y min
        assert bbox[2][0] == 16  # x max (5 + 11, where 11 = 15 - 5 + 1)
        assert bbox[2][1] == 16  # y max (5 + 11, where 11 = 15 - 5 + 1)


class TestMergeIntersectingBoxes:
    """Test box merging."""
    
    def test_merge_intersecting(self):
        """Test that intersecting boxes are merged."""
        boxes = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.int32),
            np.array([[5, 5], [15, 5], [15, 15], [5, 15]], dtype=np.int32),  # Overlaps
        ]
        
        merged = merge_intersecting_boxes(boxes, use_bounding_rect=True)
        
        # Should be merged into 1 box
        assert len(merged) == 1
    
    def test_no_merge_separate(self):
        """Test that non-intersecting boxes are not merged."""
        boxes = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.int32),
            np.array([[100, 100], [110, 100], [110, 110], [100, 110]], dtype=np.int32),
        ]
        
        merged = merge_intersecting_boxes(boxes, use_bounding_rect=True)
        
        # Should remain 2 boxes
        assert len(merged) == 2


class TestCreateBlendMask:
    """Test blend mask creation."""
    
    def test_mask_shape(self):
        """Test mask has correct shape."""
        mask = create_blend_mask(100, 100, 10)
        assert mask.shape == (100, 100)
    
    def test_mask_values_range(self):
        """Test mask values are in [0, 1]."""
        mask = create_blend_mask(100, 100, 10)
        assert mask.min() >= 0
        assert mask.max() <= 1
    
    def test_zero_margin_returns_ones(self):
        """Test zero margin returns all ones."""
        mask = create_blend_mask(50, 50, 0)
        assert np.all(mask == 1.0)
    
    def test_center_is_one(self):
        """Test center of mask is 1."""
        mask = create_blend_mask(100, 100, 20)
        # Center should be 1
        assert mask[50, 50] == 1.0
    
    def test_edges_are_zero(self):
        """Test edges of mask are 0."""
        mask = create_blend_mask(100, 100, 20)
        # Corners should be 0
        assert mask[0, 0] == 0.0
        assert mask[0, 99] == 0.0
        assert mask[99, 0] == 0.0
        assert mask[99, 99] == 0.0


class TestGetEdgeAverage:
    """Test edge average calculation."""
    
    def test_uniform_image(self):
        """Test edge average on uniform color image."""
        # Create blue image
        img = np.full((100, 100, 3), [255, 0, 0], dtype=np.uint8)
        
        avg = get_edge_average(img)
        
        # Should be blue (use approximate equality for floating point)
        assert abs(avg[0] - 255) < 0.01  # B
        assert abs(avg[1] - 0) < 0.01    # G
        assert abs(avg[2] - 0) < 0.01    # R
    
    def test_grayscale_image(self):
        """Test on grayscale image."""
        img = np.full((50, 50), 128, dtype=np.uint8)
        
        avg = get_edge_average(img)
        
        assert abs(avg[0] - 128) < 0.01


class TestGetEdgeVariance:
    """Test edge variance calculation."""
    
    def test_uniform_zero_variance(self):
        """Test uniform image has near-zero variance."""
        img = np.full((100, 100, 3), [128, 128, 128], dtype=np.uint8)
        
        var = get_edge_variance(img)
        
        assert np.all(var < 1.0)
    
    def test_high_variance(self):
        """Test noisy image has higher variance."""
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        var = get_edge_variance(img)
        
        assert np.all(var > 0)


class TestEnsureMinimumSize:
    """Test minimum size enforcement."""
    
    def test_no_padding_needed(self):
        """Test image already larger than minimum."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result, padding = ensure_minimum_size(img, min_size=64)
        
        assert result.shape == img.shape
        assert padding == (0, 0, 0, 0)
    
    def test_padding_applied(self):
        """Test padding is added for small image."""
        img = np.zeros((30, 30, 3), dtype=np.uint8)
        
        result, padding = ensure_minimum_size(img, min_size=64)
        
        assert result.shape[0] >= 64
        assert result.shape[1] >= 64
        assert sum(padding) > 0
    
    def test_padding_symmetric(self):
        """Test padding is roughly symmetric."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        
        result, (top, bottom, left, right) = ensure_minimum_size(img, min_size=64)
        
        # Padding should be roughly equal on both sides
        assert abs(top - bottom) <= 1
        assert abs(left - right) <= 1
