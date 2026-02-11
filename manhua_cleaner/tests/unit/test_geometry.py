"""Unit tests for geometry value objects."""

import pytest
from manhua_cleaner.domain.value_objects.geometry import (
    Point, BoundingBox, Quadrilateral, Contour
)


class TestPoint:
    """Tests for Point class."""
    
    def test_creation(self):
        p = Point(10, 20)
        assert p.x == 10
        assert p.y == 20
    
    def test_distance_to(self):
        p1 = Point(0, 0)
        p2 = Point(3, 4)
        assert p1.distance_to(p2) == 5.0  # 3-4-5 triangle
    
    def test_addition(self):
        p1 = Point(1, 2)
        p2 = Point(3, 4)
        result = p1 + p2
        assert result == Point(4, 6)
    
    def test_subtraction(self):
        p1 = Point(5, 5)
        p2 = Point(2, 3)
        result = p1 - p2
        assert result == Point(3, 2)
    
    def test_scalar_multiplication(self):
        p = Point(2, 3)
        result = p * 2
        assert result == Point(4, 6)


class TestBoundingBox:
    """Tests for BoundingBox class."""
    
    def test_creation(self):
        box = BoundingBox(0, 0, 100, 100)
        assert box.min_x == 0
        assert box.max_x == 100
        assert box.width == 100
        assert box.height == 100
        assert box.area == 10000
    
    def test_center(self):
        box = BoundingBox(0, 0, 100, 100)
        center = box.center
        assert center == Point(50, 50)
    
    def test_intersects(self):
        box1 = BoundingBox(0, 0, 100, 100)
        box2 = BoundingBox(50, 50, 150, 150)
        assert box1.intersects(box2)
    
    def test_no_intersection(self):
        box1 = BoundingBox(0, 0, 100, 100)
        box2 = BoundingBox(200, 200, 300, 300)
        assert not box1.intersects(box2)
    
    def test_union(self):
        box1 = BoundingBox(0, 0, 100, 100)
        box2 = BoundingBox(50, 50, 150, 150)
        union = box1.union(box2)
        assert union == BoundingBox(0, 0, 150, 150)
    
    def test_expand(self):
        box = BoundingBox(50, 50, 100, 100)
        expanded = box.expand(10)
        assert expanded == BoundingBox(40, 40, 110, 110)


class TestQuadrilateral:
    """Tests for Quadrilateral class."""
    
    def test_creation(self):
        q = Quadrilateral(
            Point(0, 0),
            Point(100, 0),
            Point(100, 100),
            Point(0, 100)
        )
        assert q.area == 10000  # Square 100x100
    
    def test_from_bbox(self):
        q = Quadrilateral.from_bbox(10, 20, 100, 50)
        assert q.area == 5000  # 100 * 50
        assert q.bounding_box == BoundingBox(10, 20, 110, 70)
    
    def test_centroid(self):
        q = Quadrilateral(
            Point(0, 0),
            Point(100, 0),
            Point(100, 100),
            Point(0, 100)
        )
        assert q.centroid == Point(50, 50)
    
    def test_iter_unpacking(self):
        q = Quadrilateral(
            Point(0, 0),
            Point(10, 0),
            Point(10, 10),
            Point(0, 10)
        )
        p1, p2, p3, p4 = q
        assert p1 == Point(0, 0)
        assert p4 == Point(0, 10)


class TestContour:
    """Tests for Contour class."""
    
    def test_creation(self):
        c = Contour((
            Point(0, 0),
            Point(10, 0),
            Point(10, 10),
            Point(0, 10)
        ))
        assert c.area == 100
    
    def test_to_quadrilateral(self):
        c = Contour((
            Point(0, 0),
            Point(10, 0),
            Point(10, 10),
            Point(0, 10)
        ))
        q = c.to_quadrilateral()
        assert q.area == 100
    
    def test_empty_contour(self):
        c = Contour(())
        assert c.area == 0.0
