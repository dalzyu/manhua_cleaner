"""Geometry value objects."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True, slots=True)
class Point:
    """2D point with integer coordinates."""
    x: float
    y: float
    
    def distance_to(self, other: Point) -> float:
        """Calculate Euclidean distance to another point."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: Point) -> Point:
        return Point(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> Point:
        return Point(self.x * scalar, self.y * scalar)


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned bounding box."""
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    
    @property
    def width(self) -> float:
        return self.max_x - self.min_x
    
    @property
    def height(self) -> float:
        return self.max_y - self.min_y
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def center(self) -> Point:
        return Point(
            (self.min_x + self.max_x) / 2,
            (self.min_y + self.max_y) / 2
        )
    
    def intersects(self, other: BoundingBox) -> bool:
        """Check if this box intersects another."""
        return not (
            self.max_x < other.min_x or
            other.max_x < self.min_x or
            self.max_y < other.min_y or
            other.max_y < self.min_y
        )
    
    def union(self, other: BoundingBox) -> BoundingBox:
        """Return bounding box containing both boxes."""
        return BoundingBox(
            min(self.min_x, other.min_x),
            min(self.min_y, other.min_y),
            max(self.max_x, other.max_x),
            max(self.max_y, other.max_y)
        )
    
    def expand(self, pixels: int) -> BoundingBox:
        """Expand box by specified pixels in all directions."""
        return BoundingBox(
            self.min_x - pixels,
            self.min_y - pixels,
            self.max_x + pixels,
            self.max_y + pixels
        )


@dataclass(frozen=True, slots=True)
class Quadrilateral:
    """Four-point polygon."""
    p1: Point
    p2: Point
    p3: Point
    p4: Point
    
    def __iter__(self) -> Iterator[Point]:
        """Allow unpacking: p1, p2, p3, p4 = quad"""
        yield self.p1
        yield self.p2
        yield self.p3
        yield self.p4
    
    @property
    def points(self) -> list[Point]:
        return [self.p1, self.p2, self.p3, self.p4]
    
    @property
    def bounding_box(self) -> BoundingBox:
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        return BoundingBox(min(xs), min(ys), max(xs), max(ys))
    
    @property
    def centroid(self) -> Point:
        """Calculate centroid (average of vertices)."""
        return Point(
            sum(p.x for p in self.points) / 4,
            sum(p.y for p in self.points) / 4
        )
    
    @property
    def area(self) -> float:
        """Calculate area using shoelace formula."""
        points = self.points + [self.p1]  # Close the polygon
        n = len(points) - 1
        area = 0.0
        for i in range(n):
            area += points[i].x * points[i + 1].y
            area -= points[i + 1].x * points[i].y
        return abs(area) / 2
    
    def expand(self, pixels: int) -> Quadrilateral:
        """Expand quadrilateral outward by pixels."""
        # Calculate edge normals and expand along bisectors
        centroid = self.centroid
        new_points = []
        
        points = self.points
        n = len(points)
        for i in range(n):
            p = points[i]
            prev_p = points[(i - 1) % n]
            next_p = points[(i + 1) % n]
            
            # Vector from p to centroid
            to_center = Point(centroid.x - p.x, centroid.y - p.y)
            to_center_len = math.sqrt(to_center.x ** 2 + to_center.y ** 2)
            
            if to_center_len > 0:
                # Normalize and expand
                scale = pixels / to_center_len
                new_p = Point(
                    p.x - to_center.x * scale,
                    p.y - to_center.y * scale
                )
            else:
                new_p = p
            
            new_points.append(new_p)
        
        return Quadrilateral(*new_points)
    
    @classmethod
    def from_bbox(cls, x: float, y: float, w: float, h: float) -> Quadrilateral:
        """Create quadrilateral from bounding box."""
        return cls(
            Point(x, y),
            Point(x + w, y),
            Point(x + w, y + h),
            Point(x, y + h)
        )


@dataclass(frozen=True, slots=True)
class Mask:
    """Binary mask for image regions."""
    data: object  # numpy array
    
    @property
    def width(self) -> int:
        return self.data.shape[1]
    
    @property
    def height(self) -> int:
        return self.data.shape[0]


@dataclass(frozen=True, slots=True)
class Contour:
    """Arbitrary polygon contour."""
    points: tuple[Point, ...]
    
    def __iter__(self) -> Iterator[Point]:
        return iter(self.points)
    
    @property
    def bounding_box(self) -> BoundingBox:
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        return BoundingBox(min(xs), min(ys), max(xs), max(ys))
    
    @property
    def centroid(self) -> Point:
        """Calculate centroid."""
        if not self.points:
            return Point(0, 0)
        return Point(
            sum(p.x for p in self.points) / len(self.points),
            sum(p.y for p in self.points) / len(self.points)
        )
    
    @property
    def area(self) -> float:
        """Calculate area using shoelace formula."""
        if len(self.points) < 3:
            return 0.0
        
        points = list(self.points) + [self.points[0]]
        n = len(points) - 1
        area = 0.0
        for i in range(n):
            area += points[i].x * points[i + 1].y
            area -= points[i + 1].x * points[i].y
        return abs(area) / 2
    
    def to_quadrilateral(self) -> Quadrilateral:
        """Convert to quadrilateral using bounding box."""
        bbox = self.bounding_box
        return Quadrilateral.from_bbox(bbox.min_x, bbox.min_y, bbox.width, bbox.height)
