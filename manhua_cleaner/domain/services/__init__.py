"""Domain services - pure business logic, no external dependencies."""

from .box_merging import merge_intersecting_boxes
from .smart_fill import apply_smart_fill
from .text_grouping import group_regions

__all__ = ['merge_intersecting_boxes', 'apply_smart_fill', 'group_regions']
