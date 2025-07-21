"""
Core Package Initializer

This __init__.py file serves two purposes:
1. Marks the 'core' directory as a Python package.
2. Exposes the most important classes and functions from submodules directly
   at the package level for easier access elsewhere in the application.
"""

# Expose core data classes
from .classes import Point, Word, TextLine, TextBox, PageData

# Expose registries for pluggable components
from .registries import LAYOUT_STRATEGIES, AUGMENTATIONS, register_layout, register_augmentation

# Expose utility functions
from .distributions import sample_from_config
from .utils import transform_points, transform_polygon, check_overlap, check_bounds, normalize_page_data

__all__ = [
    "Point",
    "Word",
    "TextLine",
    "TextBox",
    "PageData",
    "LAYOUT_STRATEGIES",
    "AUGMENTATIONS",
    "register_layout",
    "register_augmentation",
    "sample_from_config",
    "transform_points",
    "transform_polygon",
    "check_overlap",
    "check_bounds",
    "normalize_page_data",
]