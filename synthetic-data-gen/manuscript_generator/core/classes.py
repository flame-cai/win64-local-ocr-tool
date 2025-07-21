from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
from shapely.geometry import Polygon, MultiPoint

@dataclass
class Point:
    """Fundamental unit: a character with local coordinates and font size."""
    x: float
    y: float
    font_size: float

@dataclass
class Word:
    """A collection of Point objects."""
    points: List[Point] = field(default_factory=list)

@dataclass
class TextLine:
    """A collection of Word objects."""
    words: List[Word] = field(default_factory=list)

@dataclass
class PageData:
    """Container for all data of a single generated page, ready for saving."""
    sample_id: str
    points_global: np.ndarray  # Shape (N, 3) for (x, y, font_size)
    labels_textbox: np.ndarray # Shape (N,)
    labels_textline: np.ndarray # Shape (N,)
    page_dims: Tuple[float, float] # (width, height)
    meta: dict = field(default_factory=dict)

@dataclass
class TextBox:
    """
    A collection of text lines with geometric properties.
    Manages the hybrid OOP-to-vectorized workflow.
    """
    text_lines: List[TextLine]
    box_type: str
    
    # Global properties (applied in Phase 3)
    position: Tuple[float, float] = (0.0, 0.0)
    orientation_deg: float = 0.0
    
    # Internal state for consolidated data
    # These are in LOCAL coordinates until transformed
    _points_local: np.ndarray | None = field(default=None, repr=False)
    _line_ids_local: np.ndarray | None = field(default=None, repr=False)
    _hull_local: Polygon | None = field(default=None, repr=False)

    def consolidate_to_numpy(self, textline_id_start: int) -> int:
        """
        Flattens the object hierarchy (TextLines -> Words -> Points) into NumPy arrays.
        This is the core of the hybrid workflow, moving from OOP to vectorized operations.
        It populates the internal `_points_local` and `_line_ids_local` attributes.

        Args:
            textline_id_start: The starting global unique ID for text lines in this box.

        Returns:
            The next available textline_id.
        """
        points_list = []
        line_ids_list = []
        current_line_id = textline_id_start

        for text_line in self.text_lines:
            # Skip empty lines that might be generated
            if not any(word.points for word in text_line.words):
                continue
                
            for word in text_line.words:
                for point in word.points:
                    points_list.append([point.x, point.y, point.font_size])
                    line_ids_list.append(current_line_id)
            current_line_id += 1

        if not points_list: # Handle empty textboxes
            self._points_local = np.empty((0, 3), dtype=np.float32)
            self._line_ids_local = np.empty((0,), dtype=np.int32)
            self._hull_local = Polygon()
            return textline_id_start
            
        self._points_local = np.array(points_list, dtype=np.float32)
        self._line_ids_local = np.array(line_ids_list, dtype=np.int32)

        # Center the points at (0,0) for stable local transformations
        mean_pos = self._points_local[:, :2].mean(axis=0)
        self._points_local[:, :2] -= mean_pos
        
        # Calculate convex hull in local coordinates
        if len(self._points_local) >= 3:
            self._hull_local = MultiPoint(self._points_local[:, :2]).convex_hull
        elif len(self._points_local) > 0:
             # Handle cases with 1 or 2 points
            self._hull_local = MultiPoint(self._points_local[:, :2]).buffer(1)
        else:
            self._hull_local = Polygon()

        return current_line_id

    @property
    def points_local(self) -> np.ndarray:
        assert self._points_local is not None, "Must call consolidate_to_numpy() first."
        return self._points_local

    @property
    def line_ids_local(self) -> np.ndarray:
        assert self._line_ids_local is not None, "Must call consolidate_to_numpy() first."
        return self._line_ids_local

    def get_global_hull(self) -> Polygon:
        """Calculates the textbox's convex hull in global page coordinates."""
        from .utils import transform_polygon # Avoid circular import
        assert self._hull_local is not None, "Must call consolidate_to_numpy() first."
        
        return transform_polygon(self._hull_local, self.position, self.orientation_deg)