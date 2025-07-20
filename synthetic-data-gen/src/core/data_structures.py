from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
from enum import StrEnum

# --- Enums for Type Safety ---

class BoxType(StrEnum):
    MAIN_TEXT = "main_text"
    MARGINALIA = "marginalia"
    PAGE_NUMBER = "page_number"
    INTERLINEAR_GLOSS = "interlinear_gloss"
    AMBIGUOUS = "ambiguous" # For special grid/concentric layouts

class TextAlignment(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"
    JUSTIFY = "justify"


# --- Core Data Structures (OOP part of the workflow) ---

@dataclass
class Point:
    """The fundamental unit. Stores unnormalized local coordinates and font size."""
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
class TextBox:
    """
    A collection of TextLine objects representing a block of text.
    Also holds metadata about the box's planned position and shape.
    This class is primarily used during the content generation phase (Phase 1).
    """
    text_lines: List[TextLine]
    box_type: BoxType
    
    # --- The "Consolidation" Step ---
    def consolidate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Traverses the object structure and flattens all point data into NumPy arrays.
        This is the bridge from the logical OOP structure to the efficient vectorized workflow.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - points_local (np.ndarray): Shape (N, 3) for (x, y, font_size).
                - line_ids_local (np.ndarray): Shape (N,) with integer IDs for each line,
                  starting from 0 for this textbox.
        """
        all_points = []
        all_line_ids = []
        
        for line_id, text_line in enumerate(self.text_lines):
            for word in text_line.words:
                for point in word.points:
                    all_points.append([point.x, point.y, point.font_size])
                    all_line_ids.append(line_id)
                    
        if not all_points:
            # Handle empty textbox case
            return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int32)
            
        points_local = np.array(all_points, dtype=np.float32)
        line_ids_local = np.array(all_line_ids, dtype=np.int32)
        
        return points_local, line_ids_local


# --- Data Structures for Post-Consolidation ---

@dataclass
class TextBoxBlueprint:
    """
    A lightweight data structure holding the planned geometric properties of a TextBox.
    Used for layout planning (e.g., collision detection) before content is generated.
    """
    box_type: BoxType
    position: Tuple[float, float]  # Center (x, y) in global coordinates
    width: float
    height: float
    orientation_deg: float
    parent_line_info: dict = field(default_factory=dict) # For interlinear gloss

    def get_obb(self) -> np.ndarray:
        """Returns the Oriented Bounding Box vertices in global coordinates."""
        cx, cy = self.position
        w, h = self.width, self.height
        angle = np.deg2rad(self.orientation_deg)
        
        # Local corners
        local_corners = np.array([
            [-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]
        ])
        
        # Rotation matrix
        R = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])
        
        # Rotate and translate
        global_corners = local_corners @ R.T + np.array([cx, cy])
        return global_corners


@dataclass
class GeneratedPage:
    """
    Final data container for a single generated sample.
    This holds all data in its final, global, pre-normalized state.
    """
    points: np.ndarray  # Shape (N, 3) for (x, y, font_size)
    textbox_labels: np.ndarray # Shape (N,)
    textline_labels: np.ndarray # Shape (N,)
    text_boxes: List[TextBoxBlueprint] # Metadata for visualization/analysis
    page_width: float
    page_height: float
    layout_strategy: str
    seed: int
    sub_id: int = 0 # To differentiate pages from the same ambiguous layout