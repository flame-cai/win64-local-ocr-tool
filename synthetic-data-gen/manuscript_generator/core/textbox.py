# /manuscript_generator/core/textbox.py

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from shapely.geometry import Polygon

from manuscript_generator.configs.base_config import Config
from manuscript_generator.core.common import Point, Word, TextLine, TextBoxType, TextAlignment
from manuscript_generator.utils.distribution_sampler import sample_from_distribution
from manuscript_generator.utils.geometry import get_convex_hull, get_rotation_matrix, apply_transform
from manuscript_generator.core.registry import AUGMENTATIONS

@dataclass
class TextBox:
    """
    Represents a textbox. It's initially generated in local coordinates,
    then consolidated into NumPy arrays for efficient augmentation and placement.
    """
    box_type: TextBoxType
    text_lines: List[TextLine] = field(default_factory=list)
    
    # Global properties (applied in Phase 3)
    position: Tuple[float, float] = (0, 0)
    orientation_deg: float = 0.0
    
    # Internal state for processing
    points_local: Optional[np.ndarray] = None
    line_ids_local: Optional[np.ndarray] = None
    width: Optional[float] = None
    height: Optional[float] = None
    
    # Final state in global coordinates
    points_global: Optional[np.ndarray] = None
    hull_global: Optional[Polygon] = None

    def consolidate_to_numpy(self):
        """
        The "Consolidation" Step.
        Traverses the object structure (TextLine -> Word -> Point) and flattens
        all character data into efficient NumPy arrays in local coordinates.
        Also calculates initial width and height.
        """
        points_list = []
        line_ids_list = []
        
        for line_idx, text_line in enumerate(self.text_lines):
            for word in text_line.words:
                for point in word.points:
                    points_list.append([point.x, point.y, point.font_size])
                    line_ids_list.append(line_idx)
        
        if not points_list:
            self.points_local = np.empty((0, 3))
            self.line_ids_local = np.empty((0,), dtype=int)
            self.width = 0
            self.height = 0
            return

        self.points_local = np.array(points_list, dtype=np.float64)
        self.line_ids_local = np.array(line_ids_list, dtype=int)

        # Center the points at (0,0) in their local coordinate system
        min_coords = np.min(self.points_local[:, :2], axis=0)
        max_coords = np.max(self.points_local[:, :2], axis=0)
        self.width = max_coords[0] - min_coords[0]
        self.height = max_coords[1] - min_coords[1]
        
        center_offset = (min_coords + max_coords) / 2.0
        self.points_local[:, :2] -= center_offset
        
        assert self.points_local.shape[0] == self.line_ids_local.shape[0]
        assert self.points_local.shape[1] == 3

    def apply_augmentations(self, config: Config, rng: np.random.Generator):
        """Applies Phase 1 and 2 augmentations to the local point cloud."""
        # --- Phase 1: Content & Micro-Variations ---
        for aug_name in ["font_size_variation", "point_level_jitter", "congestion_jitter"]:
            aug_config = getattr(config.textbox_content, aug_name)
            if aug_config['enabled']:
                self.points_local = AUGMENTATIONS[aug_name](self.points_local, self.line_ids_local, config, rng)

        # --- Phase 2: Geometric Distortion ---
        distort_config = config.textbox_distortion
        # Apply in random order for more variety
        distortions = list(distort_config.model_dump().keys())
        rng.shuffle(distortions)

        for aug_name in distortions:
            aug_config = getattr(distort_config, aug_name)
            if aug_config.enabled and rng.random() < aug_config.probability:
                self.points_local = AUGMENTATIONS[aug_name](self.points_local, self, config, rng)
    
    def transform_to_global(self):
        """
        Applies rotation and translation to move points from local to global
        page coordinates. Must be called after setting `position` and `orientation_deg`.
        """
        assert self.points_local is not None, "Must consolidate before transforming."
        
        # 1. Rotate
        rot_matrix = get_rotation_matrix(self.orientation_deg)
        rotated_points = apply_transform(self.points_local, rot_matrix)
        
        # 2. Translate
        rotated_points[:, :2] += self.position
        self.points_global = rotated_points

        # 3. Update convex hull in global coordinates
        self.hull_global = get_convex_hull(self.points_global)

def _create_text_lines(box_config: dict, content_config: Config, rng: np.random.Generator) -> Tuple[List[TextLine], float, float]:
    """Helper to generate the raw text lines for a textbox."""
    lines_per_box = sample_from_distribution(box_config.lines_per_box, rng)
    base_font_size = sample_from_distribution(box_config.font_size, rng)
    
    char_spacing = base_font_size * sample_from_distribution(content_config.character_spacing_factor, rng)
    word_spacing = base_font_size * sample_from_distribution(content_config.word_spacing_factor, rng)
    line_spacing = base_font_size * sample_from_distribution(content_config.line_spacing_factor, rng)

    text_lines = []
    max_line_width = 0
    current_y = 0

    for _ in range(lines_per_box):
        words_per_line = sample_from_distribution(box_config.words_per_line, rng)
        current_x = 0
        words = []
        
        for _ in range(words_per_line):
            # For simplicity, we model a word as a single point cloud.
            # In a real scenario, this would come from a text corpus.
            # Here, we just generate a random number of characters.
            # chars_in_word = rng.integers(3, 10)
            chars_in_word = sample_from_distribution(content_config.chars_per_word, rng)
            points = []
            for _ in range(chars_in_word):
                # We use the base font size here; variation is an augmentation.
                points.append(Point(x=current_x, y=current_y, font_size=base_font_size))
                current_x += char_spacing
            
            words.append(Word(points=points))
            current_x += word_spacing
        
        text_lines.append(TextLine(words=words))
        if (current_x - word_spacing) > max_line_width:
            max_line_width = current_x - word_spacing

        current_y -= line_spacing # Move to the next line (y decreases downwards)
        
    return text_lines, max_line_width, (abs(current_y) - line_spacing)


def _apply_text_alignment(text_lines: List[TextLine], alignment: TextAlignment, box_width: float, content_config: Config, rng: np.random.Generator):
    """Applies alignment to the text lines *in place*."""
    if alignment == TextAlignment.LEFT:
        return # Default is left-aligned

    for line in text_lines:
        if not line.words or not line.words[-1].points:
            continue
        
        # Calculate the natural width of the line
        first_point = line.words[0].points[0]
        last_point = line.words[-1].points[-1]
        line_width = last_point.x - first_point.x

        offset = 0
        if alignment == TextAlignment.RIGHT:
            offset = box_width - line_width
        elif alignment == TextAlignment.CENTER:
            offset = (box_width - line_width) / 2
        elif alignment == TextAlignment.JUSTIFY:
            slack = box_width - line_width
            if slack > 0 and len(line.words) > 1:
                # Distribute slack by increasing word spacing
                extra_space_per_gap = slack / (len(line.words) - 1)
                cumulative_extra_space = 0
                for i in range(1, len(line.words)):
                    cumulative_extra_space += extra_space_per_gap
                    for point in line.words[i].points:
                        point.x += cumulative_extra_space
            continue # Justify logic is self-contained

        # Apply offset to all points in the line
        for word in line.words:
            for point in word.points:
                point.x += offset

def create_textbox(box_type: TextBoxType, config: Config, rng: np.random.Generator) -> TextBox:
    """Factory function to create a fully formed TextBox object."""
    content_config = config.textbox_content
    box_specific_config = getattr(content_config, box_type.value)
    
    # 1. Generate text content (Phase 1 pre-cursor)
    text_lines, max_line_width, _ = _create_text_lines(box_specific_config, content_config, rng)
    
    # 2. Apply text alignment
    alignment = TextAlignment(sample_from_distribution(box_specific_config.alignment, rng))
    _apply_text_alignment(text_lines, alignment, max_line_width, content_config, rng)
    
    # 3. Create TextBox object and consolidate to numpy
    textbox = TextBox(box_type=box_type, text_lines=text_lines)
    textbox.consolidate_to_numpy()
    
    # 4. Apply Phase 1 and 2 augmentations
    if textbox.points_local.shape[0] > 0:
        textbox.apply_augmentations(config, rng)
        
    return textbox