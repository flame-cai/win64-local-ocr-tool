import numpy as np
from typing import List, Tuple

from ..core.classes import Point, Word, TextLine, TextBox
from ..core.distributions import sample_from_config
from ..core.registries import AUGMENTATIONS
from ..config import AppConfig

class TextBoxGenerator:
    """
    Handles the generation of a single TextBox, from content creation (Phase 1)
    to geometric distortion (Phase 2).
    """
    def __init__(self, config: AppConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng

    def generate(self, box_type: str, textline_id_start: int) -> TextBox:
        """
        Orchestrates the full generation pipeline for one textbox.
        
        Returns:
            A TextBox object with its points generated and distorted in local coordinates.
        """
        # --- Phase 1: Content & Micro-Variations ---
        text_lines = self._generate_content(box_type)
        
        # Create TextBox and consolidate to NumPy
        textbox = TextBox(text_lines=text_lines, box_type=box_type)
        textbox.consolidate_to_numpy(textline_id_start)

        # Apply congestion jitter (in-place modification)
        AUGMENTATIONS['congestion_jitter'](textbox, self.config.augmentations_phase1.congestion_jitter.model_dump(), self.rng)
        
        # --- Phase 2: Geometric Distortion ---
        # These operations return a new array, so we re-assign
        points = textbox.points_local
        for aug_name in ['shear', 'stretch', 'warp']:
            aug_config = getattr(self.config.augmentations_phase2, aug_name)
            points = AUGMENTATIONS[aug_name](points, aug_config.model_dump(), self.rng)
        
        # Update the textbox with the distorted points
        textbox._points_local = points

        # Recalculate hull after distortions
        if len(textbox.points_local) >= 3:
            from shapely.geometry import MultiPoint
            textbox._hull_local = MultiPoint(textbox.points_local[:, :2]).convex_hull
        
        return textbox

    def _generate_content(self, box_type: str) -> List[TextLine]:
        """
        Generates the ideal text content (Points, Words, Lines) for a textbox.
        Applies micro-level jitters during this process.
        """
        # box_cfg = getattr(self.config.textboxes, box_type)
        box_cfg = self.config.textboxes[box_type]
        p1_cfg = self.config.augmentations_phase1

        base_font_size = sample_from_config(box_cfg.font_size.model_dump(), self.rng)
        num_lines = sample_from_config(box_cfg.lines_per_box.model_dump(), self.rng)

        text_lines: List[TextLine] = []
        y_cursor = 0.0

        for _ in range(num_lines):
            num_words = sample_from_config(box_cfg.words_per_line.model_dump(), self.rng)
            words: List[Word] = []
            x_cursor = 0.0

            for _ in range(num_words):
                # Using a placeholder for character generation
                num_chars = self.rng.integers(3, 10)
                points: List[Point] = []
                
                for _ in range(num_chars):
                    # Apply font size variation per character
                    font_size_var = sample_from_config(p1_cfg.font_size_variation.model_dump(), self.rng)
                    font_size = base_font_size * (1 + font_size_var)

                    # Apply point-level jitter
                    jitter_x = sample_from_config(p1_cfg.point_jitter.model_dump(), self.rng)
                    jitter_y = sample_from_config(p1_cfg.point_jitter.model_dump(), self.rng)
                    
                    points.append(Point(x=x_cursor + jitter_x, y=y_cursor + jitter_y, font_size=font_size))
                    
                    # Advance cursor
                    char_spacing_multiplier = sample_from_config(p1_cfg.character_spacing.model_dump(), self.rng)
                    x_cursor += font_size * char_spacing_multiplier

                words.append(Word(points))
                
                # Advance cursor for word space
                word_spacing_multiplier = sample_from_config(p1_cfg.word_spacing.model_dump(), self.rng)
                x_cursor += base_font_size * word_spacing_multiplier

            text_lines.append(TextLine(words))
            
            # Advance cursor for line space
            line_spacing_multiplier = sample_from_config(p1_cfg.line_spacing.model_dump(), self.rng)
            y_cursor += base_font_size * line_spacing_multiplier
            
        # Text alignment and justification would be applied here by adjusting point coordinates.
        # For simplicity in this example, we skip the complex justification logic.
        # A full implementation would calculate line widths and distribute space.
            
        return text_lines