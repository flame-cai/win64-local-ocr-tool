import random
from typing import Dict, Any

from ..core.data_structures import Point, Word, TextLine, TextBox, TextAlignment, BoxType
from ..utils.distribution_sampler import sample_from_config
from ..config.config_models import TextBoxTypeConfig, AugmentationProfile


class ContentGenerator:
    """
    Generates the content of a single TextBox in its local coordinate system.
    This class handles Phase 1 of the generation pipeline: creating the ideal
    text structure and applying micro-variations.
    """
    def __init__(self, box_config: TextBoxTypeConfig, aug_config: AugmentationProfile, random_state: random.Random):
        self.box_config = box_config
        self.aug_config = aug_config
        self.random = random_state

    def generate_textbox_content(self, width: float, height: float, box_type: BoxType) -> TextBox:
        """
        Main method to generate the full content of a textbox.

        Args:
            width (float): The width of the textbox blueprint.
            height (float): The height of the textbox blueprint.
            box_type (BoxType): The type of the textbox.

        Returns:
            TextBox: A TextBox object populated with text lines, words, and points.
        """
        num_lines = int(sample_from_config(self.box_config.lines_per_box, self.random))
        base_font_size = sample_from_config(self.box_config.font_size, self.random)
        
        text_lines: list[TextLine] = []
        current_y = -height / 2 # Start at the top of the box

        for _ in range(num_lines):
            # Apply line-level font size variation (Phase 1 Augmentation)
            font_size_multiplier = 1 + sample_from_config(self.aug_config.line_level_font_size_variation, self.random)
            line_font_size = base_font_size * font_size_multiplier

            line_spacing = sample_from_config(self.box_config.line_spacing, self.random) * line_font_size
            
            if current_y + line_spacing > height / 2:
                break # Stop if we run out of vertical space

            current_y += line_spacing
            
            num_words = int(sample_from_config(self.box_config.words_per_line, self.random))
            text_line = self._generate_line(num_words, line_font_size)
            
            # Position the line vertically
            for word in text_line.words:
                for point in word.points:
                    point.y += current_y
            
            text_lines.append(text_line)

        # Apply text alignment
        self._apply_text_alignment(text_lines, width)

        return TextBox(text_lines=text_lines, box_type=box_type)

    def _generate_line(self, num_words: int, font_size: float) -> TextLine:
        """Generates a single line of text."""
        words = []
        current_x = 0
        
        word_spacing = sample_from_config(self.box_config.word_spacing, self.random) * font_size

        for i in range(num_words):
            word = self._generate_word(font_size)
            
            # Position the word horizontally
            for point in word.points:
                point.x += current_x
            
            words.append(word)
            
            # Update current_x for the next word
            word_width = max(p.x for p in word.points) - min(p.x for p in word.points) if word.points else 0
            current_x += word_width + word_spacing
            
        return TextLine(words=words)

    def _generate_word(self, font_size: float) -> Word:
            """Generates a single word as a collection of points."""
            # For simplicity, we model a "word" as a random number of "characters" (points).
            num_chars = self.random.randint(3, 10)
            points = []
            current_x = 0

            char_spacing = sample_from_config(self.box_config.char_spacing, self.random) * font_size

            for _ in range(num_chars):
                # Apply point-level jitter (Phase 1 Augmentation)
                # --- THIS IS THE KEY CHANGE ---
                # Sample the 'std' parameter first from its own distribution
                jitter_std_factor = sample_from_config(self.aug_config.point_level_jitter.std, self.random)
                jitter_std = jitter_std_factor * font_size
                
                jitter_x = self.random.normalvariate(0, jitter_std)
                jitter_y = self.random.normalvariate(0, jitter_std)

                points.append(Point(x=current_x + jitter_x, y=jitter_y, font_size=font_size))
                current_x += char_spacing
            
            return Word(points=points)
    
    def _apply_text_alignment(self, text_lines: list[TextLine], box_width: float):
        """Applies horizontal alignment to all lines in a textbox."""
        alignment_choice = sample_from_config(self.box_config.alignment, self.random)
        alignment = TextAlignment(alignment_choice)
        
        box_left_edge = -box_width / 2

        for line in text_lines:
            if not any(word.points for word in line.words):
                continue

            line_points = [p for w in line.words for p in w.points]
            min_x = min(p.x for p in line_points)
            max_x = max(p.x for p in line_points)
            line_width = max_x - min_x
            
            offset = 0
            if alignment == TextAlignment.LEFT:
                offset = box_left_edge - min_x
            elif alignment == TextAlignment.RIGHT:
                offset = (box_left_edge + box_width) - max_x
            elif alignment == TextAlignment.CENTER:
                offset = box_left_edge + (box_width - line_width) / 2 - min_x
            elif alignment == TextAlignment.JUSTIFY and len(line.words) > 1:
                slack = box_width - line_width
                if slack > 0:
                    spacing_increase = slack / (len(line.words) - 1)
                    cumulative_increase = 0
                    for i, word in enumerate(line.words[1:], 1):
                        cumulative_increase += spacing_increase
                        for point in word.points:
                            point.x += cumulative_increase
                # After justifying, align to left edge
                # Recalculate min_x after justification
                min_x = min(p.x for w in line.words for p in w.points)
                offset = box_left_edge - min_x
            
            if offset != 0:
                for word in line.words:
                    for point in word.points:
                        point.x += offset