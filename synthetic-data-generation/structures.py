# structures.py (Corrected for modern Shapely versions)

from dataclasses import dataclass, field
import numpy as np
from shapely.geometry import Polygon
# Correctly import transformation functions from shapely.affinity
from shapely.affinity import rotate, translate
import os

@dataclass
class Point:
    x: float
    y: float
    font_size: float

@dataclass
class Word:
    points: list[Point] = field(default_factory=list)

@dataclass
class TextLine:
    words: list[Word] = field(default_factory=list)

@dataclass
class TextBox:
    box_type: str
    position: tuple[float, float]
    width: float
    height: float
    orientation_deg: float = 0.0
    text_lines: list[TextLine] = field(default_factory=list)
    box_id: int = -1

    def get_oriented_bounding_box(self) -> Polygon:
        """Returns a shapely Polygon representing the oriented bounding box."""
        cx, cy = self.position
        w, h = self.width, self.height
        
        # Unrotated corners around origin (0,0)
        corners = [
            (-w / 2, -h / 2), (+w / 2, -h / 2),
            (+w / 2, +h / 2), (-w / 2, +h / 2)
        ]
        
        unrotated_poly = Polygon(corners)
        
        # --- FIXED LOGIC ---
        # Use the functional API from shapely.affinity
        # 1. Rotate the polygon around its center
        rotated_poly = rotate(unrotated_poly, self.orientation_deg, origin='center', use_radians=False)
        # 2. Translate the rotated polygon to its final position on the page
        final_poly = translate(rotated_poly, xoff=cx, yoff=cy)
        
        return final_poly

    def collides_with(self, other_box: 'TextBox') -> bool:
        """Checks for collision with another TextBox using their OBBs."""
        return self.get_oriented_bounding_box().intersects(other_box.get_oriented_bounding_box())

@dataclass
class Page:
    width: float
    height: float
    textboxes: list[TextBox] = field(default_factory=list)
    _textline_id_counter: int = 0

    def add_textbox(self, textbox_to_add: TextBox, collision_check: bool = True) -> bool:
        """
        Attempts to add a textbox to the page.
        Returns True if successful, False otherwise.
        """
        if collision_check:
            for existing_box in self.textboxes:
                if textbox_to_add.collides_with(existing_box):
                    return False
        
        textbox_to_add.box_id = len(self.textboxes)
        self.textboxes.append(textbox_to_add)
        return True

    def get_next_textline_id(self) -> int:
        """Returns a globally unique ID for a text line."""
        self._textline_id_counter += 1
        return self._textline_id_counter

    def finalize_and_save_text(self, output_dir: str, sample_id: str, config: dict, rng: np.random.Generator):
        """
        Finalizes point coordinates, applies missing chars augmentation, saves text files,
        and returns the finalized data needed for visualization.
        """
        os.makedirs(output_dir, exist_ok=True)

        unnormalized_points, textbox_labels, textline_labels = [], [], []
        use_missing_chars = config['augmentations']['use_missing_chars']
        missing_chars_prob = config['missing_chars']['probability']

        for textbox in self.textboxes:
            angle_rad = np.radians(textbox.orientation_deg)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            tx, ty = textbox.position

            for text_line in textbox.text_lines:
                line_id = self.get_next_textline_id()
                for word in text_line.words:
                    for point in word.points:
                        if use_missing_chars and rng.random() < missing_chars_prob:
                            continue

                        rotated_coords = rotation_matrix @ [point.x, point.y]
                        final_x = rotated_coords[0] + tx
                        final_y = rotated_coords[1] + ty
                        
                        unnormalized_points.append((final_x, final_y, point.font_size))
                        textbox_labels.append(textbox.box_id)
                        textline_labels.append(line_id)

        normalized_points = []
        for x, y, fs in unnormalized_points:
            norm_x = x / self.width
            norm_y = y / self.height
            norm_fs = fs / self.height
            normalized_points.append((norm_x, norm_y, norm_fs))

        base_path = os.path.join(output_dir, sample_id)
        np.savetxt(f"{base_path}_input_unnormalized.txt", unnormalized_points, fmt='%.2f %.2f %.2f')
        np.savetxt(f"{base_path}_input_normalized.txt", normalized_points, fmt='%.6f %.6f %.6f')
        np.savetxt(f"{base_path}_labels_textbox.txt", textbox_labels, fmt='%d')
        np.savetxt(f"{base_path}_labels_textline.txt", textline_labels, fmt='%d')
        
        print(f"Successfully generated and saved text data for sample '{sample_id}'")
        return unnormalized_points, textbox_labels