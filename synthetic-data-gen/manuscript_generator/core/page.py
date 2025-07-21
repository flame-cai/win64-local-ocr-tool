# /manuscript_generator/core/page.py

import numpy as np
import os
from dataclasses import dataclass, field
from typing import List, Tuple
from pathlib import Path

from manuscript_generator.core.textbox import TextBox
from manuscript_generator.configs.base_config import Config

@dataclass
class Page:
    """Represents a single generated manuscript page."""
    width: int
    height: int
    textboxes: List[TextBox] = field(default_factory=list)
    
    # Final data for output
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    textbox_labels: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=int))
    textline_labels: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=int))

    def finalize(self, config: Config, rng: np.random.Generator):
        """Combines all textboxes into final page-level arrays and applies Phase 3 augmentations."""
        if not self.textboxes:
            return

        all_points = []
        all_textbox_labels = []
        all_textline_labels = []
        global_line_id_offset = 0

        for textbox_id, box in enumerate(self.textboxes):
            if box.points_global is None or box.points_global.shape[0] == 0:
                continue
                
            all_points.append(box.points_global)
            all_textbox_labels.append(np.full(box.points_global.shape[0], textbox_id))
            
            # Ensure textline IDs are globally unique
            # Check if line_ids_local is not None and has elements
            if box.line_ids_local is not None and box.line_ids_local.size > 0:
                line_ids = box.line_ids_local + global_line_id_offset
                all_textline_labels.append(line_ids)
                global_line_id_offset += np.max(box.line_ids_local) + 1
            elif box.points_global.shape[0] > 0: # Handle textboxes with points but no lines
                all_textline_labels.append(np.full(box.points_global.shape[0], global_line_id_offset))
                global_line_id_offset += 1


        if not all_points:
            return
            
        self.points = np.vstack(all_points)
        self.textbox_labels = np.concatenate(all_textbox_labels)
        self.textline_labels = np.concatenate(all_textline_labels)

        # Apply Phase 3 augmentations (e.g., Point Dropout)
        from manuscript_generator.core.registry import AUGMENTATIONS
        
        # --- Start of Fix ---
        # Call with a placeholder 'None' for the context argument to match the unified signature.
        self.points, kept_indices = AUGMENTATIONS['point_dropout'](self.points, None, config, rng)
        # --- End of Fix ---
        
        self.textbox_labels = self.textbox_labels[kept_indices]
        self.textline_labels = self.textline_labels[kept_indices]
        
    def save(self, output_dir: Path, sample_id: str):
        """Saves all generated data for this page to disk."""
        sample_dir = output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # --- Normalization ---
        # Scale coords so the longest dimension is in [0, 1]
        longest_dim = max(self.width, self.height)
        if longest_dim == 0: # Avoid division by zero for empty pages
            return

        points_normalized = self.points.copy()
        points_normalized[:, :2] /= longest_dim
        # Normalize font size by page height
        if self.height > 0:
            points_normalized[:, 2] /= self.height
        
        # --- File Saving ---
        np.savetxt(sample_dir / "inputs_unnormalized.txt", self.points, fmt="%.2f %.2f %d")
        np.savetxt(sample_dir / "inputs_normalized.txt", points_normalized, fmt="%.6f %.6f %.6f")
        np.savetxt(sample_dir / "labels_textbox.txt", self.textbox_labels, fmt="%d")
        np.savetxt(sample_dir / "labels_textline.txt", self.textline_labels, fmt="%d")
        
        # --- Metadata ---
        with open(sample_dir / "meta_data.txt", "w") as f:
            f.write(f"sample_id: {sample_id}\n")
            f.write(f"page_width: {self.width}\n")
            f.write(f"page_height: {self.height}\n")
            f.write(f"num_points: {self.points.shape[0]}\n")
            f.write(f"num_textboxes: {len(self.textboxes)}\n")
            if self.textline_labels.size > 0:
                 f.write(f"num_textlines: {len(np.unique(self.textline_labels))}\n")
            else:
                 f.write(f"num_textlines: 0\n")
            f.write("\n--- TextBoxes ---\n")
            for i, box in enumerate(self.textboxes):
                num_pts = box.points_global.shape[0] if box.points_global is not None else 0
                f.write(f"  - id: {i}, type: {box.box_type}, orientation: {box.orientation_deg:.2f}, num_points: {num_pts}\n")