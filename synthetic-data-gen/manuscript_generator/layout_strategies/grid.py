# /manuscript_generator/layout_strategies/grid.py

import numpy as np
from typing import List

from manuscript_generator.core.registry import register_layout, AUGMENTATIONS
from manuscript_generator.core.page import Page
from manuscript_generator.core.textbox import TextBox
from manuscript_generator.core.common import TextBoxType
from manuscript_generator.configs.base_config import Config
from manuscript_generator.utils.distribution_sampler import sample_from_distribution
from manuscript_generator.utils.geometry import get_rotation_matrix, apply_transform

@register_layout("grid")
def generate_grid_layout(config: Config, rng: np.random.Generator) -> List[Page]:
    """
    Generates an ambiguous grid layout.
    Returns TWO Page objects with the same points but different textline labels.
    """
    grid_config = config.grid_layout
    rows = sample_from_distribution(grid_config.rows, rng)
    cols = sample_from_distribution(grid_config.cols, rng)
    spacing = sample_from_distribution(grid_config.spacing, rng)

    # Generate grid points
    x = np.arange(cols) * spacing
    y = np.arange(rows) * spacing
    xx, yy = np.meshgrid(x, y)
    points_flat = np.stack([xx.ravel(), yy.ravel()], axis=1)
    
    # Add font size - constant for grid layout for maximum ambiguity
    base_font_size = spacing / 2.0
    font_sizes = np.full((points_flat.shape[0], 1), base_font_size)
    points_local_initial = np.hstack([points_flat, font_sizes])
    
    # Create the two interpretations of line labels
    labels_horizontal = np.arange(rows).repeat(cols)
    labels_vertical = np.tile(np.arange(cols), rows)

    pages = []
    # Loop for each interpretation (horizontal/vertical)
    for labels_initial, interpretation in [(labels_horizontal, "horizontal"), (labels_vertical, "vertical")]:
        # Create a single textbox for the grid
        textbox = TextBox(box_type=TextBoxType.GRID)
        
        # Manually set the consolidated arrays. We copy them so modifications for one
        # interpretation (e.g., horizontal) don't affect the next (vertical).
        textbox.points_local = points_local_initial.copy()
        textbox.line_ids_local = labels_initial.copy()
        textbox.width = (cols - 1) * spacing
        textbox.height = (rows - 1) * spacing

        # Apply limited augmentations as per spec
        for aug_name in grid_config.augmentations:
            # --- Start of Fix ---
            if aug_name == "point_dropout":
                # point_dropout returns a tuple: (filtered_points, kept_indices)
                filtered_points, kept_indices = AUGMENTATIONS[aug_name](
                    textbox.points_local, textbox.line_ids_local, config, rng
                )
                textbox.points_local = filtered_points
                # CRITICAL: We must also update the labels to match the dropped points.
                textbox.line_ids_local = textbox.line_ids_local[kept_indices]
            else:
                # Other augmentations return a single numpy array and don't change point count.
                textbox.points_local = AUGMENTATIONS[aug_name](
                    textbox.points_local, textbox.line_ids_local, config, rng
                )
            # --- End of Fix ---

        # Place the grid on a page
        page_width = int(textbox.width * 1.5) if textbox.width else 500
        page_height = int(textbox.height * 1.5) if textbox.height else 500
        page = Page(width=page_width, height=page_height, textboxes=[textbox])
        
        textbox.position = (page_width / 2, page_height / 2)
        textbox.orientation_deg = rng.uniform(-15, 15)
        textbox.transform_to_global()

        pages.append(page)

    return pages