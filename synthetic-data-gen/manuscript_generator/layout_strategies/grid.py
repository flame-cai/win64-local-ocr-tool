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
    Generates an ambiguous grid layout on a randomly sized and positioned page.
    For each geometric arrangement, it returns TWO Page objects with the same
    points but different textline labels (horizontal vs. vertical reading order).
    """
    grid_config = config.grid_layout
    rows = sample_from_distribution(grid_config.rows, rng)
    cols = sample_from_distribution(grid_config.cols, rng)
    spacing = sample_from_distribution(grid_config.spacing, rng)

    # Generate grid points in a local coordinate system (centered around origin later)
    x = np.arange(cols) * spacing
    y = np.arange(rows) * spacing
    xx, yy = np.meshgrid(x, y)
    points_flat = np.stack([xx.ravel(), yy.ravel()], axis=1)

    # Add font size - constant for grid layout for maximum ambiguity
    base_font_size = spacing / 2.0
    font_sizes = np.full((points_flat.shape[0], 1), base_font_size)
    points_local_initial = np.hstack([points_flat, font_sizes])

    # Create the two interpretations of line labels
    # 1. Horizontal reading order (row-major)
    labels_horizontal = np.arange(rows).repeat(cols)
    # 2. Vertical reading order (column-major)
    labels_vertical = np.tile(np.arange(cols), rows)

    # --- FIX: Generate page dimensions FIRST, independently of the grid size. ---
    page_width = sample_from_distribution(config.page.width, rng)
    page_height = sample_from_distribution(config.page.height, rng)

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

        # --- FIX: Create the page with the pre-determined random dimensions. ---
        page = Page(width=page_width, height=page_height, textboxes=[textbox])
        
        # --- FIX: Place the grid RANDOMLY within the page boundaries. ---
        # 1. Recalculate the grid's bounding box after local augmentations.
        if textbox.points_local.shape[0] > 0:
            min_coords = np.min(textbox.points_local[:, :2], axis=0)
            max_coords = np.max(textbox.points_local[:, :2], axis=0)
            grid_w = max_coords[0] - min_coords[0]
            grid_h = max_coords[1] - min_coords[1]
        else: # Handle case where all points were dropped
            grid_w, grid_h = 0, 0
            
        # 2. Sample a random top-left position for the grid's bounding box.
        # The `max(0, ...)` ensures the range isn't negative if the grid happens
        # to be larger than the page (it will just be placed at 0).
        rand_x = rng.uniform(0, max(0, page.width - grid_w))
        rand_y = rng.uniform(0, max(0, page.height - grid_h))
        
        # 3. The `textbox.position` refers to its center. We must adjust our
        # local points to be centered at (0,0) and then set the global position.
        # This is what the `consolidate_to_numpy` method normally does.
        if textbox.points_local.shape[0] > 0:
            center_offset = (min_coords + max_coords) / 2.0
            textbox.points_local[:, :2] -= center_offset
        
        # The final position is the top-left corner plus half the dimensions.
        textbox.position = (rand_x + grid_w / 2, rand_y + grid_h / 2)
        
        # Apply final orientation and transform to global coordinates
        textbox.orientation_deg = rng.uniform(-15, 15)
        textbox.transform_to_global()

        pages.append(page)

    return pages