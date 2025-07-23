# /manuscript_generator/layout_strategies/rejection_sampling.py

import numpy as np
from typing import List

from manuscript_generator.core.registry import register_layout
from manuscript_generator.core.page import Page
from manuscript_generator.core.textbox import create_textbox
from manuscript_generator.core.common import TextBoxType
from manuscript_generator.configs.base_config import Config
from manuscript_generator.utils.distribution_sampler import sample_from_distribution
from manuscript_generator.utils.geometry import check_overlap, get_rotation_matrix

@register_layout("rejection_sampling")
def generate_rejection_sampling_layout(config: Config, rng: np.random.Generator) -> List[Page]:
    """
    Generates a page layout by creating textboxes and placing them using rejection sampling
    to avoid overlaps.
    """
    page_width = sample_from_distribution(config.page.width, rng)
    page_height = sample_from_distribution(config.page.height, rng)
    page = Page(width=page_width, height=page_height)

    num_textboxes = sample_from_distribution(config.rejection_sampling.num_textboxes, rng)
    
    box_types_config = config.rejection_sampling.textbox_type_probabilities
    types_to_generate = rng.choice(
        list(box_types_config.keys()),
        size=num_textboxes,
        p=list(box_types_config.values())
    )
    
    for box_type_str in types_to_generate:
        box_type = TextBoxType(box_type_str)
        
        max_box_attempts = sample_from_distribution(config.rejection_sampling.max_box_generation_attempts, rng)
        
        for _ in range(max_box_attempts):
            textbox = create_textbox(box_type, config, rng)
            if textbox.points_local is None or textbox.points_local.shape[0] < 3:
                continue

            max_placement_attempts = sample_from_distribution(config.rejection_sampling.max_placement_attempts, rng)
            placed = False
            for _ in range(max_placement_attempts):
                textbox.position = (rng.uniform(0, page.width), rng.uniform(0, page.height))
                
                # --- Start of Fix ---
                # This logic block ensures textbox.orientation_deg is always a number.
                orientation_choice = sample_from_distribution(config.page_augmentations.orientation_deg, rng)
                
                final_orientation = 0.0
                if orientation_choice == "other":
                    # Sample a base orientation and add a random offset from the 'other' range
                    base_angle = rng.choice([0, 90, -90])
                    offset = sample_from_distribution(config.page_augmentations.orientation_other_range, rng)
                    final_orientation = base_angle + offset
                else:
                    # The choice is already a number (0, 90, or -90)
                    final_orientation = float(orientation_choice)
                
                textbox.orientation_deg = final_orientation
                # --- End of Fix ---

                textbox.transform_to_global()

                if textbox.hull_global is None: continue
                min_x, min_y, max_x, max_y = textbox.hull_global.bounds
                if not (0 <= min_x < page_width and 0 <= min_y < page_height and
                        0 < max_x <= page_width and 0 < max_y <= page_height):
                    continue

                has_overlap = False
                for existing_box in page.textboxes:
                    if check_overlap(textbox.hull_global, existing_box.hull_global):
                        has_overlap = True
                        break
                
                if not has_overlap:
                    page.textboxes.append(textbox)
                    placed = True
                    break
            
            if placed:
                break
    

    return [page]
