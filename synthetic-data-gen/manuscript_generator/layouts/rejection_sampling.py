from typing import List, Tuple, Generator
import numpy as np

from ..core.classes import TextBox, PageData
from ..core.registries import register_layout
from ..core.distributions import sample_from_config
from ..core.utils import check_overlap, check_bounds
from ..generation.textbox_generator import TextBoxGenerator
from ..config import AppConfig

@register_layout("rejection_sampling")
def generate_rejection_sampling(config: AppConfig, rng: np.random.Generator) -> Generator[PageData, None, None]:
    """
    Generates a page layout by randomly placing textboxes and rejecting overlaps.
    """
    layout_cfg = config.layout.rejection_sampling
    page_width = sample_from_config(config.page.width.model_dump(), rng)
    page_height = sample_from_config(config.page.height.model_dump(), rng)
    page_dims = (page_width, page_height)
    
    num_boxes_to_place = sample_from_config(layout_cfg.num_textboxes.model_dump(), rng)
    placed_textboxes: List[TextBox] = []
    
    textbox_gen = TextBoxGenerator(config, rng)
    
    global_textline_id_counter = 1 # Start IDs from 1
    
    for textbox_idx in range(num_boxes_to_place):
        
        # --- Choose Textbox Type ---
        box_type = rng.choice(
            list(layout_cfg.textbox_type_probs.keys()),
            p=list(layout_cfg.textbox_type_probs.values())
        )

        generation_success = False
        for _ in range(layout_cfg.max_generation_attempts):
            
            # --- Generate a new textbox ---
            new_textbox = textbox_gen.generate(box_type, global_textline_id_counter)
            if new_textbox.points_local.shape[0] == 0: continue # Skip empty boxes

            placement_success = False
            for _ in range(layout_cfg.max_placement_attempts):
                # --- Attempt to place it ---
                new_textbox.position = (rng.uniform(0, page_width), rng.uniform(0, page_height))
                new_textbox.orientation_deg = rng.uniform(0, 360)

                new_hull = new_textbox.get_global_hull()

                # Check 1: Page bounds
                if not check_bounds(new_hull, page_dims):
                    continue
                
                # Check 2: Overlap with existing boxes
                is_overlapping = False
                for existing_box in placed_textboxes:
                    if check_overlap(new_hull, existing_box.get_global_hull(), allow_touch=not layout_cfg.allow_overlap):
                        is_overlapping = True
                        break
                
                if not is_overlapping:
                    placement_success = True
                    break
            
            if placement_success:
                generation_success = True
                placed_textboxes.append(new_textbox)
                # Increment textline counter by the number of lines actually added
                if new_textbox.line_ids_local.size > 0:
                    global_textline_id_counter = np.max(new_textbox.line_ids_local) + 1
                break
        
        # (Optional) Add interlinear gloss if main_text was just placed
        if generation_success and box_type == 'main_text' and rng.random() < config.page.interlinear_gloss_prob:
             # A full implementation would find a specific line and place the gloss relative to it.
             # This is a simplified placement for demonstration.
            pass

    # --- Consolidate all page data ---
    all_points, all_tb_labels, all_tl_labels = [], [], []
    for tb_id, textbox in enumerate(placed_textboxes, 1):
        points_global = textbox.points_local # Already distorted
        
        # Final transformation from local to global page coordinates
        from ..core.utils import transform_points
        points_global = transform_points(points_global, textbox.position, textbox.orientation_deg)
        
        all_points.append(points_global)
        all_tb_labels.append(np.full(len(points_global), tb_id))
        all_tl_labels.append(textbox.line_ids_local)

    if not all_points: # Handle empty pages
        yield PageData(sample_id="", points_global=np.empty((0,3)), labels_textbox=np.empty(0), labels_textline=np.empty(0), page_dims=page_dims)
        return

    final_points = np.concatenate(all_points, axis=0)
    final_tb_labels = np.concatenate(all_tb_labels, axis=0)
    final_tl_labels = np.concatenate(all_tl_labels, axis=0)
    
    # --- Phase 3: Page-Level Augmentations ---
    from ..augmentations.phase3_page import apply_point_dropout
    final_points, surviving_indices = apply_point_dropout(final_points, config.augmentations_phase3.point_dropout.model_dump(), rng)
    final_tb_labels = final_tb_labels[surviving_indices]
    final_tl_labels = final_tl_labels[surviving_indices]

    meta = {
        "page_width": page_width,
        "page_height": page_height,
        "num_textboxes": len(placed_textboxes),
        "textbox_types": [tb.box_type for tb in placed_textboxes],
        "layout_strategy": "rejection_sampling",
    }
    
    page_data = PageData(
        sample_id="", # Will be set by the main loop
        points_global=final_points,
        labels_textbox=final_tb_labels,
        labels_textline=final_tl_labels,
        page_dims=page_dims,
        meta=meta
    )
    yield page_data