import random
import numpy as np
from typing import List, Tuple, Dict, Any

from ..core.registries import register_layout
from ..core.data_structures import TextBoxBlueprint, BoxType
from ..utils.distribution_sampler import sample_from_config
from ..utils.collision_detection import check_collision_sat
from ..config.config_models import Config

# --- Layout Strategy Implementations ---

@register_layout("rejection_sampling")
def rejection_sampling_layout(config: Config, page_width: float, page_height: float, random_state: random.Random) -> List[TextBoxBlueprint]:
    """
    Generates a list of TextBoxBlueprints using rejection sampling.
    Places a main text box first, then adds others around it.
    """
    layout_config = config.layout_strategies.rejection_sampling
    placed_blueprints: List[TextBoxBlueprint] = []
    
    # Adjacency bias: try to place new boxes near existing ones
    def get_placement_position(box_width, box_height):
        # First box is placed near the center
        if not placed_blueprints:
            px = random_state.uniform(-page_width / 4, page_width / 4)
            py = random_state.uniform(-page_height / 4, page_height / 4)
            return px, py
        
        # Subsequent boxes are placed near an existing box
        anchor_box = random_state.choice(placed_blueprints)
        angle = random_state.uniform(0, 2 * np.pi)
        distance = (max(anchor_box.width, anchor_box.height) + max(box_width, box_height)) / 2 * 0.8
        
        px = anchor_box.position[0] + np.cos(angle) * distance
        py = anchor_box.position[1] + np.sin(angle) * distance
        return px, py

    for item in layout_config.generation_queue:
        box_type_str = item['box_type']
        box_type = BoxType(box_type_str)
        box_config = config.textbox_types[box_type_str]
        
        count = int(sample_from_config(item['count'], random_state))

        for _ in range(count):
            for _ in range(layout_config.max_placement_attempts):
                width = sample_from_config(box_config.width, random_state)
                height = sample_from_config(box_config.height, random_state)
                orientation = sample_from_config(box_config.orientation_deg, random_state)
                position = get_placement_position(width, height)

                new_blueprint = TextBoxBlueprint(
                    box_type=box_type,
                    position=position,
                    width=width,
                    height=height,
                    orientation_deg=orientation
                )
                
                # Check for collisions with already placed boxes
                has_collision = False
                new_obb = new_blueprint.get_obb()
                
                # Page boundary check
                if np.any(new_obb < [-page_width/2, -page_height/2]) or np.any(new_obb > [page_width/2, page_height/2]):
                    has_collision = True
                
                if not has_collision:
                    for existing_bp in placed_blueprints:
                        if check_collision_sat(new_obb, existing_bp.get_obb()):
                            has_collision = True
                            break
                
                if not has_collision:
                    placed_blueprints.append(new_blueprint)
                    # Handle probabilistic interlinear gloss generation
                    if box_type == BoxType.MAIN_TEXT and random_state.random() < box_config.interlinear_gloss_probability:
                        # This part is simplified. A full implementation would need
                        # to generate gloss relative to specific lines *after* content generation.
                        # For layout planning, we can approximate a small box attached to it.
                        # The logic is complex, so we'll omit full gloss placement from this strategy
                        # and assume it's handled at a later stage or by a more complex strategy.
                        pass
                    break
                    
    return placed_blueprints


@register_layout("grid")
def grid_layout(config: Config, page_width: float, page_height: float, random_state: random.Random) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Generates points in a perfect grid and provides two labeling interpretations.
    Returns:
        - A single numpy array of points.
        - A dictionary of label arrays {'horizontal': ndarray, 'vertical': ndarray}.
    """
    grid_config = config.layout_strategies.grid
    rows = int(sample_from_config(grid_config.rows, random_state))
    cols = int(sample_from_config(grid_config.cols, random_state))
    spacing = sample_from_config(grid_config.spacing, random_state)
    base_font_size = 20 # Constant font size for ambiguity

    points = []
    for r in range(rows):
        for c in range(cols):
            x = c * spacing
            y = r * spacing
            points.append([x, y, base_font_size])
    
    points_array = np.array(points, dtype=np.float32)
    # Center the grid
    points_array[:, 0] -= (cols - 1) * spacing / 2
    points_array[:, 1] -= (rows - 1) * spacing / 2
    
    # Generate labels
    labels_horizontal = np.repeat(np.arange(rows), cols)
    labels_vertical = np.tile(np.arange(cols), rows)

    return points_array, {"horizontal": labels_horizontal, "vertical": labels_vertical}


@register_layout("concentric_circles")
def concentric_circles_layout(config: Config, page_width: float, page_height: float, random_state: random.Random) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Generates points in concentric circles and provides two labeling interpretations.
    """
    circle_config = config.layout_strategies.concentric_circles
    num_spokes = int(sample_from_config(circle_config.num_spokes, random_state))
    num_circles = int(sample_from_config(circle_config.num_circles, random_state))
    radial_step = sample_from_config(circle_config.radial_step, random_state)
    start_radius = sample_from_config(circle_config.start_radius, random_state)
    base_font_size = 20

    points = []
    thetas = np.linspace(0, 2 * np.pi, num_spokes, endpoint=False)
    
    for i in range(num_circles):
        radius = start_radius + i * radial_step
        for theta in thetas:
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append([x, y, base_font_size])
    
    points_array = np.array(points, dtype=np.float32)
    
    # Generate labels
    # Circular reading: each circle is a line
    labels_circular = np.repeat(np.arange(num_circles), num_spokes)
    # Radial reading: each spoke is a line
    labels_radial = np.tile(np.arange(num_spokes), num_circles)
    
    return points_array, {"circular": labels_circular, "radial": labels_radial}