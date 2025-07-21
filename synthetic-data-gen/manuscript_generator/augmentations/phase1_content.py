import numpy as np
from ..core.classes import TextBox
from ..core.distributions import sample_from_config
from ..core.registries import register_augmentation

@register_augmentation("congestion_jitter")
def apply_congestion_jitter(textbox: TextBox, config: dict, rng: np.random.Generator):
    """
    Applies congestion jitter to a percentage of points in a textbox.
    This simulates hurried writing, making some points closer to other lines.
    Modifies the textbox's local points in-place.
    """
    if not config["enable"] or textbox.points_local.shape[0] == 0:
        return

    # Determine which points to jitter
    num_points = textbox.points_local.shape[0]
    indices_to_jitter = rng.choice(
        num_points,
        size=int(num_points * config["probability"]),
        replace=False
    )

    if len(indices_to_jitter) == 0:
        return

    # Calculate average line spacing for this textbox to scale the jitter
    line_heights = []
    unique_lines = np.unique(textbox.line_ids_local)
    if len(unique_lines) > 1:
        for line_id in unique_lines:
            line_points = textbox.points_local[textbox.line_ids_local == line_id]
            if line_points.size > 0:
                line_heights.append(np.mean(line_points[:, 1]))
        
        if len(line_heights) > 1:
            avg_line_spacing = np.mean(np.diff(sorted(line_heights)))
        else: # Fallback if only one line has points
            avg_line_spacing = np.mean(textbox.points_local[:, 2]) * 2
    else: # Fallback for single-line textboxes
        avg_line_spacing = np.mean(textbox.points_local[:, 2]) * 2
        
    strength = sample_from_config(config['strength'], rng)
    jitter_scale = avg_line_spacing * strength
    
    # Apply jitter
    jitter_offsets = rng.normal(loc=0.0, scale=jitter_scale, size=(len(indices_to_jitter), 2))
    
    # We primarily want vertical jitter to cause line overlap issues
    jitter_offsets[:, 0] *= 0.2 # Reduce horizontal component
    
    textbox.points_local[indices_to_jitter, :2] += jitter_offsets