# /manuscript_generator/augmentations/phase1_content.py

import numpy as np

from manuscript_generator.core.registry import register_augmentation
from manuscript_generator.utils.distribution_sampler import sample_from_distribution
from manuscript_generator.configs.base_config import Config

@register_augmentation("font_size_variation")
def apply_font_size_variation(points: np.ndarray, line_ids: np.ndarray, config: Config, rng: np.random.Generator) -> np.ndarray:
    """Applies a slight random variation to each point's font size."""
    aug_config = config.textbox_content.font_size_variation
    variation = sample_from_distribution(aug_config['variation_factor'], rng)
    
    # --- Start of Fix ---
    # Apply multiplicatively. The scale (standard deviation) for the normal distribution
    # cannot be negative, so we take the absolute value of the sampled variation.
    noise = rng.normal(0, abs(variation), size=points.shape[0])
    points[:, 2] *= (1 + noise)
    # --- End of Fix ---

    points[:, 2] = np.maximum(1, points[:, 2]) # Ensure font size is positive
    return points

@register_augmentation("point_level_jitter")
def apply_point_level_jitter(points: np.ndarray, line_ids: np.ndarray, config: Config, rng: np.random.Generator) -> np.ndarray:
    """Adds a slight random offset to each point's (x, y) coordinates."""
    aug_config = config.textbox_content.point_level_jitter
    jitter_std_factor = sample_from_distribution(aug_config['jitter_std_factor'], rng)
    
    # Jitter standard deviation is proportional to the point's font size
    jitter_std = points[:, 2] * jitter_std_factor
    
    # Generate jitter for x and y and stack it
    jitter_x = rng.normal(0, jitter_std)
    jitter_y = rng.normal(0, jitter_std)
    points[:, :2] += np.stack([jitter_x, jitter_y], axis=1)
    
    return points

@register_augmentation("congestion_jitter")
def apply_congestion_jitter(points: np.ndarray, line_ids: np.ndarray, config: Config, rng: np.random.Generator) -> np.ndarray:
    """Simulates a congested writing style by heavily jittering a percentage of points."""
    aug_config = config.textbox_content.congestion_jitter
    percentage = sample_from_distribution(aug_config['percentage'], rng)
    strength_factor = sample_from_distribution(aug_config['strength_factor'], rng)

    num_points_to_jitter = int(points.shape[0] * percentage)
    if num_points_to_jitter == 0:
        return points

    # Calculate average line spacing for this textbox
    unique_lines = np.unique(line_ids)
    if len(unique_lines) > 1:
        line_y_positions = [np.mean(points[line_ids == i, 1]) for i in unique_lines]
        avg_line_spacing = np.mean(np.diff(np.sort(line_y_positions)))
    else: # Fallback for single-line textboxes
        avg_line_spacing = np.mean(points[:, 2]) * 2 # Estimate based on font size

    jitter_std = abs(avg_line_spacing) * strength_factor

    # Select random points to jitter
    indices_to_jitter = rng.choice(points.shape[0], num_points_to_jitter, replace=False)

    # Apply strong jitter
    jitter_x = rng.normal(0, jitter_std, size=num_points_to_jitter)
    jitter_y = rng.normal(0, jitter_std, size=num_points_to_jitter)
    
    points[indices_to_jitter, 0] += jitter_x
    points[indices_to_jitter, 1] += jitter_y
    
    return points