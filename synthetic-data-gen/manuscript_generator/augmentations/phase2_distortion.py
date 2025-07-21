# /manuscript_generator/augmentations/phase2_distortion.py

import numpy as np

from manuscript_generator.core.registry import register_augmentation
from manuscript_generator.utils.distribution_sampler import sample_from_distribution
from manuscript_generator.utils.geometry import apply_transform
from manuscript_generator.configs.base_config import Config
from manuscript_generator.core.textbox import TextBox

@register_augmentation("shear")
def apply_shear(points: np.ndarray, textbox: TextBox, config: Config, rng: np.random.Generator) -> np.ndarray:
    """Applies a shear transformation to the textbox points."""
    aug_config = config.textbox_distortion.shear
    sx = sample_from_distribution(aug_config.shear_factor_x, rng)
    sy = sample_from_distribution(aug_config.shear_factor_y, rng)
    shear_matrix = np.array([[1, sx], [sy, 1]])
    return apply_transform(points, shear_matrix)

@register_augmentation("stretch")
def apply_stretch(points: np.ndarray, textbox: TextBox, config: Config, rng: np.random.Generator) -> np.ndarray:
    """Applies non-uniform scaling to the textbox points."""
    aug_config = config.textbox_distortion.stretch
    stretch_x = sample_from_distribution(aug_config.stretch_factor_x, rng)
    stretch_y = sample_from_distribution(aug_config.stretch_factor_y, rng)
    stretch_matrix = np.array([[stretch_x, 0], [0, stretch_y]])
    return apply_transform(points, stretch_matrix)

@register_augmentation("warp_curl")
def apply_warp_curl(points: np.ndarray, textbox: TextBox, config: Config, rng: np.random.Generator) -> np.ndarray:
    """Applies a non-linear, wave-like distortion."""
    # --- Start of Fix ---
    # Guard against division by zero for textboxes with no height or width
    if textbox.height is None or textbox.height <= 0 or textbox.width is None or textbox.width <= 0:
        return points
    # --- End of Fix ---

    aug_config = config.textbox_distortion.warp_curl
    
    # Amplitudes are relative to the textbox size for consistent effect
    amp_x = textbox.height * sample_from_distribution(aug_config.amplitude_factor, rng)
    amp_y = textbox.width * sample_from_distribution(aug_config.amplitude_factor, rng)

    # Frequencies
    freq_x = (2 * np.pi / textbox.height) * sample_from_distribution(aug_config.frequency_factor_y, rng)
    freq_y = (2 * np.pi / textbox.width) * sample_from_distribution(aug_config.frequency_factor_x, rng)

    phase_x = sample_from_distribution(aug_config.phase_x, rng)
    phase_y = sample_from_distribution(aug_config.phase_y, rng)

    x, y = points[:, 0], points[:, 1]
    
    new_x = x + amp_x * np.sin(freq_x * y + phase_x)
    new_y = y + amp_y * np.sin(freq_y * x + phase_y)
    
    points[:, 0] = new_x
    points[:, 1] = new_y
    return points