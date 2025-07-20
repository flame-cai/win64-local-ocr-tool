import numpy as np
import random
from ..core.registries import register_augmentation
from ...config.config_models import AugmentationProfile
from ..utils.distribution_sampler import sample_from_config

def apply_phase2_augmentations(points: np.ndarray, aug_profile: AugmentationProfile, random_state: random.Random) -> np.ndarray:
    """
    Applies a chain of Phase 2 geometric distortions to a textbox's point cloud.
    These are applied in local coordinates before placing the box on the page.
    """
    if points.shape[0] == 0:
        return points

    # Apply Shear
    if random_state.random() < aug_profile.shear.prob:
        points = shear(points, aug_profile.shear, random_state)
    
    # Apply Stretch
    if random_state.random() < aug_profile.stretch.prob:
        points = stretch(points, aug_profile.stretch, random_state)

    # Apply Warp/Curl
    if random_state.random() < aug_profile.warp.prob:
        points = warp(points, aug_profile.warp, random_state)

    return points

@register_augmentation("shear")
def shear(points: np.ndarray, params, random_state: random.Random) -> np.ndarray:
    """Applies shear transformation to the point cloud."""
    shear_x = sample_from_config(params.x_factor, random_state)
    shear_y = sample_from_config(params.y_factor, random_state)
    
    transform_matrix = np.array([
        [1, shear_x],
        [shear_y, 1]
    ])
    
    points[:, :2] = points[:, :2] @ transform_matrix.T
    return points

@register_augmentation("stretch")
def stretch(points: np.ndarray, params, random_state: random.Random) -> np.ndarray:
    """Applies non-uniform scaling to the point cloud."""
    stretch_x = sample_from_config(params.x_factor, random_state)
    stretch_y = sample_from_config(params.y_factor, random_state)
    
    points[:, 0] *= stretch_x
    points[:, 1] *= stretch_y
    return points

@register_augmentation("warp")
def warp(points: np.ndarray, params, random_state: random.Random) -> np.ndarray:
    """Applies non-linear, wave-like distortions to simulate page curl."""
    coords = points[:, :2]
    
    # Determine max dimensions for relative amplitude
    width = np.max(coords[:, 0]) - np.min(coords[:, 0]) if coords.shape[0] > 0 else 1
    height = np.max(coords[:, 1]) - np.min(coords[:, 1]) if coords.shape[0] > 0 else 1

    amp_factor = sample_from_config(params.amplitude, random_state)
    freq = sample_from_config(params.frequency, random_state)
    axis = sample_from_config(params.axis, random_state)

    if axis == 'x':
        amplitude = amp_factor * height
        # Normalize y-coordinates to [0, 2*pi*freq] for the sine wave
        norm_coords = (coords[:, 1] - np.min(coords[:, 1])) / height * 2 * np.pi * freq
        coords[:, 0] += np.sin(norm_coords) * amplitude
    else: # axis == 'y'
        amplitude = amp_factor * width
        # Normalize x-coordinates to [0, 2*pi*freq]
        norm_coords = (coords[:, 0] - np.min(coords[:, 0])) / width * 2 * np.pi * freq
        coords[:, 1] += np.sin(norm_coords) * amplitude
        
    points[:, :2] = coords
    return points