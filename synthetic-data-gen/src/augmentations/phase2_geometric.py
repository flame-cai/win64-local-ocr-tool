import numpy as np
import random

# Remove unused registry import
# from ..core.registries import register_augmentation 
from ..config.config_models import AugmentationProfile, AugmentationParam
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
        points = _shear(points, aug_profile.shear, random_state)
    
    # Apply Stretch
    if random_state.random() < aug_profile.stretch.prob:
        points = _stretch(points, aug_profile.stretch, random_state)

    # Apply Warp/Curl
    if random_state.random() < aug_profile.warp.prob:
        points = _warp(points, aug_profile.warp, random_state)

    return points

# --- Helper Functions (internal to this module) ---

def _shear(points: np.ndarray, params: AugmentationParam, random_state: random.Random) -> np.ndarray:
    """Applies shear transformation to the point cloud."""
    shear_x = sample_from_config(params.x_factor, random_state)
    shear_y = sample_from_config(params.y_factor, random_state)
    
    transform_matrix = np.array([
        [1, shear_x],
        [shear_y, 1]
    ])
    
    points[:, :2] = points[:, :2] @ transform_matrix.T
    return points

def _stretch(points: np.ndarray, params: AugmentationParam, random_state: random.Random) -> np.ndarray:
    """Applies non-uniform scaling to the point cloud."""
    stretch_x = sample_from_config(params.x_factor, random_state)
    stretch_y = sample_from_config(params.y_factor, random_state)
    
    points[:, 0] *= stretch_x
    points[:, 1] *= stretch_y
    return points

def _warp(points: np.ndarray, params: AugmentationParam, random_state: random.Random) -> np.ndarray:
    """Applies non-linear, wave-like distortions to simulate page curl."""
    coords = points[:, :2]
    
    width = np.max(coords[:, 0]) - np.min(coords[:, 0]) if coords.shape[0] > 0 else 1
    height = np.max(coords[:, 1]) - np.min(coords[:, 1]) if coords.shape[0] > 0 else 1

    amp_factor = sample_from_config(params.amplitude, random_state)
    freq = sample_from_config(params.frequency, random_state)
    axis = sample_from_config(params.axis, random_state)

    if axis == 'x':
        amplitude = amp_factor * height
        norm_coords = (coords[:, 1] - np.min(coords[:, 1])) / (height + 1e-6) * 2 * np.pi * freq
        coords[:, 0] += np.sin(norm_coords) * amplitude
    else: # axis == 'y'
        amplitude = amp_factor * width
        norm_coords = (coords[:, 0] - np.min(coords[:, 0])) / (width + 1e-6) * 2 * np.pi * freq
        coords[:, 1] += np.sin(norm_coords) * amplitude
        
    points[:, :2] = coords
    return points