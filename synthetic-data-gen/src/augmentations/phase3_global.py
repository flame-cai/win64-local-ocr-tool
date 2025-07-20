import numpy as np
import random
from ..core.registries import register_augmentation
from ...config.config_models import AugmentationProfile
from ..utils.distribution_sampler import sample_from_config

def apply_phase3_augmentations(points: np.ndarray, aug_profile: AugmentationProfile, random_state: random.Random) -> np.ndarray:
    """
    Applies Phase 3 augmentations to the entire page's point cloud.
    These are applied in global coordinates.
    """
    if points.shape[0] == 0:
        return points

    # Apply Point Dropout
    points = point_dropout(points, aug_profile.point_dropout, random_state)
    
    # Apply Global Jitter
    if points.shape[0] > 0: # Check again after dropout
        points = global_jitter(points, aug_profile.global_jitter, random_state)

    return points

@register_augmentation("point_dropout")
def point_dropout(points: np.ndarray, params, random_state: random.Random) -> np.ndarray:
    """Randomly removes points from the final page."""
    dropout_prob = params.prob
    if dropout_prob == 0:
        return points
        
    n_points = points.shape[0]
    mask = random_state.choices([True, False], 
                                weights=[1 - dropout_prob, dropout_prob], 
                                k=n_points)
    return points[mask]

@register_augmentation("global_jitter")
def global_jitter(points: np.ndarray, params, random_state: random.Random) -> np.ndarray:
    """Adds a small random offset to the global coordinates of every point."""
    std = sample_from_config(params, random_state)
    if std == 0:
        return points
        
    jitter = np.random.normal(loc=0, scale=std, size=(points.shape[0], 2))
    points[:, :2] += jitter
    return points