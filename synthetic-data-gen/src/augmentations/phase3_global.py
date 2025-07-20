import numpy as np
import random
from typing import Tuple

# Remove the import for the registry, as it's no longer used here
# from ..core.registries import register_augmentation 
from ..config.config_models import AugmentationProfile, JitterAugmentation
from ..utils.distribution_sampler import sample_from_config

def apply_phase3_augmentations(
    points: np.ndarray,
    textbox_labels: np.ndarray,
    textline_labels: np.ndarray,
    aug_profile: AugmentationProfile,
    random_state: random.Random
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies all Phase 3 augmentations to the final page data.
    This function is the single point of entry for Phase 3. It ensures that
    augmentations which change the number of points (like dropout) are
    applied correctly to the points array AND all corresponding label arrays,
    keeping them perfectly synchronized.
    """
    if points.shape[0] == 0:
        return points, textbox_labels, textline_labels

    # --- 1. Point Dropout ---
    dropout_prob = aug_profile.point_dropout.prob
    if dropout_prob > 0:
        n_points = points.shape[0]
        dropout_mask = np.array([random_state.random() > dropout_prob for _ in range(n_points)])
        points = points[dropout_mask]
        textbox_labels = textbox_labels[dropout_mask]
        textline_labels = textline_labels[dropout_mask]

    # --- 2. Global Jitter ---
    if points.shape[0] > 0:
        points = _global_jitter(points, aug_profile.global_jitter, random_state)

    return points, textbox_labels, textline_labels


def _global_jitter(points: np.ndarray, params: JitterAugmentation, random_state: random.Random) -> np.ndarray:
    """
    Helper function to add a small random offset to the global coordinates.
    """
    std = sample_from_config(params.std, random_state)
    std = abs(std)

    if std == 0:
        return points

    jitter = np.random.normal(loc=0, scale=std, size=(points.shape[0], 2))
    points[:, :2] += jitter

    return points