import numpy as np
from ..core.registries import register_augmentation

@register_augmentation("point_dropout")
def apply_point_dropout(points: np.ndarray, config: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly removes points from the final page.

    Returns:
        A tuple of (surviving_points, surviving_indices)
    """
    if config["prob"] == 0 or points.shape[0] == 0:
        return points, np.arange(points.shape[0])

    dropout_mask = rng.random(size=points.shape[0]) > config["prob"]
    surviving_indices = np.where(dropout_mask)[0]
    
    return points[surviving_indices], surviving_indices