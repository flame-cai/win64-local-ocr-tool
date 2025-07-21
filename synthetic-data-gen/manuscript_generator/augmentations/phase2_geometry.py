import numpy as np
from ..core.registries import register_augmentation
from ..core.distributions import sample_from_config

@register_augmentation("shear")
def apply_shear(points: np.ndarray, config: dict, rng: np.random.Generator) -> np.ndarray:
    """Applies shear transformation to a point cloud."""
    if rng.random() > config["prob"] or points.shape[0] == 0:
        return points
    
    factor_x = sample_from_config(config["factor_x"], rng)
    factor_y = sample_from_config(config["factor_y"], rng)
    
    shear_matrix = np.array([[1, factor_x], [factor_y, 1]])
    
    transformed_points = points.copy()
    transformed_points[:, :2] = transformed_points[:, :2] @ shear_matrix.T
    return transformed_points

@register_augmentation("stretch")
def apply_stretch(points: np.ndarray, config: dict, rng: np.random.Generator) -> np.ndarray:
    """Applies non-uniform scaling to a point cloud."""
    if rng.random() > config["prob"] or points.shape[0] == 0:
        return points

    factor_x = sample_from_config(config["factor_x"], rng)
    factor_y = sample_from_config(config["factor_y"], rng)

    stretch_matrix = np.array([[factor_x, 0], [0, factor_y]])

    transformed_points = points.copy()
    transformed_points[:, :2] = transformed_points[:, :2] @ stretch_matrix.T
    return transformed_points

@register_augmentation("warp")
def apply_warp(points: np.ndarray, config: dict, rng: np.random.Generator) -> np.ndarray:
    """Applies a non-linear wave-like distortion to a point cloud."""
    if rng.random() > config["prob"] or points.shape[0] == 0:
        return points

    transformed_points = points.copy()
    coords = transformed_points[:, :2]

    amp_x = sample_from_config(config["amplitude_x"], rng)
    freq_x = sample_from_config(config["frequency_x"], rng)
    amp_y = sample_from_config(config["amplitude_y"], rng)
    freq_y = sample_from_config(config["frequency_y"], rng)
    
    # Apply warp
    coords[:, 0] += amp_y * np.sin(freq_y * coords[:, 1])
    coords[:, 1] += amp_x * np.sin(freq_x * coords[:, 0])

    transformed_points[:, :2] = coords
    return transformed_points