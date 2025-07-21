# /manuscript_generator/augmentations/phase3_page.py

import numpy as np

from manuscript_generator.core.registry import register_augmentation
from manuscript_generator.utils.distribution_sampler import sample_from_distribution
from manuscript_generator.configs.base_config import Config
from typing import Any, Optional

@register_augmentation("point_dropout")
def apply_point_dropout(points: np.ndarray, context: Optional[Any], config: Config, rng: np.random.Generator) -> np.ndarray:
    """
    Randomly removes points from the final page.
    The 'context' argument is ignored but included for a consistent signature.
    """
    aug_config = config.page_augmentations.point_dropout
    if not aug_config['enabled']:
        return points, np.arange(points.shape[0])
        
    dropout_prob = sample_from_distribution(aug_config['probability'], rng)
    if dropout_prob == 0:
        return points, np.arange(points.shape[0])
        
    mask = rng.random(size=points.shape[0]) > dropout_prob
    
    # We need to return the indices that were kept to update the labels as well
    kept_indices = np.where(mask)[0]
    return points[mask], kept_indices