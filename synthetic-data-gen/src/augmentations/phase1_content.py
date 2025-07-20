import numpy as np
import random
from ..core.registries import register_augmentation

# Phase 1 augmentations (like point-level jitter and line-level font size variation)
# are applied directly within the `ContentGenerator`. This is because they are
# tightly coupled with the logical structure (points, lines) being created.
# They are not vectorized operations on a consolidated point cloud.
#
# This file is a placeholder to represent the logical separation.
# We could move the jitter logic here into a callable function if we wanted
# to make it more pluggable, but the current implementation inside
# ContentGenerator is clear and efficient for this phase.

@register_augmentation("point_level_jitter")
def point_level_jitter(point: np.ndarray, std: float, random_state: random.Random) -> np.ndarray:
    """
    Applies a small random offset to the local (x, y) coordinates of a point.
    NOTE: This is a conceptual function. The actual implementation is within ContentGenerator
    for efficiency during point creation.
    """
    jitter_x = random_state.normalvariate(0, std)
    jitter_y = random_state.normalvariate(0, std)
    point[0] += jitter_x
    point[1] += jitter_y
    return point