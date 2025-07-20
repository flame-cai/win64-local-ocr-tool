import numpy as np
from typing import Tuple

from .data_structures import TextBoxBlueprint

def transform_to_global(points_local: np.ndarray, blueprint: TextBoxBlueprint) -> np.ndarray:
    """
    Transforms points from a textbox's local coordinate system to the global
    page coordinate system.

    The local system is assumed to be centered at (0,0). The transformation
    involves rotation followed by translation.

    Args:
        points_local (np.ndarray): Array of shape (N, 3) with (x, y, font_size)
                                   in the local coordinate system.
        blueprint (TextBoxBlueprint): The blueprint containing the global position
                                      and orientation for the textbox.

    Returns:
        np.ndarray: Array of shape (N, 3) with (x, y, font_size) in the
                    global page coordinate system. The font_size remains unchanged.
    """
    if points_local.shape[0] == 0:
        return points_local.copy()

    coords_local = points_local[:, :2]
    font_sizes = points_local[:, 2]

    angle_rad = np.deg2rad(blueprint.orientation_deg)
    
    # Create 2D rotation matrix
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a],
                                [sin_a,  cos_a]])

    # Apply rotation
    coords_rotated = coords_local @ rotation_matrix.T

    # Apply translation
    translation_vector = np.array(blueprint.position)
    coords_global = coords_rotated + translation_vector

    # Reassemble the array with font sizes
    points_global = np.hstack((coords_global, font_sizes[:, np.newaxis]))

    return points_global


def normalize_page_coordinates(points_global: np.ndarray, page_width: float, page_height: float) -> np.ndarray:
    """
    Normalizes all point coordinates and font sizes for the final output.
    - Coordinates are scaled such that the longest page dimension maps to [0, 1].
    - Font sizes are scaled by the page height.

    Args:
        points_global (np.ndarray): Array of shape (N, 3) with (x, y, font_size)
                                    in the global coordinate system.
        page_width (float): The total width of the page.
        page_height (float): The total height of the page.

    Returns:
        np.ndarray: Array of shape (N, 3) with normalized (x, y, font_size).
    """
    if points_global.shape[0] == 0:
        return points_global.copy()

    points_normalized = points_global.copy()
    longest_dim = max(page_width, page_height)

    # Normalize x, y coordinates
    # We assume the origin (0,0) of the page is at the top-left corner for normalization.
    # The generation logic places things around a central origin, so we shift first.
    points_normalized[:, 0] += page_width / 2
    points_normalized[:, 1] += page_height / 2
    
    points_normalized[:, 0] /= longest_dim
    points_normalized[:, 1] /= longest_dim

    # Normalize font size
    points_normalized[:, 2] /= page_height

    return points_normalized