# /manuscript_generator/utils/geometry.py

import numpy as np
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull

def get_rotation_matrix(angle_deg: float) -> np.ndarray:
    """Returns a 2D rotation matrix for a given angle in degrees."""
    angle_rad = np.deg2rad(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s], [s, c]])

def apply_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Applies a 2x2 transformation matrix to a set of (x, y) coordinates.
    Assumes points is an (N, 3) array of [x, y, font_size].
    """
    assert points.shape[1] == 3, "Points array must be of shape (N, 3)"
    assert matrix.shape == (2, 2), "Transformation matrix must be 2x2"
    
    transformed_coords = points[:, :2] @ matrix.T
    return np.hstack([transformed_coords, points[:, 2, np.newaxis]])

def get_convex_hull(points: np.ndarray) -> Polygon:
    """
    Computes the convex hull of a set of points and returns it as a Shapely Polygon.
    Returns None if hull cannot be computed (e.g., fewer than 3 points).
    """
    if points.shape[0] < 3:
        return None
    try:
        hull = ConvexHull(points[:, :2])
        return Polygon(points[hull.vertices, :2])
    except:
        # Scipy can fail on colinear points
        return None

def check_overlap(poly1: Polygon, poly2: Polygon, buffer: float = 1.0) -> bool:
    """Checks if two Shapely Polygons overlap."""
    if poly1 is None or poly2 is None:
        return True # Treat inability to form a polygon as an overlap to be safe
    return poly1.buffer(buffer).intersects(poly2.buffer(buffer))