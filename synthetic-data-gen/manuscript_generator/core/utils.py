import numpy as np
from shapely.geometry import Polygon
from typing import Tuple

def transform_points(points: np.ndarray, translation: Tuple[float, float], rotation_deg: float) -> np.ndarray:
    """
    Applies rotation and translation to a set of points.
    Operates in-place on a copy to avoid side effects.

    Args:
        points: NumPy array of shape (N, 3) with (x, y, font_size).
        translation: (tx, ty) tuple for translation.
        rotation_deg: Rotation angle in degrees.

    Returns:
        Transformed points as a new NumPy array.
    """
    if points.shape[0] == 0:
        return points.copy()
        
    transformed_points = points.copy()
    coords = transformed_points[:, :2]
    
    # Rotation
    if rotation_deg != 0:
        angle_rad = np.radians(rotation_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        coords = coords @ rotation_matrix.T
    
    # Translation
    coords += np.array(translation)
    
    transformed_points[:, :2] = coords
    return transformed_points


def transform_polygon(polygon: Polygon, translation: Tuple[float, float], rotation_deg: float) -> Polygon:
    """
    Applies rotation and translation to a Shapely Polygon.

    Args:
        polygon: The Shapely Polygon to transform.
        translation: (tx, ty) tuple for translation.
        rotation_deg: Rotation angle in degrees.

    Returns:
        The transformed Shapely Polygon.
    """
    from shapely.affinity import translate, rotate
    if polygon.is_empty:
        return polygon
        
    # Shapely's rotate is around an origin, default is 'center'
    # For consistency, we rotate around (0,0) then translate
    poly = rotate(polygon, angle=rotation_deg, origin=(0, 0))
    poly = translate(poly, xoff=translation[0], yoff=translation[1])
    return poly

def check_overlap(poly1: Polygon, poly2: Polygon, allow_touch: bool = True) -> bool:
    """Checks if two polygons overlap."""
    if poly1.is_empty or poly2.is_empty:
        return False
    if allow_touch:
        return poly1.intersects(poly2) and not poly1.touches(poly2)
    return poly1.intersects(poly2)

def check_bounds(polygon: Polygon, page_dims: Tuple[float, float]) -> bool:
    """Checks if a polygon is entirely within the page boundaries."""
    page_width, page_height = page_dims
    page_box = Polygon([(0, 0), (page_width, 0), (page_width, page_height), (0, page_height)])
    return page_box.contains(polygon)

def normalize_page_data(points: np.ndarray, page_dims: Tuple[float, float]) -> np.ndarray:
    """
    Normalizes point coordinates and font sizes for model input.
    - Coordinates are scaled so the longest page dimension is in [0, 1].
    - Font sizes are scaled by page height.
    """
    if points.shape[0] == 0:
        return points.copy()
        
    normalized_points = points.copy()
    page_width, page_height = page_dims
    longest_dim = max(page_width, page_height)
    
    # Normalize coordinates
    normalized_points[:, 0] /= longest_dim # x
    normalized_points[:, 1] /= longest_dim # y
    
    # Normalize font size
    normalized_points[:, 2] /= page_height
    
    return normalized_points