import numpy as np

def check_collision_sat(obb1: np.ndarray, obb2: np.ndarray) -> bool:
    """
    Checks for collision between two Oriented Bounding Boxes (OBBs) using the
    Separating Axis Theorem (SAT).

    Args:
        obb1 (np.ndarray): An array of shape (4, 2) representing the vertices
                           of the first OBB.
        obb2 (np.ndarray): An array of shape (4, 2) representing the vertices
                           of the second OBB.

    Returns:
        bool: True if the OBBs are colliding, False otherwise.
    """
    polygons = [obb1, obb2]
    for polygon in polygons:
        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len(polygon)]
            
            edge = p2 - p1
            # The normal is perpendicular to the edge
            normal = np.array([-edge[1], edge[0]])
            
            # Project both polygons onto the normal
            minA, maxA = project_polygon(normal, polygons[0])
            minB, maxB = project_polygon(normal, polygons[1])
            
            # Check for overlap on the axis
            if maxA < minB or maxB < minA:
                # Found a separating axis, no collision
                return False
                
    # No separating axis found, so the polygons must be colliding
    return True

def project_polygon(axis: np.ndarray, polygon: np.ndarray) -> tuple[float, float]:
    """
    Projects a polygon's vertices onto an axis.

    Args:
        axis (np.ndarray): The 1D axis to project onto.
        polygon (np.ndarray): The polygon vertices.

    Returns:
        tuple[float, float]: The min and max projection values.
    """
    # Normalize the axis to avoid issues with magnitude
    axis_norm = axis / np.linalg.norm(axis)
    
    # Project all vertices onto the axis
    projections = polygon @ axis_norm
    
    return np.min(projections), np.max(projections)