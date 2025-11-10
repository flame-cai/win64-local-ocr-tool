# segment_from_point_clusters.py

import os
import shutil
import numpy as np
import cv2
from sklearn.linear_model import RANSACRegressor
from scipy.interpolate import UnivariateSpline
import math
import json
import matplotlib.pyplot as plt
from annotator.segmentation.utils import loadImage
from flask import current_app
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix



# segment_from_point_clusters.py (Add this helper function)

def _generate_baselines_from_textline_points(points_filepath, textline_labels_filepath):
    """
    Generates baseline polylines by connecting points using a Minimum Spanning Tree (MST).

    This improved method correctly handles baselines of any orientation (horizontal,
    vertical, curved, etc.) by first finding the MST of the points for each textline
    and then extracting the longest path within that tree.

    Returns:
        A dictionary mapping each textline label to its ordered baseline polyline.
    """
    baselines = {}
    if not all(os.path.exists(p) for p in [points_filepath, textline_labels_filepath]):
        current_app.logger.warning("Points or textline label file not found. Cannot generate baselines.")
        return baselines

    try:
        # all_points = np.loadtxt(points_filepath, dtype=int)*2 # Scale back to original size
        all_points = np.loadtxt(points_filepath, dtype=float).astype(int)*2
        with open(textline_labels_filepath, "r") as f:
            textline_labels = [int(line.strip()) for line in f if line.strip()]

        assert len(all_points) == len(textline_labels), \
            "Point and textline label files must have the same number of entries."

        points_per_textline = defaultdict(list)
        for point, t_label in zip(all_points, textline_labels):
            if t_label > -1:
                points_per_textline[t_label].append(point)

        for t_label, points in points_per_textline.items():
            if len(points) < 2:
                if len(points) == 1:
                    baselines[t_label] = points
                continue
            
            points_arr = np.array(points)

            # 1. Build a distance matrix for all points in the line
            dist_matrix = squareform(pdist(points_arr, 'euclidean'))

            # 2. Compute the Minimum Spanning Tree
            mst = minimum_spanning_tree(csr_matrix(dist_matrix))

            # 3. Convert MST to an adjacency list for easy traversal
            adj_list = defaultdict(list)
            rows, cols = mst.nonzero()
            for r, c in zip(rows, cols):
                adj_list[r].append(c)
                adj_list[c].append(r)

            # 4. Find the longest path in the tree (the baseline)
            # Find a leaf node (degree 1) to start the search
            start_node = -1
            for i in range(len(points)):
                if len(adj_list[i]) == 1:
                    start_node = i
                    break
            if start_node == -1: # Handle cycles or single-point cases
                 start_node = 0

            # First DFS to find the farthest node from our starting leaf
            farthest_node, _ = _dfs(start_node, adj_list, len(points))
            
            # Second DFS from that farthest node to find the actual longest path
            end_node, parents = _dfs(farthest_node, adj_list, len(points))

            # 5. Reconstruct the path from the parent pointers
            path = []
            curr = end_node
            while curr is not None:
                path.append(points[curr])
                curr = parents[curr]
            
            # The path is reconstructed backwards, so reverse it
            baselines[t_label] = path[::-1]

        current_app.logger.info(f"Successfully generated {len(baselines)} baselines using MST.")

    except Exception as e:
        current_app.logger.error(f"Error generating baselines with MST: {e}", exc_info=True)
        return {}

    return baselines


def _dfs(start_node, adj_list, num_nodes):
    """Helper function to perform DFS and find the farthest node and parent path."""
    distances = [-1] * num_nodes
    parents = {node: None for node in range(num_nodes)}
    stack = [(start_node, 0)]  # (node, distance)

    max_dist = -1
    farthest_node = start_node

    visited = set()

    while stack:
        node, dist = stack.pop()
        
        if node in visited:
            continue
        visited.add(node)
        
        distances[node] = dist
        if dist > max_dist:
            max_dist = dist
            farthest_node = node
        
        for neighbor in adj_list[node]:
            if neighbor not in visited:
                parents[neighbor] = node
                stack.append((neighbor, dist + 1))
    
    return farthest_node, parents


def resize_with_padding(image, target_size, background_color=(0, 0, 0)):
    """
    Resizes an image to a target size while maintaining its aspect ratio by padding.
    """
    target_w, target_h = target_size
    if image is None or image.size == 0:
        return np.full((target_h, target_w, 3), background_color, dtype=np.uint8)
        
    h, w = image.shape[:2]

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    padded_image = np.full((target_h, target_w, 3), background_color, dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    padded_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image
    
    return padded_image

def visualize_projection_profile(profile, crop_shape, orientation='horizontal', color=(255, 255, 255), thickness=1):
    """
    Visualizes a 1D projection profile, creating an image that corresponds to the
    dimensions of the original crop.
    """
    if profile is None or len(profile) == 0:
        return np.zeros((crop_shape[0], crop_shape[1], 3), dtype=np.uint8)

    crop_h, crop_w = crop_shape
    max_val = np.max(profile)
    if max_val == 0:
        max_val = 1

    if orientation == 'horizontal':
        vis_image = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
        for i, val in enumerate(profile):
            length = int((val / max_val) * crop_w)
            if i < crop_h:
                cv2.line(vis_image, (0, i), (length, i), color, thickness)
    else:  # vertical
        vis_image = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
        for i, val in enumerate(profile):
            length = int((val / max_val) * crop_h)
            if i < crop_w:
                cv2.line(vis_image, (i, crop_h - 1), (i, crop_h - 1 - length), color, thickness)
            
    return vis_image

def create_debug_collage(original_uncropped, padded_crop, cleaned_blob, component_viz_img, heatmap_crop, config):
    """
    Creates a 2x4 collage of debugging images for a single bounding box.
    """
    TILE_SIZE = (200, 200)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.4
    FONT_COLOR = (255, 255, 255)

    # Prepare heatmap-related visualizations
    binarize_t = int(config['BINARIZE_THRESHOLD'] * 255)
    _, bin_heat_crop = cv2.threshold(heatmap_crop, binarize_t, 255, cv2.THRESH_BINARY)
    h_prof_heat = np.sum(bin_heat_crop, axis=1) / 255
    v_prof_heat = np.sum(bin_heat_crop, axis=0) / 255
    v_prof_heat_viz = visualize_projection_profile(v_prof_heat, bin_heat_crop.shape, 'vertical', color=(0, 0, 255))
    h_prof_heat_viz = visualize_projection_profile(h_prof_heat, bin_heat_crop.shape, 'horizontal', color=(0, 0, 255))
    heatmap_colorized = cv2.applyColorMap(heatmap_crop, cv2.COLORMAP_JET)

    # Create resized tiles for the collage
    orig_uncropped_tile = resize_with_padding(original_uncropped, TILE_SIZE)
    padded_crop_tile = resize_with_padding(padded_crop, TILE_SIZE)
    cleaned_blob_tile = resize_with_padding(cleaned_blob, TILE_SIZE)
    component_viz_tile = resize_with_padding(component_viz_img, TILE_SIZE)
    heat_crop_tile = resize_with_padding(heatmap_colorized, TILE_SIZE)
    v_prof_heat_tile = resize_with_padding(v_prof_heat_viz, TILE_SIZE)
    h_prof_heat_tile = resize_with_padding(h_prof_heat_viz, TILE_SIZE)
    bin_heat_tile = resize_with_padding(bin_heat_crop, TILE_SIZE)

    # Add labels to each tile
    cv2.putText(orig_uncropped_tile, "Original BBox", (5, 15), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(padded_crop_tile, "Padded BBox", (5, 15), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(cleaned_blob_tile, "Cleaned Blob", (5, 15), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(component_viz_tile, "Analyzed Components", (5, 15), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(heat_crop_tile, "Heatmap", (5, 15), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(v_prof_heat_tile, "V-Profile (Heat)", (5, 15), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(h_prof_heat_tile, "H-Profile (Heat)", (5, 15), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(bin_heat_tile, "Binarized Heatmap", (5, 15), FONT, FONT_SCALE, FONT_COLOR, 1)

    # Assemble the collage
    row1 = cv2.hconcat([orig_uncropped_tile, padded_crop_tile, cleaned_blob_tile, component_viz_tile])
    row2 = cv2.hconcat([heat_crop_tile, v_prof_heat_tile, h_prof_heat_tile, bin_heat_tile])
    collage = cv2.vconcat([row1, row2])
    
    return collage


def analyze_and_clean_blob(blob, line_type, config):
    """
    Analyzes connected components in a blob, identifies noise touching boundaries,
    and returns a cropped version of the blob along with crop coordinates.
    """
    if blob.size == 0:
        return blob, np.zeros_like(blob, dtype=np.uint8), [0, 0, 0, 0]

    _, bin_blob = cv2.threshold(blob, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_blob, connectivity=8)
    
    component_viz_img = np.zeros((blob.shape[0], blob.shape[1], 3), dtype=np.uint8)
    component_viz_img[labels != 0] = [255, 0, 0] # Default: Blue for valid components

    crop_h, crop_w = blob.shape
    crop_coords = [0, crop_h, 0, crop_w]

    if num_labels > 1:
        for i in range(1, num_labels):
            x_c, y_c, w_c, h_c, _ = stats[i]
            
            is_touching_boundary = (y_c == 0 or y_c + h_c == crop_h) if line_type != 'vertical' else (x_c == 0 or x_c + w_c == crop_w)
            is_size_constrained = (h_c <= config['CC_SIZE_THRESHOLD_RATIO'] * crop_h) if line_type != 'vertical' else (w_c <= config['CC_SIZE_THRESHOLD_RATIO'] * crop_w)

            if is_touching_boundary and is_size_constrained:
                component_viz_img[labels == i] = [0, 0, 255] # Red for noise
                if line_type != 'vertical':
                    if y_c == 0: crop_coords[0] = max(crop_coords[0], y_c + h_c)
                    if y_c + h_c == crop_h: crop_coords[1] = min(crop_coords[1], y_c)
                else:
                    if x_c == 0: crop_coords[2] = max(crop_coords[2], x_c + w_c)
                    if x_c + w_c == crop_w: crop_coords[3] = min(crop_coords[3], x_c)
    
    if crop_coords[0] >= crop_coords[1] or crop_coords[2] >= crop_coords[3]:
        return np.array([]), np.array([]), [0, 0, 0, 0]

    top, bottom, left, right = crop_coords
    cleaned_blob = blob[top:bottom, left:right]
    final_viz_img = component_viz_img[top:bottom, left:right]

    return cleaned_blob, final_viz_img, crop_coords

def gen_bounding_boxes(det, binarize_threshold):
    img = np.uint8(det)
    binarize_t = int(binarize_threshold * 255)
    _, img1 = cv2.threshold(img, binarize_t, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours]

def load_node_features_and_labels(points_file, labels_file):
    # points = np.loadtxt(points_file, dtype=int)
    points = np.loadtxt(points_file, dtype=float, ndmin=2).astype(int)
    with open(labels_file, "r") as f: labels = [line.strip() for line in f]
    features, filtered_labels = [], []
    for point, label in zip(points, labels):
        if label.lower() != "none":
            features.append(point)
            filtered_labels.append(int(label))
    return np.array(features), np.array(filtered_labels)

def assign_labels_and_plot(bounding_boxes, points, labels, image, output_path):
    if len(image.shape) == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    labeled_bboxes = []
    for x_min, y_min, w, h in bounding_boxes:
        x_max, y_max = x_min + w, y_min + h
        pts = [(p[0], p[1], lab) for p, lab in zip(points, labels) if x_min <= p[0] <= x_max and y_min <= p[1] <= y_max]
        if pts and len({lab for _, _, lab in pts}) == 1:
            labeled_bboxes.append((x_min, y_min, w, h, pts[0][2]))
        elif pts:
            pts.sort(key=lambda p: p[1])
            boundaries = [y_min] + [max(y_min, min(y_max, int((pts[i-1][1] + pts[i][1]) / 2))) for i in range(1, len(pts)) if pts[i][2] != pts[i-1][2]] + [y_max]
            for i in range(1, len(boundaries)):
                top, bot = boundaries[i-1], boundaries[i]
                seg_label = next((lab for _, py, lab in pts if top <= py <= bot), None)
                if seg_label: labeled_bboxes.append((x_min, top, w, bot - top, seg_label))
    for x, y, w, h, label in labeled_bboxes:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, str(label), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite(output_path, image)
    print(f"Annotated image saved as: {output_path}")
    return labeled_bboxes

def detect_line_type(boxes):
    if len(boxes) < 2: return 'horizontal', None
    centers = sorted([(x + w//2, y + h//2) for x, y, w, h, _ in boxes], key=lambda p: p[0])
    x_coords, y_coords = [p[0] for p in centers], [p[1] for p in centers]
    x_range, y_range = (max(coords) - min(coords) for coords in (x_coords, y_coords)) if centers else (0, 0)
    if x_range < y_range * 0.3: return 'vertical', None
    if y_range < x_range * 0.3: return 'horizontal', None
    try:
        X, y = np.array(x_coords).reshape(-1, 1), np.array(y_coords)
        ransac = RANSACRegressor(random_state=42).fit(X, y)
        if ransac.score(X, y) > 0.85: return 'slanted', {'slope': ransac.estimator_.coef_[0], 'intercept': ransac.estimator_.intercept_}
        return 'curved', {'spline': UnivariateSpline(x_coords, y_coords, s=len(centers)*2)}
    except: return 'horizontal', None

def transform_boxes_to_horizontal(boxes, line_type, params):
    if line_type == 'horizontal': return boxes
    t_boxes = []
    if line_type == 'vertical':
        for x, y, w, h, label in boxes: t_boxes.append((y, -x - w, h, w, label))
    elif line_type == 'slanted' and params:
        angle = math.atan(params['slope'])
        cos_a, sin_a = math.cos(-angle), math.sin(-angle)
        for x, y, w, h, label in boxes:
            cx, cy = x + w//2, y + h//2
            t_boxes.append((int(cx*cos_a - cy*sin_a - w/2), int(cx*sin_a + cy*cos_a - h/2), w, h, label))
    else: return boxes
    return t_boxes

def normalize_coordinates(boxes):
    if not boxes: return []
    min_x, min_y = min(b[0] for b in boxes), min(b[1] for b in boxes)
    return [(x - min_x, y - min_y, w, h, label) for x, y, w, h, label in boxes]

def crop_img(img):
    mask = img != int(np.median(img))
    if not np.any(mask): return img
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1]

def get_bboxes_for_lines(img, unique_labels, bounding_boxes, debug_mode=False, debug_info=None):
    """
    Generates cropped line images. Uses DYNAMIC PADDING for the initial cleaning crop
    and STATIC PADDING from the config for the final canvas border.
    """
    line_images_data = []
    line_bounding_boxes_data = {label: [] for label in unique_labels}
    box_counter = 0
    config = debug_info.get('CONFIG', {}) if debug_info else {}


    for l in unique_labels:
        filtered_boxes = [box for box in bounding_boxes if box[4] == l]
        if not filtered_boxes:
            continue

        line_type, params = detect_line_type(filtered_boxes)

        if line_type == 'horizontal':
            PADDING_RATIO_V = config.get('BBOX_PAD_V', 0.7) # Default to 70% if not in config
            PADDING_RATIO_H = config.get('BBOX_PAD_H', 0.5) # Default to 50% if not in config
        else:
            PADDING_RATIO_V = config.get('BBOX_PAD_H', 0.5) # Default to 50% if not in config
            PADDING_RATIO_H = config.get('BBOX_PAD_V', 0.7) # Default to 70% if not in config
     
        
        cleaned_blobs_for_line = []
        final_coords_for_line = []

        for box in filtered_boxes:
            box_counter += 1
            orig_x, orig_y, orig_w, orig_h, _ = box
            try:
                # Get the original, unpadded crop for debugging.
                original_uncropped_blob = img[orig_y:orig_y + orig_h, orig_x:orig_x + orig_w]
                
                # Use dynamic, ratio-based padding for the initial crop to help cleaning.
                dynamic_pad_v = int(orig_h * PADDING_RATIO_V)
                dynamic_pad_h = int(orig_w * PADDING_RATIO_H)

                y1 = max(0, orig_y - dynamic_pad_v)
                y2 = orig_y + orig_h + dynamic_pad_v
                x1 = max(0, orig_x - dynamic_pad_h)
                x2 = orig_x + orig_w + dynamic_pad_h

                blob = img[y1:y2, x1:x2]
                if blob.size == 0:
                    continue

                cleaned_blob, component_viz_img, crop_coords = analyze_and_clean_blob(blob, line_type, config)
                
                if cleaned_blob.size == 0:
                    if debug_mode and debug_info:
                        # Even if the blob is empty, save a debug collage to see why
                        det_resized = debug_info.get('det_resized')
                        if det_resized is not None:
                            heatmap_crop = det_resized[y1:y2, x1:x2]
                            collage = create_debug_collage(original_uncropped_blob, blob, cleaned_blob, 
                                                           component_viz_img, heatmap_crop, config)
                            cv2.imwrite(os.path.join(debug_info['DEBUG_DIR'], f"line_{l:03d}_box_{box_counter:04d}_EMPTY.jpg"), collage)
                    continue

                c_top, _, c_left, _ = crop_coords
                final_box_x = x1 + c_left
                final_box_y = y1 + c_top
                final_box_w = cleaned_blob.shape[1]
                final_box_h = cleaned_blob.shape[0]

                if final_box_w > 0 and final_box_h > 0:
                    cleaned_blobs_for_line.append(cleaned_blob)
                    final_coords_for_line.append([final_box_x, final_box_y, final_box_w, final_box_h])

                if debug_mode and debug_info:
                    det_resized = debug_info.get('det_resized')
                    if det_resized is not None:
                        heatmap_crop = det_resized[y1:y2, x1:x2]
                        if heatmap_crop.size > 0:
                            collage = create_debug_collage(original_uncropped_blob, blob, cleaned_blob,
                                                           component_viz_img, heatmap_crop, config)
                            cv2.imwrite(os.path.join(debug_info['DEBUG_DIR'], f"line_{l:03d}_box_{box_counter:04d}.jpg"), collage)

            except Exception as e:
                print(f"Warning: Skipped box during analysis in line {l}: {e}")
        
        line_bounding_boxes_data[l] = final_coords_for_line
        
    return line_bounding_boxes_data


def find_closest_points_between_contours(c1, c2):
    """
    Finds the closest pair of points between two contours using a robust, iterative approach.
    This method avoids NumPy broadcasting errors by comparing points individually.
    """
    min_dist = float('inf')
    closest_pts = (None, None)
    
    # Iterate through each point in the first contour
    for p1 in c1:
        # Iterate through each point in the second contour
        for p2 in c2:
            # Calculate the Euclidean distance between the two points.
            # p1[0] and p2[0] are used to access the (x, y) coordinates.
            dist = np.linalg.norm(p1[0] - p2[0])
            
            # If this distance is smaller than the smallest one found so far, update it.
            if dist < min_dist:
                min_dist = dist
                closest_pts = (tuple(p1[0]), tuple(p2[0]))
                
    return closest_pts, min_dist
# segment_from_point_clusters.py (Modified main function)



def segmentLinesFromPointClusters(manuscript_name, page, upscale_heatmap=True, debug_mode=True):
    BASE_PATH = os.path.join(current_app.config['DATA_PATH'], 'manuscripts')
    IMAGE_FILEPATH = os.path.join(BASE_PATH, manuscript_name, "leaves", f"{page}.jpg")
    HEATMAP_FILEPATH = os.path.join(BASE_PATH, manuscript_name, "heatmaps", f"{page}.jpg")
    GNN_DATA_PATH = os.path.join(BASE_PATH, manuscript_name, "gnn-dataset")
    POINTS_FILEPATH = os.path.join(GNN_DATA_PATH, f"{page}_inputs_unnormalized.txt")
    TEXTLINE_LABELS_FILEPATH = os.path.join(GNN_DATA_PATH, f"{page}_labels_textline.txt")
    LINES_DIR = os.path.join(BASE_PATH, manuscript_name, "lines", page)
    DEBUG_DIR = os.path.join(BASE_PATH, manuscript_name, "debug", page)
    POLYGON_DIR = os.path.join(BASE_PATH, manuscript_name, "polygons", page)
    POLY_VISUALIZATIONS_DIR = os.path.join(DEBUG_DIR, "poly_visualizations")

    textline_polygons = {}
    baselines = {}

    for d in [LINES_DIR, POLYGON_DIR]:
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d)

    image = loadImage(IMAGE_FILEPATH)
    if image is None:
        error_msg = f"Failed to load image at path: {IMAGE_FILEPATH}"
        current_app.logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    det = loadImage(HEATMAP_FILEPATH)
    if det is None:
        error_msg = f"Failed to load heatmap at path: {HEATMAP_FILEPATH}"
        current_app.logger.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    if det.ndim == 3: det = det[:, :, 0]

    h_img, w_img = image.shape[:2]; h_heat, w_heat = det.shape[:2]
    features, labels = load_node_features_and_labels(POINTS_FILEPATH, TEXTLINE_LABELS_FILEPATH)

    if upscale_heatmap:
        det_resized = cv2.resize(det, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
        processing_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if features.size > 0:
            features[:, :2] = (features[:, :2].astype(np.float64) * [w_img / w_heat, h_img / h_heat]).astype(int)
    else:
        image = cv2.resize(image, (w_heat, h_heat))
        h_img, w_img = image.shape[:2]
        det_resized = det
        processing_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    CONFIG = {
        'BINARIZE_THRESHOLD': 0.5098,
        'CC_SIZE_THRESHOLD_RATIO': 0.4, 
        'PAGE_MEDIAN_COLOR': int(np.median(processing_image)),
        'BBOX_PAD_V': 0.7, # 70% vertical padding for horizontal lines
        'BBOX_PAD_H': 0.5 # 50% horizontal
    }

    bounding_boxes = gen_bounding_boxes(det_resized, CONFIG['BINARIZE_THRESHOLD'])
    labeled_bboxes = assign_labels_and_plot(bounding_boxes, features, labels, image.copy(),
        output_path=os.path.join(BASE_PATH, manuscript_name, "frontend-graph-data", f"{page}.jpg"))

    unique_labels = sorted(list(set(b[4] for b in labeled_bboxes)))
    
    debug_info = None
    if upscale_heatmap and debug_mode:
        print("Debug mode is ON.")
        if os.path.exists(DEBUG_DIR): shutil.rmtree(DEBUG_DIR)
        os.makedirs(DEBUG_DIR)
        os.makedirs(POLY_VISUALIZATIONS_DIR)
        debug_info = {"DEBUG_DIR": DEBUG_DIR, "det_resized": det_resized, "CONFIG": CONFIG}

    line_bounding_boxes_data = get_bboxes_for_lines(processing_image, unique_labels, labeled_bboxes,
        debug_mode=(upscale_heatmap and debug_mode), debug_info=debug_info)
    
    poly_viz_page_img = image.copy()
    colors = [plt.cm.get_cmap('hsv', len(unique_labels) + 1)(i) for i in range(len(unique_labels))]
    label_to_color_idx = {label: i for i, label in enumerate(unique_labels)}

    baselines = _generate_baselines_from_textline_points(POINTS_FILEPATH, TEXTLINE_LABELS_FILEPATH)

    for line_label, cleaned_boxes in line_bounding_boxes_data.items():
        if not cleaned_boxes: continue

        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        for (x, y, w, h) in cleaned_boxes:
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        
        # --- START OF THE PRECISE FIX ---
        # First, find the initial disconnected contours
        initial_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Now, use your original, robust bridging logic to modify the mask
        if len(initial_contours) > 1:
            avg_line_height = np.mean([box[3] for box in cleaned_boxes])
            box_groups = [[] for _ in initial_contours]
            for box in cleaned_boxes:
                center_x, center_y = box[0] + box[2] // 2, box[1] + box[3] // 2
                for i, contour in enumerate(initial_contours):
                    point_to_test = (float(center_x), float(center_y))
                    if cv2.pointPolygonTest(contour, point_to_test, False) >= 0:
                        box_groups[i].append(box)
                        break
            
            box_groups = [group for group in box_groups if group]
            if len(box_groups) > 1:
                connected_groups = [box_groups[0]]
                unconnected_groups = box_groups[1:]

                while unconnected_groups:
                    best_dist = float('inf')
                    closest_box_pair = (None, None)
                    group_to_add_index = -1

                    for i, u_group in enumerate(unconnected_groups):
                        for c_group in connected_groups:
                            for u_box in u_group:
                                for c_box in c_group:
                                    dist = np.linalg.norm(
                                        np.array([u_box[0] + u_box[2]/2, u_box[1] + u_box[3]/2]) -
                                        np.array([c_box[0] + c_box[2]/2, c_box[1] + c_box[3]/2])
                                    )
                                    if dist < best_dist:
                                        best_dist = dist
                                        closest_box_pair = (u_box, c_box)
                                        group_to_add_index = i
                    
                    if closest_box_pair[0] is not None:
                        box1, box2 = closest_box_pair
                        left_box, right_box = (box1, box2) if box1[0] < box2[0] else (box2, box1)
                        
                        y_center1 = left_box[1] + left_box[3] / 2
                        y_center2 = right_box[1] + right_box[3] / 2
                        bridge_y_center = (y_center1 + y_center2) / 2
                        
                        bridge_y1 = int(bridge_y_center - avg_line_height / 2)
                        bridge_y2 = int(bridge_y_center + avg_line_height / 2)
                        bridge_x1 = left_box[0] + left_box[2]
                        bridge_x2 = right_box[0]
                        
                        # Your logic correctly modifies the mask here
                        cv2.rectangle(mask, (bridge_x1, bridge_y1), (bridge_x2, bridge_y2), 255, -1)

                    if unconnected_groups and group_to_add_index != -1:
                        connected_groups.append(unconnected_groups.pop(group_to_add_index))
                    else:
                        break # Exit if no more groups can be connected

        # After the mask has been finalized with bridges, find the definitive contour(s) from it.
        final_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if final_contours:
            # The largest contour on the FINAL mask is the correct polygon for the entire line.
            polygon = max(final_contours, key=cv2.contourArea)
            # --- END OF THE PRECISE FIX ---
            
            polygon_points_xy = [point[0].tolist() for point in polygon]
            textline_polygons[line_label] = polygon_points_xy

            line_filename_base = f"line{line_label+1:03d}"
            with open(os.path.join(POLYGON_DIR, f"{line_filename_base}.json"), 'w') as f:
                json.dump(polygon_points_xy, f)

            if upscale_heatmap and debug_mode:
                color_idx = label_to_color_idx.get(line_label, 0)
                color = tuple(c * 255 for c in colors[color_idx][:3])
                cv2.drawContours(poly_viz_page_img, [polygon], -1, color, 2)
            
            # The line saving logic now uses the same, complete polygon
            x, y, w, h = cv2.boundingRect(polygon) 
            cropped_line_image = processing_image[y:y+h, x:x+w]
            new_img = np.ones(cropped_line_image.shape, dtype=np.uint8) * CONFIG['PAGE_MEDIAN_COLOR']
            mask_polygon = np.zeros(cropped_line_image.shape[:2], dtype=np.uint8)
            polygon_shifted = polygon - [x, y]
            cv2.drawContours(mask_polygon, [polygon_shifted], -1, 255, -1)
            new_img[mask_polygon == 255] = cropped_line_image[mask_polygon == 255]
            cv2.imwrite(os.path.join(LINES_DIR, f"{line_filename_base}.jpg"), new_img)

    if upscale_heatmap and debug_mode:
        viz_path = os.path.join(POLY_VISUALIZATIONS_DIR, f"{page}_all_polygons.jpg")
        cv2.imwrite(viz_path, poly_viz_page_img)
        print(f"Polygon visualization saved to {viz_path}")

    print(f"Successfully generated and saved {len(unique_labels)} line images and associated data.")
    
    return textline_polygons, baselines