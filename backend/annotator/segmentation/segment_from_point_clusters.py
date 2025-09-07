import os
import shutil
import numpy as np
import cv2
from sklearn.linear_model import RANSACRegressor
from scipy.interpolate import UnivariateSpline
import math
from annotator.segmentation.utils import loadImage
from flask import current_app

def resize_with_padding(image, target_size, background_color=(0, 0, 0)):
    """
    Resizes an image to a target size while maintaining its aspect ratio by padding.
    """
    target_w, target_h = target_size
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

def create_debug_collage(original_crop, heatmap_crop, line_type, config):
    """
    Creates a 2x4 collage of debugging images for a single bounding box.
    Includes a specialized connected components visualization.
    """
    TILE_SIZE = (200, 200)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.4
    FONT_COLOR = (255, 255, 255)

    # --- Binarize Images ---
    _, bin_orig_crop = cv2.threshold(original_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, bin_heat_crop = cv2.threshold(heatmap_crop, config['BINARIZE_THRESHOLD'], 255, cv2.THRESH_BINARY)
    
    # --- Connected Components Visualization ---
    crop_h, crop_w = bin_orig_crop.shape
    # Create a color image to draw components on
    colored_components_img = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_orig_crop, connectivity=8)
    
    # Default color for all components is blue
    colored_components_img[labels != 0] = [255, 0, 0] 

    # Iterate through each component (label 0 is the background)
    for i in range(1, num_labels):
        x_c, y_c, w_c, h_c, _ = stats[i]
        
        is_touching_boundary = False
        is_size_constrained = False
        
        if line_type == 'vertical':
            # For vertical text, check left/right boundaries and width constraint
            is_touching_boundary = (x_c == 0) or (x_c + w_c == crop_w)
            is_size_constrained = (w_c <= config['CC_SIZE_THRESHOLD_RATIO'] * crop_w)
        else: # Default to horizontal
            # For horizontal text, check top/bottom boundaries and height constraint
            is_touching_boundary = (y_c == 0) or (y_c + h_c == crop_h)
            is_size_constrained = (h_c <= config['CC_SIZE_THRESHOLD_RATIO'] * crop_h)

        # If both criteria are met, color the component red
        if is_touching_boundary and is_size_constrained:
            colored_components_img[labels == i] = [0, 0, 255]

    # --- Projection Profiles (from binarized images) ---
    h_prof_orig = np.sum(bin_orig_crop, axis=1) / 255
    v_prof_orig = np.sum(bin_orig_crop, axis=0) / 255
    h_prof_heat = np.sum(bin_heat_crop, axis=1) / 255
    v_prof_heat = np.sum(bin_heat_crop, axis=0) / 255

    # --- Create Visualizations for Each Component ---
    v_prof_orig_viz = visualize_projection_profile(v_prof_orig, bin_orig_crop.shape, 'vertical', color=(0, 255, 0))
    h_prof_orig_viz = visualize_projection_profile(h_prof_orig, bin_orig_crop.shape, 'horizontal', color=(0, 255, 0))
    v_prof_heat_viz = visualize_projection_profile(v_prof_heat, bin_heat_crop.shape, 'vertical', color=(0, 0, 255))
    h_prof_heat_viz = visualize_projection_profile(h_prof_heat, bin_heat_crop.shape, 'horizontal', color=(0, 0, 255))

    # --- Create Padded Tiles for the Collage ---
    orig_crop_tile = resize_with_padding(original_crop, TILE_SIZE)
    v_prof_orig_tile = resize_with_padding(v_prof_orig_viz, TILE_SIZE)
    h_prof_orig_tile = resize_with_padding(h_prof_orig_viz, TILE_SIZE)
    cc_tile = resize_with_padding(colored_components_img, TILE_SIZE)

    heatmap_colorized = cv2.applyColorMap(heatmap_crop, cv2.COLORMAP_JET)
    heat_crop_tile = resize_with_padding(heatmap_colorized, TILE_SIZE)
    v_prof_heat_tile = resize_with_padding(v_prof_heat_viz, TILE_SIZE)
    h_prof_heat_tile = resize_with_padding(h_prof_heat_viz, TILE_SIZE)
    bin_heat_tile = resize_with_padding(bin_heat_crop, TILE_SIZE)
    
    # Add labels
    cv2.putText(orig_crop_tile, "Original", (5, 15), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(v_prof_orig_tile, "V-Profile", (5, 15), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(h_prof_orig_tile, "H-Profile", (5, 15), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(cc_tile, "Analyzed Components", (5, 15), FONT, FONT_SCALE, FONT_COLOR, 1)
    
    cv2.putText(heat_crop_tile, "Heatmap", (5, 15), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(v_prof_heat_tile, "V-Profile", (5, 15), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(h_prof_heat_tile, "H-Profile", (5, 15), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(bin_heat_tile, "Binarized Heatmap", (5, 15), FONT, FONT_SCALE, FONT_COLOR, 1)

    # --- Assemble Final Collage ---
    row1 = cv2.hconcat([orig_crop_tile, v_prof_orig_tile, h_prof_orig_tile, cc_tile])
    row2 = cv2.hconcat([heat_crop_tile, v_prof_heat_tile, h_prof_heat_tile, bin_heat_tile])
    collage = cv2.vconcat([row1, row2])
    
    return collage

def gen_bounding_boxes(det, binarize_threshold):
    img = np.uint8(det)
    _, img1 = cv2.threshold(img, binarize_threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    return bounding_boxes

def load_node_features_and_labels(points_file, labels_file):
    points = np.loadtxt(points_file, dtype=int)
    with open(labels_file, "r") as f:
        labels = [line.strip() for line in f]
    filtered_node_features, filtered_labels = [], []
    for point, label in zip(points, labels):
        if label.lower() != "none":
            filtered_node_features.append(point)
            filtered_labels.append(int(label))
    return np.array(filtered_node_features), np.array(filtered_labels)

def assign_labels_and_plot(bounding_boxes, points, labels, image, output_path):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    labeled_bboxes = []
    for x_min, y_min, w, h in bounding_boxes:
        x_max, y_max = x_min + w, y_min + h
        pts_in_bbox = [(p[0], p[1], lab) for p, lab in zip(points, labels) if x_min <= p[0] <= x_max and y_min <= p[1] <= y_max]
        if pts_in_bbox and len({lab for _, _, lab in pts_in_bbox}) == 1:
            bbox_label = pts_in_bbox[0][2]
            labeled_bboxes.append((x_min, y_min, w, h, bbox_label))
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, str(bbox_label), (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        elif pts_in_bbox:
            pts_in_bbox.sort(key=lambda p: p[1])
            boundaries = [y_min]
            prev_label = pts_in_bbox[0][2]
            for i in range(1, len(pts_in_bbox)):
                if pts_in_bbox[i][2] != prev_label:
                    boundary = max(y_min, min(y_max, int((pts_in_bbox[i-1][1] + pts_in_bbox[i][1]) / 2)))
                    boundaries.append(boundary)
                    prev_label = pts_in_bbox[i][2]
            boundaries.append(y_max)
            for idx in range(1, len(boundaries)):
                seg_top, seg_bottom = boundaries[idx-1], boundaries[idx]
                seg_label = next((lab for _, py, lab in pts_in_bbox if seg_top <= py <= seg_bottom), None)
                if seg_label is not None:
                    new_h = seg_bottom - seg_top
                    labeled_bboxes.append((x_min, seg_top, w, new_h, seg_label))
                    cv2.rectangle(image, (x_min, seg_top), (x_max, seg_bottom), (0, 0, 255), 2)
                    cv2.putText(image, str(seg_label), (x_min, seg_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    for point, label in zip(points, labels):
        if label is not None:
            cv2.circle(image, (point[0], point[1]), 5, (0, 0, 255), -1)
            cv2.putText(image, str(label), (point[0] + 5, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
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
        r_squared = ransac.score(X, y)
        if r_squared > 0.85:
            return 'slanted', {'slope': ransac.estimator_.coef_[0], 'intercept': ransac.estimator_.intercept_}
        else:
            return 'curved', {'spline': UnivariateSpline(x_coords, y_coords, s=len(centers)*2), 'x_coords': x_coords, 'y_coords': y_coords}
    except: return 'horizontal', None

def transform_boxes_to_horizontal(boxes, line_type, params):
    if line_type == 'horizontal': return boxes
    transformed_boxes = []
    if line_type == 'vertical':
        for x, y, w, h, label in boxes: transformed_boxes.append((y, -x - w, h, w, label))
    elif line_type == 'slanted' and params:
        angle = math.atan(params['slope'])
        cos_a, sin_a = math.cos(-angle), math.sin(-angle)
        for x, y, w, h, label in boxes:
            cx, cy = x + w//2, y + h//2
            new_cx, new_cy = cx * cos_a - cy * sin_a, cx * sin_a + cy * cos_a
            transformed_boxes.append((int(new_cx - w//2), int(new_cy - h//2), w, h, label))
    elif line_type == 'curved' and params:
        min_x, max_x = min(params['x_coords']), max(params['x_coords'])
        for x, y, w, h, label in boxes:
            cx = x + w//2
            try:
                progress = (cx - min_x) / (max_x - min_x)
                transformed_boxes.append((int(progress * (max_x - min_x)), 0, w, h, label))
            except: transformed_boxes.append((x, y, w, h, label))
    else: return boxes
    return transformed_boxes

def normalize_coordinates(boxes):
    if not boxes: return []
    min_x = min(b[0] for b in boxes)
    min_y = min(b[1] for b in boxes)
    return [(x - min_x, y - min_y, w, h, label) for x, y, w, h, label in boxes]

def crop_img(img):
    mask = img != int(np.median(img))
    if not np.any(mask): return img
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1]

def gen_line_images(img, unique_labels, bounding_boxes, debug_mode=False, debug_info=None):
    line_images = []
    box_counter = 0
    config = debug_info.get('CONFIG', {}) if debug_info else {}
    pad_v = config.get('LINE_GEN_PAD_V', 15)
    pad_h = config.get('LINE_GEN_PAD_H', 10)

    for l in unique_labels:
        filtered_boxes = [box for box in bounding_boxes if box[4] == l]
        if not filtered_boxes: continue
        
        line_type, params = detect_line_type(filtered_boxes)
        transformed_boxes = normalize_coordinates(transform_boxes_to_horizontal(filtered_boxes, line_type, params))
        if not transformed_boxes: continue
        
        min_x, max_x = min(b[0] for b in transformed_boxes), max(b[0] + b[2] for b in transformed_boxes)
        min_y, max_y = min(b[1] for b in transformed_boxes), max(b[1] + b[3] for b in transformed_boxes)
        
        new_img = np.ones((max_y - min_y + 20 + (2 * pad_v), max_x - min_x + 40), dtype=np.uint8) * int(np.median(img))
        
        for (new_x, new_y, _, _, _), (orig_x, orig_y, orig_w, orig_h, _) in zip(transformed_boxes, filtered_boxes):
            box_counter += 1
            try:
                y1, y2 = max(0, orig_y - pad_v), orig_y + orig_h + pad_v
                x1, x2 = max(0, orig_x - pad_h), orig_x + orig_w + pad_h
                blob = img[y1:y2, x1:x2]
                if blob.size == 0: continue

                if debug_mode and debug_info:
                    det_resized = debug_info.get('det_resized')
                    DEBUG_DIR = debug_info.get('DEBUG_DIR')
                    if det_resized is not None and DEBUG_DIR:
                        heatmap_crop = det_resized[y1:y2, x1:x2]
                        if heatmap_crop.size > 0:
                            collage = create_debug_collage(blob, heatmap_crop, line_type, config)
                            cv2.imwrite(os.path.join(DEBUG_DIR, f"line_{l:03d}_box_{box_counter:04d}.jpg"), collage)
                
                if line_type == 'vertical': blob = cv2.rotate(blob, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                target_y, target_x = new_y - min_y + pad_v, new_x - min_x + 10
                h_b, w_b = blob.shape[:2]
                h_n, w_n = new_img.shape[:2]
                if target_y < h_n and target_x < w_n:
                    y_end, x_end = min(target_y + h_b, h_n), min(target_x + w_b, w_n)
                    new_img[target_y:y_end, target_x:x_end] = blob[:y_end-target_y, :x_end-target_x]
            except Exception as e: print(f"Warning: Skipped box: {e}")
        
        line_images.append(crop_img(new_img))
    return line_images

def segmentLinesFromPointClusters(manuscript_name, page, upscale_heatmap=True, debug_mode=True):
    # --- Centralized Configuration for Tunable Parameters ---
    CONFIG = {
        'BINARIZE_THRESHOLD': 100,
        'LINE_GEN_PAD_V': 5,          # Vertical padding for line image generation
        'LINE_GEN_PAD_H': 10,          # Horizontal padding for line image generation
        'CC_SIZE_THRESHOLD_RATIO': 0.4 # 40% threshold for connected component analysis
    }

    BASE_PATH = os.path.join(current_app.config['DATA_PATH'], 'manuscripts')
    IMAGE_FILEPATH = os.path.join(BASE_PATH, manuscript_name, "leaves", f"{page}.jpg")
    HEATMAP_FILEPATH = os.path.join(BASE_PATH, manuscript_name, "heatmaps", f"{page}.jpg")
    POINTS_FILEPATH = os.path.join(BASE_PATH, manuscript_name, "gnn-dataset", f"{page}_inputs_unnormalized.txt")
    LABELS_FILEPATH = os.path.join(BASE_PATH, manuscript_name, "gnn-dataset", f"{page}_labels_textline.txt")
    LINES_DIR = os.path.join(BASE_PATH, manuscript_name, "lines", page)
    DEBUG_DIR = os.path.join(BASE_PATH, manuscript_name, "debug", page)

    if not os.path.exists(LINES_DIR): os.makedirs(LINES_DIR)

    image = loadImage(IMAGE_FILEPATH)
    det = loadImage(HEATMAP_FILEPATH)
    if det.ndim == 3: det = det[:, :, 0]

    h_img, w_img = image.shape[:2]
    h_heat, w_heat = det.shape[:2]

    features, labels = load_node_features_and_labels(POINTS_FILEPATH, LABELS_FILEPATH)

    if upscale_heatmap:
        print("Upscaling heatmap to match original image resolution.")
        det_resized = cv2.resize(det, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
        processing_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # ### BUG FIX ###
        # Apply scaling only to the first two (x, y) columns, leaving other columns untouched.
        if features.size > 0:
            # Create a float copy for scaling calculation
            scaled_features = features.astype(np.float64)
            # Apply scaling only to the first two columns (x, y)
            scaled_features[:, :2] *= [w_img / w_heat, h_img / h_heat]
            # Convert back to int and reassign
            features = scaled_features.astype(int)
    else:
        print("Downscaling image to match heatmap resolution.")
        det_resized = det
        processing_image = cv2.cvtColor(cv2.resize(image, (w_heat, h_heat)), cv2.COLOR_BGR2GRAY)

    bounding_boxes = gen_bounding_boxes(det_resized, CONFIG['BINARIZE_THRESHOLD'])
    labeled_bboxes = assign_labels_and_plot(bounding_boxes, features, labels, processing_image.copy(),
        output_path=os.path.join(BASE_PATH, manuscript_name, "frontend-graph-data", f"{page}.jpg"))

    unique_labels = sorted(list(set(b[4] for b in labeled_bboxes)))
    
    debug_info = None
    if upscale_heatmap and debug_mode:
        print(f"Debug mode is ON. Saving bounding box collages to {DEBUG_DIR}")
        if os.path.exists(DEBUG_DIR): shutil.rmtree(DEBUG_DIR)
        os.makedirs(DEBUG_DIR)
        debug_info = {"DEBUG_DIR": DEBUG_DIR, "det_resized": det_resized, "CONFIG": CONFIG}

    line_images = gen_line_images(processing_image, unique_labels, labeled_bboxes,
        debug_mode=(upscale_heatmap and debug_mode), debug_info=debug_info)

    if upscale_heatmap and debug_mode: print("Finished saving debug images.")

    if os.path.exists(LINES_DIR): shutil.rmtree(LINES_DIR)
    os.makedirs(LINES_DIR)
    for i, line_img in enumerate(line_images):
        cv2.imwrite(os.path.join(LINES_DIR, f"line{i+1:03d}.jpg"), line_img)
    
    print(f"Successfully generated and saved {len(line_images)} line images in {LINES_DIR}")