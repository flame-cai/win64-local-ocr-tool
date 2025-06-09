import os
import shutil
import numpy as np
import cv2
from sklearn.linear_model import RANSACRegressor
from scipy.interpolate import UnivariateSpline
import math
from annotator.segmentation.utils import loadImage
from flask import current_app


def gen_bounding_boxes(det, binarize_threshold):
    img = np.uint8(det)
    _, img1 = cv2.threshold(img, binarize_threshold, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    # Extract bounding boxes from contours
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))

    return bounding_boxes


def load_points_and_labels(points_file, labels_file):
    # Load points
    points = np.loadtxt(points_file, dtype=int)

    # Load labels, handling 'None' entries
    with open(labels_file, "r") as f:
        labels = [line.strip() for line in f]

    # Convert labels to integers where possible, otherwise mark as None
    filtered_points = []
    filtered_labels = []

    for point, label in zip(points, labels):
        if label.lower() != "none":  # Exclude 'None' labels
            filtered_points.append(point)
            filtered_labels.append(int(label))  # Convert valid labels to int

    return np.array(filtered_points), np.array(filtered_labels)



def assign_labels_and_plot(bounding_boxes, points, labels, image, output_path):
    """
    Assigns labels to given bounding boxes based on the labels of the points they contain. 
    If a bounding box contains points with different labels (typically in tall boxes),
    the bounding box is split maximally along the vertical direction into non-overlapping 
    sub-boxes such that each sub-box contains points of only one label. The result is visualized 
    by overlaying both the bounding boxes and the labeled points on the image.
    """
    # Convert image to color if it is grayscale.
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    labeled_bboxes = []
    for bbox in bounding_boxes:
        x_min, y_min, w, h = bbox
        x_max, y_max = x_min + w, y_min + h

        # Gather points (with labels) inside the bounding box.
        pts_in_bbox = [
            (px, py, lab)
            for (px, py), lab in zip(points, labels)
            if x_min <= px <= x_max and y_min <= py <= y_max
        ]

        # If all points inside have the same label, draw the original box (green).
        if pts_in_bbox and len({lab for (_, _, lab) in pts_in_bbox}) == 1:
            bbox_label = pts_in_bbox[0][2]
            labeled_bboxes.append((x_min, y_min, w, h, bbox_label))
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, str(bbox_label), (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Handle boxes with multiple labels: split maximally along the vertical axis.
        elif pts_in_bbox:
            # Sort points by vertical (y) coordinate.
            pts_in_bbox.sort(key=lambda p: p[1])
            boundaries = [y_min]
            prev_label = pts_in_bbox[0][2]

            # Compute split boundaries at label change.
            for i in range(1, len(pts_in_bbox)):
                current_label = pts_in_bbox[i][2]
                if current_label != prev_label:
                    boundary = int((pts_in_bbox[i-1][1] + pts_in_bbox[i][1]) / 2)
                    boundary = max(boundary, y_min)
                    boundary = min(boundary, y_max)
                    boundaries.append(boundary)
                    prev_label = current_label
            boundaries.append(y_max)

            # Create sub-boxes based on the computed boundaries.
            for idx in range(1, len(boundaries)):
                seg_top = boundaries[idx - 1]
                seg_bottom = boundaries[idx]
                seg_label = None
                for (px, py, lab) in pts_in_bbox:
                    if seg_top <= py <= seg_bottom:
                        seg_label = lab
                        break
                if seg_label is not None:
                    new_h = seg_bottom - seg_top
                    labeled_bboxes.append((x_min, seg_top, w, new_h, seg_label))
                    cv2.rectangle(image, (x_min, seg_top), (x_max, seg_bottom), (0, 0, 255), 2)
                    cv2.putText(image, str(seg_label), (x_min, seg_top - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw all points with their labels.
    for (px, py), label in zip(points, labels):
        if label is not None:
            cv2.circle(image, (px, py), 5, (0, 0, 255), -1)
            cv2.putText(image, str(label), (px + 5, py - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imwrite(output_path, image)
    print(f"Annotated image saved as: {output_path}")

    return labeled_bboxes



def detect_line_type(boxes):
    """Detect if boxes form horizontal, vertical, slanted, or curved line"""
    if len(boxes) < 2:
        return 'horizontal', None
    
    # Extract center points
    centers = [(x + w//2, y + h//2) for x, y, w, h, _ in boxes]
    centers.sort(key=lambda p: p[0])  # Sort by x-coordinate
    
    x_coords = [p[0] for p in centers]
    y_coords = [p[1] for p in centers]
    
    x_range = max(x_coords) - min(x_coords)
    y_range = max(y_coords) - min(y_coords)
    
    # Check if vertical (x doesn't change much, y changes significantly)
    if x_range < y_range * 0.3:
        return 'vertical', None
    
    # Check if horizontal (y doesn't change much, x changes significantly)
    if y_range < x_range * 0.3:
        return 'horizontal', None
    
    # For slanted/curved, fit a line and check linearity
    try:
        X = np.array(x_coords).reshape(-1, 1)
        y = np.array(y_coords)
        
        # Use RANSAC for robust line fitting
        ransac = RANSACRegressor(random_state=42)
        ransac.fit(X, y)
        
        # Calculate R² to determine if it's linear
        y_pred = ransac.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1
        
        if r_squared > 0.85:  # High linearity = slanted line
            slope = ransac.estimator_.coef_[0]
            intercept = ransac.estimator_.intercept_
            return 'slanted', {'slope': slope, 'intercept': intercept}
        else:  # Low linearity = curved line
            # Fit spline for curved line
            spline = UnivariateSpline(x_coords, y_coords, s=len(centers)*2)
            return 'curved', {'spline': spline, 'x_coords': x_coords, 'y_coords': y_coords}
            
    except:
        return 'horizontal', None

def transform_boxes_to_horizontal(boxes, line_type, params):
    """Transform boxes from various orientations to horizontal layout"""
    transformed_boxes = []
    
    if line_type == 'horizontal':
        return boxes
    
    elif line_type == 'vertical':
        # For vertical text, rotate 90 degrees and swap coordinates
        for x, y, w, h, label in boxes:
            # New coordinates after rotation
            new_x = y
            new_y = -x - w  # Negative to maintain reading order
            new_w = h
            new_h = w
            transformed_boxes.append((new_x, new_y, new_w, new_h, label))
    
    elif line_type == 'slanted' and params:
        slope = params['slope']
        intercept = params['intercept']
        angle = math.atan(slope)
        
        # Rotate boxes to make line horizontal
        cos_a = math.cos(-angle)
        sin_a = math.sin(-angle)
        
        for x, y, w, h, label in boxes:
            # Rotate center point
            cx, cy = x + w//2, y + h//2
            new_cx = cx * cos_a - cy * sin_a
            new_cy = cx * sin_a + cy * cos_a
            
            # For simplicity, keep original width/height (could be improved)
            new_x = int(new_cx - w//2)
            new_y = int(new_cy - h//2)
            transformed_boxes.append((new_x, new_y, w, h, label))
    
    elif line_type == 'curved' and params:
        spline = params['spline']
        x_coords = params['x_coords']
        
        # For curved lines, straighten by mapping each point
        for x, y, w, h, label in boxes:
            cx = x + w//2
            # Find position along the curve
            try:
                curve_progress = (cx - min(x_coords)) / (max(x_coords) - min(x_coords))
                # Map to horizontal position
                new_x = int(curve_progress * (max(x_coords) - min(x_coords)))
                new_y = 0  # All at same height for horizontal line
                transformed_boxes.append((new_x, new_y, w, h, label))
            except:
                transformed_boxes.append((x, y, w, h, label))
    
    else:
        return boxes
    
    return transformed_boxes

def normalize_coordinates(boxes):
    """Normalize coordinates to positive values"""
    if not boxes:
        return boxes
    
    min_x = min(x for x, _, _, _, _ in boxes)
    min_y = min(y for _, y, _, _, _ in boxes)
    
    return [(x - min_x, y - min_y, w, h, label) for x, y, w, h, label in boxes]

def gen_line_images(img2, unique_labels, bounding_boxes):
    """Generate line images with support for various text orientations"""
    line_images = []
    pad = 5
    
    for l in unique_labels:
        # Filter bounding boxes for the current label
        filtered_boxes = [box for box in bounding_boxes if box[4] == l]
        if not filtered_boxes:
            continue
        
        # Detect line orientation
        line_type, params = detect_line_type(filtered_boxes)
        
        # Transform boxes to horizontal layout
        transformed_boxes = transform_boxes_to_horizontal(filtered_boxes, line_type, params)
        transformed_boxes = normalize_coordinates(transformed_boxes)
        
        if not transformed_boxes:
            continue
        
        # Calculate dimensions for the new image - fix broadcasting bug
        min_x = min(x for x, _, _, _, _ in transformed_boxes)
        min_y = min(y for _, y, _, _, _ in transformed_boxes)
        max_x = max(x + w for x, _, w, _, _ in transformed_boxes)
        max_y = max(y + h for _, y, _, h, _ in transformed_boxes)
        
        # Ensure dimensions account for padding
        total_width = max_x - min_x + 40  # Extra padding
        total_height = max_y - min_y + 20 + (2 * pad)  # Account for blob padding
        
        # Create background image
        new_img = np.ones((total_height, total_width), dtype=np.uint8) * int(np.median(img2))
        
        # Place each character/box with bounds checking
        for (new_x, new_y, new_w, new_h, _), (orig_x, orig_y, orig_w, orig_h, _) in zip(transformed_boxes, filtered_boxes):
            try:
                # Extract from original image
                blob = img2[max(0, orig_y - pad):orig_y + orig_h + pad, 
                           max(0, orig_x - 10):orig_x + orig_w + 10]
                
                if blob.size == 0:
                    continue
                
                # Handle rotation for vertical text
                if line_type == 'vertical':
                    blob = cv2.rotate(blob, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # Calculate target position with bounds checking
                target_y = new_y - min_y + pad
                target_x = new_x - min_x + 10
                
                # Ensure we don't exceed image boundaries
                target_y_end = min(target_y + blob.shape[0], new_img.shape[0])
                target_x_end = min(target_x + blob.shape[1], new_img.shape[1])
                
                # Only proceed if we have valid target area
                if target_y < target_y_end and target_x < target_x_end:
                    blob_h = target_y_end - target_y
                    blob_w = target_x_end - target_x
                    
                    # Crop blob to fit if necessary
                    new_img[target_y:target_y_end, target_x:target_x_end] = blob[:blob_h, :blob_w]
                    
            except Exception as e:
                print(f"Warning: Skipped box due to error: {e}")
                continue
        
        line_images.append(crop_img(new_img))
    
    return line_images

def crop_img(img):
    """Crop image to remove excess whitespace"""
    # Find non-background pixels (assuming background is median value)
    background_val = int(np.median(img))
    mask = img != background_val
    
    if not np.any(mask):
        return img
    
    # Find bounding box of content
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    
    return img[y0:y1, x0:x1]

def segmentLinesFromPointClusters(manuscript_name, page):
    BASE_PATH = os.path.join(current_app.config['DATA_PATH'], 'manuscripts')
    IMAGE_FILEPATH = os.path.join(BASE_PATH, manuscript_name, "leaves", f"{page}.jpg")
    HEATMAP_FILEPATH = os.path.join(BASE_PATH, manuscript_name, "heatmaps", f"{page}.jpg")
    POINTS_FILEPATH = os.path.join(BASE_PATH, manuscript_name, "points-2D", f"{page}_points.txt")
    LABELS_FILEPATH = os.path.join(BASE_PATH, manuscript_name, "points-2D", f"{page}_labels.txt")

    # Check if the manuscript lines directory exists
    if os.path.exists(os.path.join(BASE_PATH, manuscript_name, "lines", page)) == False:
        os.makedirs(os.path.join(BASE_PATH, manuscript_name, "lines", page))
        print("making the lines directory")
    LINES_DIR = os.path.join(BASE_PATH, manuscript_name, "lines", page)

    image = loadImage(IMAGE_FILEPATH)
    det = loadImage(HEATMAP_FILEPATH)
    filtered_points, filtered_labels = load_points_and_labels(POINTS_FILEPATH, LABELS_FILEPATH)

    det = det.squeeze()
    print(det.shape)
    if len(det.shape) == 3:  
        det = det[:, :, 0]  # Keep only one channel
    print(det.shape)

    #print(image.shape) this is x2 scale
    img2 = cv2.cvtColor(cv2.resize(image, det.shape[::-1]), cv2.COLOR_BGR2GRAY) 

    binarize_threshold = 100
    bounding_boxes = gen_bounding_boxes(det, binarize_threshold)
    labeled_bboxes = assign_labels_and_plot(bounding_boxes, filtered_points, filtered_labels, img2, output_path=os.path.join(BASE_PATH, manuscript_name, "points-2D", f"{page}.jpg"))

    # Sort by the numeric label (5th element)
    # sorted_bboxes = sorted(labeled_bboxes, key=lambda x: x[4])

    # Get unique labels
    unique_labels = set(label for _, _, _, _, label in labeled_bboxes)
    # print(f"UNIQUE_LABELS: {unique_labels}")
    line_images = gen_line_images(img2,unique_labels,labeled_bboxes)

    shutil.rmtree(LINES_DIR)
    os.makedirs(LINES_DIR)

    for i in range(len(line_images)):
        cv2.imwrite(os.path.join(LINES_DIR, f"line{i+1:03d}.jpg"),line_images[i])


