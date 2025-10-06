# routes.py

import os
import threading
import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime

from flask import Blueprint, request, send_from_directory, current_app, abort, send_file
import base64
from flask import Response
import json
import io

from werkzeug.utils import secure_filename
from PIL import Image
import torch
import gc
import numpy as np
import matplotlib.pyplot as plt
import cv2
from annotator.segmentation.segment_old_method import segment_lines
from annotator.segmentation.segment_from_point_clusters import segmentLinesFromPointClusters
from annotator.segmentation.segment_graph import handle_save_graph, handle_load_graph, generate_labels_from_graph, images2points
from annotator.recognition.recognition import recognise_characters,recognise_single_page_characters
from annotator.finetune.finetune import finetune
from natsort import natsorted
from collections import defaultdict

bp = Blueprint("main", __name__)

# setting up logging
import logging
from pythonjsonlogger import jsonlogger
# Create a logger instance
logger = logging.getLogger("backend_routes_logger")
logger.setLevel(logging.INFO)
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s')
# Attach formatter to handler, and handler to logger
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)


def _format_points(points):
    """Formats a list of [x, y] points into the 'x1,y1 x2,y2 ...' string format."""
    if not points or not isinstance(points, list):
        return ""
    return " ".join([f"{p[0]},{p[1]}" for p in points])

# routes.py

def create_page_xml(manuscript_name, page_name, image_dims, textline_polygons, baselines, region_to_textlines_map, output_dir):
    """
    Generates and saves a PAGE-XML file from segmentation data.
    This function now assumes it receives a correctly prepared region_to_textlines_map.
    """
    logger.info(f"Starting PAGE-XML generation for {manuscript_name}/{page_name}")
    assert isinstance(image_dims, dict) and 'width' in image_dims and 'height' in image_dims, "image_dims must be a dict with 'width' and 'height'."
    assert isinstance(textline_polygons, dict), "textline_polygons must be a dictionary."
    assert isinstance(baselines, dict), "baselines must be a dictionary."
    assert isinstance(region_to_textlines_map, dict), "region_to_textlines_map must be a dictionary."
    
    # ... (XML namespace and root element setup is unchanged) ...
    ns = {
        'pc': 'https://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15',
        'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
    }
    ET.register_namespace('', ns['pc'])
    ET.register_namespace('xsi', ns['xsi'])
    pc_gts = ET.Element(f"{{{ns['pc']}}}PcGts", {
        f"{{{ns['xsi']}}}schemaLocation": f"{ns['pc']} {ns['pc']}/pagecontent.xsd"
    })
    metadata = ET.SubElement(pc_gts, f"{{{ns['pc']}}}Metadata")
    creator = ET.SubElement(metadata, f"{{{ns['pc']}}}Creator")
    creator.text = "Sanskrit Manuscript Annotation Tool"
    created = ET.SubElement(metadata, f"{{{ns['pc']}}}Created")
    created.text = datetime.utcnow().isoformat() + "Z"
    last_change = ET.SubElement(metadata, f"{{{ns['pc']}}}LastChange")
    last_change.text = created.text
    page_el = ET.SubElement(pc_gts, f"{{{ns['pc']}}}Page", {
        'imageWidth': str(image_dims['width']),
        'imageHeight': str(image_dims['height']),
        'imageFilename': f"{page_name}.jpg"
    })

    # --- START OF MODIFICATION: REMOVED FALLBACK LOGIC ---
    # The map is now expected to be correct when it arrives.
    if not region_to_textlines_map:
        logger.warning("No regions or textlines found to generate PAGE-XML content.")
    # --- END OF MODIFICATION ---

    # Create TextRegions and TextLines
    for region_id, textline_ids in sorted(region_to_textlines_map.items()):
        if not textline_ids:
            logger.warning(f"Region {region_id} has no associated textlines, skipping.")
            continue
            
        all_region_points = []
        for line_id in textline_ids:
            if line_id in textline_polygons:
                all_region_points.extend(textline_polygons[line_id])

        if not all_region_points:
            logger.warning(f"Could not find any polygon points for textlines in region {region_id}, skipping region.")
            continue

        hull = cv2.convexHull(np.array(all_region_points, dtype=np.int32))
        region_points_str = _format_points([p[0].tolist() for p in hull])

        text_region = ET.SubElement(page_el, f"{{{ns['pc']}}}TextRegion", {'id': f'region_{region_id}', 'custom': '0'})
        ET.SubElement(text_region, f"{{{ns['pc']}}}Coords", {'points': region_points_str})
        
        for line_id in sorted(textline_ids):
            if line_id not in textline_polygons or line_id not in baselines:
                logger.warning(f"Missing polygon or baseline for line_id {line_id} in region {region_id}, skipping.")
                continue

            textline_el = ET.SubElement(text_region, f"{{{ns['pc']}}}TextLine", {'id': f'line_{region_id}_{line_id}', 'custom': '0'})
            line_points_str = _format_points(textline_polygons[line_id])
            ET.SubElement(textline_el, f"{{{ns['pc']}}}Coords", {'points': line_points_str})
            baseline_points_str = _format_points(baselines[line_id])
            ET.SubElement(textline_el, f"{{{ns['pc']}}}Baseline", {'points': baseline_points_str})
            text_equiv = ET.SubElement(textline_el, f"{{{ns['pc']}}}TextEquiv")
            unicode_el = ET.SubElement(text_equiv, f"{{{ns['pc']}}}Unicode")
            unicode_el.text = ""

        region_text_equiv = ET.SubElement(text_region, f"{{{ns['pc']}}}TextEquiv")
        region_unicode_el = ET.SubElement(region_text_equiv, f"{{{ns['pc']}}}Unicode")
        region_unicode_el.text = ""

    # ... (XML saving logic is unchanged) ...
    xml_string = ET.tostring(pc_gts, 'utf-8')
    reparsed = minidom.parseString(xml_string)
    pretty_xml = reparsed.toprettyxml(indent="  ", encoding="UTF-8")
    output_filepath = os.path.join(output_dir, f"{page_name}.xml")
    with open(output_filepath, "wb") as f:
        f.write(pretty_xml)
    logger.info(f"Successfully saved PAGE-XML to {output_filepath}")

@bp.route("/", methods=["GET"])
def hello():
    return "Sanskrit Manuscript Annotation Tool"

# GET AVAILABLE RECOGNITION MODELS
@bp.route("/models", methods=["GET"])
def get_models():
    current_app.logger.info("Getting Available text recogntion models")
    return os.listdir(os.path.join(current_app.config['DATA_PATH'], 'models', 'recognition'))


# NEW MANUSCRIPT PROCESSING
@bp.route("/new-process-manuscript", methods=["POST"])
def new_process_manuscript():
    current_app.logger.info("Processing new Manuscript, converting to heatmap, and saving character 2D Points")
    MANUSCRIPTS_PATH = os.path.join(current_app.config['DATA_PATH'], 'manuscripts')
    manuscript_name = request.form["manuscript_name"]
    folder_path = os.path.join(MANUSCRIPTS_PATH, manuscript_name)
    leaves_folder_path = os.path.join(folder_path, "leaves")

    try:
        os.makedirs(leaves_folder_path, exist_ok=True)
    except Exception as e:
        print(f"An error occurred: {e}")

    for file_key in request.files:
        uploaded_file = request.files[file_key]
        original_filename = uploaded_file.filename
        base_filename = os.path.splitext(original_filename)[0]

        # Open uploaded image file as a PIL image
        image = Image.open(uploaded_file)

        # --- MODIFICATION START ---
        # Check if the image is too big (height or width > 3000 pixels)
        width, height = image.size
        if width > 3000 or height > 3000:
            print(f"Image '{original_filename}' is too large ({width}x{height}). Downscaling by 50%.")
            new_width = width // 2
            new_height = height // 2
            
            try:
                from PIL import Image as PILImage
                resampling_filter = PILImage.Resampling.LANCZOS
            except AttributeError:
                resampling_filter = Image.LANCZOS

            image = image.resize((new_width, new_height), resampling_filter)
        # --- MODIFICATION END ---

        if image.mode in ("RGBA", "P", "LA"):
            image = image.convert("RGB")

        new_filename = f"{base_filename}.jpg"
        image.save(os.path.join(leaves_folder_path, new_filename), "JPEG")
        print(f"Saved: {new_filename}")

    images2points(os.path.join(folder_path, "leaves")) 
    torch.cuda.empty_cache()
    gc.collect()

    return Response(json.dumps({"message": "Files uploaded and points processing initiated."}), status=200, mimetype='application/json')


# NEW ENDPOINT to get pages for a manuscript
@bp.route("/manuscript/<string:manuscript_name>/pages", methods=["GET"])
def get_manuscript_pages(manuscript_name):
    """Returns a sorted list of page names (without extension) for a given manuscript."""
    current_app.logger.info(f"Fetching pages for manuscript: {manuscript_name}")
    MANUSCRIPTS_PATH = os.path.join(current_app.config['DATA_PATH'], 'manuscripts')
    leaves_folder_path = os.path.join(MANUSCRIPTS_PATH, manuscript_name, "leaves")

    if not os.path.isdir(leaves_folder_path):
        current_app.logger.error(f"Leaves folder not found for manuscript: {manuscript_name}")
        abort(404, description=f"Manuscript '{manuscript_name}' not found or has no pages.")

    try:
        page_files = [
            os.path.splitext(f)[0] for f in os.listdir(leaves_folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))
        ]
        sorted_pages = natsorted(page_files)
        return json.dumps(sorted_pages)
        
    except Exception as e:
        current_app.logger.error(f"Error reading pages for manuscript {manuscript_name}: {e}")
        return json.dumps({"error": "Could not retrieve page list."}), 500


# AUTO GENERATE GRAPH or load previously UPDATED GRAPH
@bp.route("/semi-segment/<manuscript_name>/<page>", methods=["GET"])
def get_node_features_and_graph(manuscript_name, page):
    current_app.logger.info("Getting Manuscript Page, Points and previously updated graph (if available)")
    MANUSCRIPTS_PATH = os.path.join(current_app.config['DATA_PATH'], 'manuscripts')
    GNN_DATASET_PATH = os.path.join(MANUSCRIPTS_PATH, manuscript_name, "gnn-dataset")
    IMAGE_FILEPATH = os.path.join(MANUSCRIPTS_PATH, manuscript_name, "leaves", f"{page}.jpg")
    POINTS_FILEPATH = os.path.join(GNN_DATASET_PATH, f"{page}_inputs_unnormalized.txt")
    REGION_LABELS_FILEPATH = os.path.join(GNN_DATASET_PATH, f"{page}_labels_region.txt")
    GRAPH_FILEPATH = os.path.join(
        MANUSCRIPTS_PATH, manuscript_name, "frontend-graph-data"
    )

    try:
        image = plt.imread(IMAGE_FILEPATH)
        image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
        height, width = image.shape[:2]
        _image = Image.fromarray((image * 255).astype(np.uint8)) if image.dtype == np.float32 else Image.fromarray(image)
        if _image.mode != "RGB":
            _image = _image.convert("RGB")
        response = {"dimensions": [width, height]}
        buffered = io.BytesIO()
        _image.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        response["image"] = img_str
        
        if not os.path.exists(POINTS_FILEPATH):
            return {"error": "2D Points not found"}, 404
        with open(POINTS_FILEPATH, "r") as f:
            points_raw = [row.strip().split() for row in f.readlines()]
        points = [[float(coord) for coord in point] for point in points_raw]
        response["points"] = points

        graph_file_name = f"{page}_graph_updated.pt"
        full_file_path = os.path.join(GRAPH_FILEPATH, graph_file_name)
        if os.path.exists(full_file_path):
            graph_data = handle_load_graph(
                page_number=page,
                input_dir=GRAPH_FILEPATH,
                update=True
            )
            current_app.logger.info("Loaded existing graph")
            response["graph"] = graph_data
        else:
            print(f"Existing graph not found: {full_file_path}, graph will be generated in frontend")
        
        # --- Load Region Labels if they exist ---
        if os.path.exists(REGION_LABELS_FILEPATH):
            with open(REGION_LABELS_FILEPATH, "r") as f:
                # Read labels, strip whitespace, and convert to integer
                region_labels = [int(line.strip()) for line in f if line.strip()]
                response["region_labels"] = region_labels
                current_app.logger.info(f"Loaded region labels for {page}")


        return response, 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}, 500


# SAVING AUTO GENERATED GRAPH
@bp.route("/save-graph/<manuscript_name>/<page>", methods=["POST"])
def save_graph(manuscript_name, page):
    MANUSCRIPTS_PATH = os.path.join(current_app.config['DATA_PATH'], 'manuscripts')
    try:
        data = request.get_json()
        graph_data = data.get('graph')
        
        if not graph_data:
            return {"error": "No graph data provided"}, 400
        
        GRAPH_FILEPATH = os.path.join(
            MANUSCRIPTS_PATH, manuscript_name, "frontend-graph-data"
        )
        handle_save_graph(graph_data, manuscript_name, page, output_dir=GRAPH_FILEPATH)
        current_app.logger.info(f"Saving autogenerated graph for {manuscript_name}, page {page}")
        return {"success": True}, 200
        
    except Exception as e:
        print(f"Error saving graph: {str(e)}")
        return {"error": str(e)}, 500


# SAVE UPDATED GRAPH (after adding/deleting edges), SEGMENT LINES, and then RECOGNIZE text content
@bp.route("/semi-segment/<string:manuscript_name>/<string:page>", methods=["POST"])
def make_semi_segments(manuscript_name, page):
    try:
        MANUSCRIPTS_PATH = os.path.join(current_app.config['DATA_PATH'], 'manuscripts')
        GNN_DATASET_PATH = os.path.join(MANUSCRIPTS_PATH, manuscript_name, "gnn-dataset")
        TEXTLINE_LABELS_FILEPATH = os.path.join(GNN_DATASET_PATH, f"{page}_labels_textline.txt")
        REGION_LABELS_FILEPATH = os.path.join(GNN_DATASET_PATH, f"{page}_labels_region.txt")
        IMAGE_FILEPATH = os.path.join(MANUSCRIPTS_PATH, manuscript_name, "leaves", f"{page}.jpg")
        XML_OUTPUT_DIR = os.path.join(MANUSCRIPTS_PATH, manuscript_name, "page-xml")

        os.makedirs(os.path.dirname(TEXTLINE_LABELS_FILEPATH), exist_ok=True)
        os.makedirs(XML_OUTPUT_DIR, exist_ok=True)
        
        GRAPH_FILEPATH = os.path.join(
            MANUSCRIPTS_PATH, manuscript_name, "frontend-graph-data"
        )
        
        request_data = request.json

        if 'graph' in request_data:
            # ... (graph saving logic is unchanged) ...
            graph_data = request_data['graph']
            logger.info(f"Saving updated Graph for: {manuscript_name}/{page}.")
            handle_save_graph(graph_data, manuscript_name, page, output_dir=GRAPH_FILEPATH, update=True)
            logger.info(f"Generating Labels from updated Graph for: {manuscript_name}/{page}.")
            labels = generate_labels_from_graph(graph_data)
            with open(TEXTLINE_LABELS_FILEPATH, "w") as f:
                f.write("\n".join(map(str, labels)))
        
        if 'regionLabels' in request_data and request_data['regionLabels']:
            # ... (region label saving logic is unchanged) ...
            labels_data = request_data['regionLabels']
            with open(REGION_LABELS_FILEPATH, "w") as f:
                f.write("\n".join(map(str, labels_data)))
            logger.info(f"Saved region labels for {manuscript_name}/{page}.")

        textline_polygons, baselines = segmentLinesFromPointClusters(manuscript_name, page)
        logger.info(f"Line Segmentation complete for {manuscript_name}/{page}.")
        assert textline_polygons is not None, "segmentLinesFromPointClusters must return textline polygons."
        assert baselines is not None, "segmentLinesFromPointClusters must return baselines."

        # --- PAGE-XML Generation ---
        logger.info("Proceeding with PAGE-XML generation.")
        try:
            image = cv2.imread(IMAGE_FILEPATH)
            if image is None: raise FileNotFoundError(f"Image not found at {IMAGE_FILEPATH}")
            h, w, _ = image.shape
            image_dims = {'width': w, 'height': h}

            # --- START OF REVISED LOGIC ---
            region_to_textlines_map = defaultdict(list)
            textline_to_region_map = {}
            
            # Build map only if region labels exist and are meaningful
            if os.path.exists(REGION_LABELS_FILEPATH) and os.path.exists(TEXTLINE_LABELS_FILEPATH):
                with open(REGION_LABELS_FILEPATH, "r") as f:
                    region_labels = [int(line.strip()) for line in f if line.strip()]
                with open(TEXTLINE_LABELS_FILEPATH, "r") as f:
                    textline_labels = [int(line.strip()) for line in f if line.strip()]

                # Check if there are any actual region labels other than -1
                has_valid_regions = any(r != -1 for r in region_labels)

                if has_valid_regions and len(region_labels) == len(textline_labels):
                    logger.info("Mapping textlines to provided regions.")
                    for i, tl_id in enumerate(textline_labels):
                        if tl_id != -1 and tl_id not in textline_to_region_map:
                            region_id = region_labels[i]
                            if region_id != -1:
                                textline_to_region_map[tl_id] = region_id
                    
                    for tl_id, r_id in textline_to_region_map.items():
                        region_to_textlines_map[r_id].append(tl_id)

            # If the map is still empty after the above logic, it means no regions were defined.
            # Create a single default region containing all detected textlines.
            if not region_to_textlines_map and textline_polygons:
                logger.warning("No valid region labels found. Grouping all textlines into a single default region.")
                all_line_ids = sorted(list(textline_polygons.keys()))
                region_to_textlines_map[0] = all_line_ids
            # --- END OF REVISED LOGIC ---

            create_page_xml(
                manuscript_name=manuscript_name,
                page_name=page,
                image_dims=image_dims,
                textline_polygons=textline_polygons,
                baselines=baselines,
                region_to_textlines_map=dict(region_to_textlines_map),
                output_dir=XML_OUTPUT_DIR
            )
        except Exception as xml_e:
            logger.error(f"Error during PAGE-XML generation: {str(xml_e)}", exc_info=True)
        
        # ... (Text recognition and response logic is unchanged) ...
        recognized_line_data = {}
        model_name_from_request = request_data.get("modelName")
        if model_name_from_request:
            current_app.logger.info(f"Starting text recognition for {manuscript_name}/{page} with model {model_name_from_request}.")
            manuscript_folder_path = os.path.join(MANUSCRIPTS_PATH, manuscript_name)
            recognized_line_data = recognise_single_page_characters(
                manuscript_folder_path, model_name_from_request, manuscript_name, page
            )
            current_app.logger.info(f"Text recognition finished for {manuscript_name}/{page}.")
        else:
            current_app.logger.info("No model name provided. Skipping text recognition step.")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
        return Response(json.dumps({
            "message": f"Updated Graph and Segmentation for: {manuscript_name} page {page}",
            "lines": recognized_line_data
        }), status=200, mimetype='application/json')

    except Exception as e:
        current_app.logger.error(f"Error in POST /semi-segment: {str(e)}", exc_info=True)
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')
    

# GET LINE IMAGES
@bp.route("/line-images/<manuscript_name>/<page>/<line>", methods=["GET"])
def serve_line_image(manuscript_name, page, line):
    current_app.logger.info(f"Getting line image ({line}) in  manuscript {manuscript_name},page {page}")
    base_dir   = current_app.config['DATA_PATH']
    folder     = os.path.join(base_dir, 'manuscripts', manuscript_name, 'lines', page)
    filename   = f"{line}.jpg" 

    absolute_path = os.path.abspath(os.path.join(folder, filename))
    if not os.path.exists(absolute_path):
        current_app.logger.error(f"Line image not found at path {absolute_path}")
        abort(404)
    return send_file(absolute_path, mimetype='image/jpeg')


# FINE TUNING
def finetune_context(data, app_context):
    with app_context:
        finetune(data)

@bp.route("/fine-tune", methods=["POST"])
def do_finetune():
    current_app.logger.info("Finetuning Recognition Model")
    thread = threading.Thread(
        target=finetune_context, args=(request.json, current_app.app_context())
    )
    thread.start()
    return "Success", 200


# ALL OLD FUNCTIONS BELOW
@bp.route("/uploaded-manuscripts", methods=["GET"])
def get_manuscripts():
    current_app.logger.info("Getting list of already uploaded manuscripts")
    MANUSCRIPTS_PATH = os.path.join(current_app.config['DATA_PATH'], 'manuscripts')
    return os.listdir(MANUSCRIPTS_PATH)

@bp.route("/recognise", methods=["POST"])
def recognise_manuscript():
    current_app.logger.info("Recognizing text content from line images cropped from all pages of the manuscript")
    MANUSCRIPTS_PATH = os.path.join(current_app.config['DATA_PATH'], 'manuscripts')
    manuscript_name = request.json.get("manuscript_name")
    model = request.json.get("model")
    folder_path = os.path.join(MANUSCRIPTS_PATH, manuscript_name)
    lines = recognise_characters(folder_path, model, manuscript_name)
    return lines, 200

# OLD LINE SEGMENTATION FOR SIMPLE LAYOUT
@bp.route("/upload-manuscript", methods=["POST"])
def annotate():
    MANUSCRIPTS_PATH = os.path.join(current_app.config['DATA_PATH'], 'manuscripts')
    manuscript_name = request.form["manuscript_name"]
    model = request.form["model"]
    folder_path = os.path.join(MANUSCRIPTS_PATH, manuscript_name)
    leaves_folder_path = os.path.join(folder_path, "leaves")

    try:
        os.makedirs(leaves_folder_path, exist_ok=True)
    except Exception as e:
        print(f"An error occured: {e}")

    for file in request.files:
        filename = request.files[file].filename
        request.files[file].save(os.path.join(leaves_folder_path, filename))

    print("image2heatmap2points")
    images2points(os.path.join(folder_path, "leaves"))
    print("now segmenting lines the old way")
    segment_lines(os.path.join(folder_path, "leaves"))
    lines = recognise_characters(folder_path, model, manuscript_name)
    torch.cuda.empty_cache()
    gc.collect()
    return lines, 200