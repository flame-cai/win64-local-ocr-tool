import os
import threading

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

        os.makedirs(os.path.dirname(TEXTLINE_LABELS_FILEPATH), exist_ok=True)
        
        GRAPH_FILEPATH = os.path.join(
            MANUSCRIPTS_PATH, manuscript_name, "frontend-graph-data"
        )
        
        request_data = request.json

        if 'graph' in request_data:
            graph_data = request_data['graph']
            current_app.logger.info(f"Saving updated Graph for: {manuscript_name}/{page}.")
            handle_save_graph(graph_data, manuscript_name, page, output_dir=GRAPH_FILEPATH, update=True)
            
            current_app.logger.info(f"Generating Labels from updated Graph for: {manuscript_name}/{page}.")
            labels = generate_labels_from_graph(graph_data)
            
            with open(TEXTLINE_LABELS_FILEPATH, "w") as f:
                f.write("\n".join(map(str, labels)))
        
        # --- Save Region Labels if provided ---
        if 'regionLabels' in request_data:
            region_labels = request_data['regionLabels']
            with open(REGION_LABELS_FILEPATH, "w") as f:
                f.write("\n".join(map(str, region_labels)))
            current_app.logger.info(f"Saved region labels for {manuscript_name}/{page}.")


        segmentLinesFromPointClusters(manuscript_name, page)
        current_app.logger.info(f"Line Segmentation complete with updated graph for {manuscript_name}/{page}.")

        # --- MODIFIED RECOGNITION LOGIC ---
        recognized_line_data = {}  # Default to an empty dictionary
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

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return Response(json.dumps({
            "message": f"Updated Graph and Segmentation for: {manuscript_name} page {page}",
            "lines": recognized_line_data
        }), status=200, mimetype='application/json')

    except Exception as e:
        current_app.logger.error(f"Error in POST /semi-segment: {str(e)}")
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