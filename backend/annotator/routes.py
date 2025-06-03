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
from annotator.segmentation.segment_graph import save_graph_for_gnn, load_graph_for_gnn, generate_labels_from_graph, images2points
from annotator.recognition.recognition import recognise_characters,recognise_single_page_characters
from annotator.finetune.finetune import finetune

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
    current_app.logger.info("Processing new Manuscript, converting to heatmap, and saveing character 2D Points")
    MANUSCRIPTS_PATH = os.path.join(current_app.config['DATA_PATH'], 'manuscripts')
    manuscript_name = request.form["manuscript_name"]
    folder_path = os.path.join(MANUSCRIPTS_PATH, manuscript_name)
    leaves_folder_path = os.path.join(folder_path, "leaves")

    try:
        os.makedirs(leaves_folder_path, exist_ok=True)
    except Exception as e:
        print(f"An error occured: {e}")

    for file_key in request.files:
        uploaded_file = request.files[file_key]
        original_filename = uploaded_file.filename
        base_filename = os.path.splitext(original_filename)[0]

        # Open uploaded image file as a PIL image
        image = Image.open(uploaded_file)

        # Convert to RGB if needed (JPEG doesn't support some modes like RGBA)
        if image.mode in ("RGBA", "P", "LA"):
            image = image.convert("RGB")

        # Build new filename with .jpg extension
        new_filename = f"{base_filename}.jpg"

        # Save image as JPEG in leaves_folder_path
        image.save(os.path.join(leaves_folder_path, new_filename), "JPEG")

        print(f"Saved: {new_filename}")

    images2points(os.path.join(folder_path, "leaves"))
    torch.cuda.empty_cache()
    gc.collect()

    return Response(json.dumps({"message": "Files uploaded and points processing initiated."}), status=200, mimetype='application/json')




# AUTO GENERATE GRAPH or load previously UPDATED GRAPH
@bp.route("/semi-segment/<manuscript_name>/<page>", methods=["GET"])
def get_points_and_graph(manuscript_name, page):
    current_app.logger.info("Getting Manuscript Page, Points and previously updated graph (if available)")
    MANUSCRIPTS_PATH = os.path.join(current_app.config['DATA_PATH'], 'manuscripts')
    IMAGE_FILEPATH= os.path.join(MANUSCRIPTS_PATH, manuscript_name, "leaves", f"{page}.jpg")
    POINTS_FILEPATH = os.path.join(
        MANUSCRIPTS_PATH, manuscript_name, "points-2D", f"{page}_points.txt"
    )
    GRAPH_FILEPATH = os.path.join(
        MANUSCRIPTS_PATH, manuscript_name, "points-2D"
    )
    try:
        image = plt.imread(IMAGE_FILEPATH)
        image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2)) # resize image, because heatmap is half
        # Store original dimensions
        height, width = image.shape[:2]
        _image = Image.fromarray((image * 255).astype(np.uint8)) if image.dtype == np.float32 else Image.fromarray(image)
        # Convert to RGB if not already
        if _image.mode != "RGB":
            _image = _image.convert("RGB")
        # Send original dimensions in response
        response = {"dimensions": [width, height]}
        # Convert image to base64 for sending in response
        buffered = io.BytesIO()
        _image.save(buffered, format="JPEG", quality=85)  # Reduced quality for better performance
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        response["image"] = img_str
        
        
        if not os.path.exists(POINTS_FILEPATH):
            return {"error": "2D Points not found"}, 404
        # Load points from file
        with open(POINTS_FILEPATH, "r") as f:
            points_raw = [row.strip().split() for row in f.readlines()]
        # Convert to numeric values
        points = [[float(coord) for coord in point] for point in points_raw]
        # Always include points in response
        response["points"] = points

        # If graph already exist before, load it, else create a new graph in frontend
        graph_file_name = f"{manuscript_name}_page{page}_graph_updated.pt"
        full_file_path = os.path.join(GRAPH_FILEPATH, graph_file_name)
        # Check if the file exists and load it
        if os.path.exists(full_file_path):
            graph_data = load_graph_for_gnn(
                manuscript_name=manuscript_name,
                page_number=page,
                input_dir=GRAPH_FILEPATH,
                update=True  # we are loading previously updated graph
            )
            current_app.logger.info("Loaded existing graph")
            response["graph"] = graph_data
        else:
            print(f"Existing graph not found: {full_file_path}, graph will be generated in frontend")
            # Don't include graph in response - frontend will generate it
            # response["graph"] will be None/undefined
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
            MANUSCRIPTS_PATH, manuscript_name, "points-2D"
        )
        
        # Save the graph using existing save function
        save_graph_for_gnn(graph_data, manuscript_name, page, output_dir=GRAPH_FILEPATH)
        
        current_app.logger.info(f"Saving the updated graph for {manuscript_name}, page {page}")
        return {"success": True}, 200
        
    except Exception as e:
        print(f"Error saving graph: {str(e)}")
        return {"error": str(e)}, 500


# SAVE UPDATED GRAPH (after adding/deleting edges), SEGMENT LINES, and then RECOGNIZE text content
@bp.route("/semi-segment/<string:manuscript_name>/<string:page>", methods=["POST"])
def make_semi_segments(manuscript_name, page):
    try:
        MANUSCRIPTS_PATH = os.path.join(current_app.config['DATA_PATH'], 'manuscripts')
        POINTS_FILEPATH = os.path.join(
            MANUSCRIPTS_PATH, manuscript_name, "points-2D", f"{page}_labels.txt"
        )
        GRAPH_FILEPATH = os.path.join(
            MANUSCRIPTS_PATH, manuscript_name, "points-2D"
        )
        
        # Parse request data
        request_data = request.json

        # Extract graph data if available
        if 'graph' in request_data:
            graph_data = request_data['graph']
            
            # Save graph for GNN processing
            current_app.logger.info(f"Saving updated Graph for: {manuscript_name}/{page}.")
            save_graph_for_gnn(graph_data, manuscript_name, page, output_dir=GRAPH_FILEPATH, update=True)
            
            # Generate labels from connected components in the graph
            current_app.logger.info(f"Generating Labels from updated Graph for: {manuscript_name}/{page}.")
            labels = generate_labels_from_graph(graph_data)
            
            # Save the labels to the appropriate file
            with open(POINTS_FILEPATH, "w") as f:
                f.write("\n".join(map(str, labels)))
            
            # Also save the modifications log if present
            if 'modifications' in request_data:
                modifications_path = os.path.join(GRAPH_FILEPATH, f"{manuscript_name}_page{page}_modifications.json")
                with open(modifications_path, 'w') as f:
                    json.dump(request_data['modifications'], f, indent=2)
        

        # Run manual segmentation after saving labels
        segmentLinesFromPointClusters(manuscript_name, page)
        current_app.logger.info(f"Line Segmentation complete with updated graph for {manuscript_name}/{page}.")


        model_name_from_request = request_data.get("modelName")
        if not model_name_from_request: # handling error of old version of the app
            current_app.logger.error("Model name not provided in POST /semi-segment request.")
            recognized_line_data = ''
            # return Response(json.dumps({"error": "Model name not provided"}), status=400, mimetype='application/json')
        else:
            # NOW, PERFORM CHARACTER RECOGNITION FOR THIS PAGE
            current_app.logger.info(f"Starting text recognition from segmented line images {manuscript_name}/{page} with model {model_name_from_request}.")
            manuscript_folder_path = os.path.join(MANUSCRIPTS_PATH, manuscript_name)
            recognized_line_data = recognise_single_page_characters(
                manuscript_folder_path, model_name_from_request, manuscript_name, page
            )
            current_app.logger.info(f"Text recognition from segmented line images finished for {manuscript_name}/{page}.")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return Response(json.dumps({
            "message": f"Updated Graph, Updated Segmentation, Updated Recognition Done for : {manuscript_name} page {page}",
            "lines": recognized_line_data # Return the recognized lines for the current page
        }), status=200, mimetype='application/json')

    except Exception as e:
        current_app.logger.error(f"Error in POST /semi-segment: {str(e)}")
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')



# GET LINE IMAGES
@bp.route("/line-images/<manuscript_name>/<page>/<line>", methods=["GET"])
def serve_line_image(manuscript_name, page, line):
    current_app.logger.info(f"Getting line image ({line}) in  manuscript {manuscript_name},page {page}")
    # Build the folder and filename exactly how you want them
    base_dir   = current_app.config['DATA_PATH']
    folder     = os.path.join(base_dir, 'manuscripts', manuscript_name, 'lines', page)
    filename   = f"{line}.jpg" 

    # Resolve to an absolute path
    absolute_path = os.path.abspath(os.path.join(folder, filename))
    exists = os.path.exists(absolute_path)
    current_app.logger.info("Will serve file", extra={"absolute_path": absolute_path, "exists": exists})
    # If it’s not on disk, 404
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

# OPEN PREVIOUSLY UPLOADED MANUSCRIPTS
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


# FULLY AUTOMATIC AND RECOGNIZE TEXT CONTENTS
@bp.route("/upload-manuscript", methods=["POST"])
def annotate():
    MANUSCRIPTS_PATH = os.path.join(current_app.config['DATA_PATH'], 'manuscripts')
    uploaded_files = request.files
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
    # find_gpu_tensors()

    return lines, 200

