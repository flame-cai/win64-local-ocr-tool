import os
import threading

from flask import Blueprint, request, send_from_directory, current_app, abort, send_file
from PIL import Image
import torch
import gc

from annotator.segmentation.segment_old import segment_lines
from annotator.segmentation.segment_from_point_clusters import segmentLinesFromPointClusters
from annotator.segmentation.segment_graph import save_graph_for_gnn, load_graph_for_gnn, generate_labels_from_graph, images2points
from annotator.recognition.recognition import recognise_characters,recognise_single_page_characters
from annotator.finetune.finetune import finetune


#importing GNN libraries

import numpy as np
import matplotlib.pyplot as plt
import cv2
import io
import base64
from flask import Response
import json



bp = Blueprint("main", __name__)


@bp.route("/")
def hello():
    return "Sanskrit Manuscript Annotation Tool"


@bp.route("/models")
def get_models():
    return os.listdir(os.path.join(current_app.config['DATA_PATH'], 'models', 'recognition'))


# @bp.route("/line-images/<string:manuscript_name>/<string:page>/<string:line>")
# def serve_line_image(manuscript_name, page, line):
#     MANUSCRIPTS_PATH = os.path.join(current_app.config['DATA_PATH'], 'manuscripts')
#     return send_from_directory(
#         os.path.join(MANUSCRIPTS_PATH, manuscript_name, "lines", page), line + ".jpg"
#     )

# @bp.route("/line-images/<string:manuscript_name>/<string:page>/<string:line>")
# def serve_line_image(manuscript_name, page, line):
#     MANUSCRIPTS_PATH = os.path.join(current_app.config['DATA_PATH'], 'manuscripts')
#     folder_path = os.path.join(MANUSCRIPTS_PATH, manuscript_name, "lines", page)
#     file_name = line + ".jpg"  # or .jpg depending on what you have

#     print("Looking for image at:", os.path.join(folder_path, file_name))

#     if not os.path.exists(os.path.join(folder_path, file_name)):
#         print("Not found!")
#         abort(404)

#     return send_from_directory(folder_path, file_name)

@bp.route("/line-images/<manuscript_name>/<page>/<line>")
def serve_line_image(manuscript_name, page, line):
    # Build the folder and filename exactly how you want them
    base_dir   = current_app.config['DATA_PATH']
    folder     = os.path.join(base_dir, 'manuscripts', manuscript_name, 'lines', page)
    filename   = f"{line}.jpg"   # or '.png' if that's what you have

    # Resolve to an absolute path
    absolute_path = os.path.abspath(os.path.join(folder, filename))
    print("Will serve:", absolute_path, "Exists?", os.path.exists(absolute_path))

    # If it’s not on disk, 404
    if not os.path.exists(absolute_path):
        abort(404)

    # Send it directly
    return send_file(absolute_path, mimetype='image/jpeg')





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


def finetune_context(data, app_context):
    # app_context.push()
    with app_context:
        finetune(data)


@bp.route("/fine-tune", methods=["POST"])
def do_finetune():
    MANUSCRIPTS_PATH = os.path.join(current_app.config['DATA_PATH'], 'manuscripts')
    thread = threading.Thread(
        target=finetune_context, args=(request.json, current_app.app_context())
    )
    thread.start()
    return "Success", 200


@bp.route("/uploaded-manuscripts", methods=["GET"])
def get_manuscripts():
    MANUSCRIPTS_PATH = os.path.join(current_app.config['DATA_PATH'], 'manuscripts')
    return os.listdir(MANUSCRIPTS_PATH)


@bp.route("/recognise", methods=["POST"])
def recognise_manuscript():
    MANUSCRIPTS_PATH = os.path.join(current_app.config['DATA_PATH'], 'manuscripts')
    manuscript_name = request.json.get("manuscript_name")
    model = request.json.get("model")
    print(manuscript_name)
    print(model)
    folder_path = os.path.join(MANUSCRIPTS_PATH, manuscript_name)
    print(folder_path)
    lines = recognise_characters(folder_path, model, manuscript_name)
    print(lines)
    return lines, 200


@bp.route("/segment/<string:manuscript_name>/<string:page>", methods=["GET"])
def get_points(manuscript_name, page):
    MANUSCRIPTS_PATH = os.path.join(current_app.config['DATA_PATH'], 'manuscripts')
    try:
        IMAGE_FILEPATH = os.path.join(
            MANUSCRIPTS_PATH, manuscript_name, "leaves", f"{page}.jpg"
        )
        image = Image.open(IMAGE_FILEPATH)
        width, height = image.size
        response = {"dimensions": [width, height]}
        POINTS_FILEPATH = os.path.join(
            MANUSCRIPTS_PATH, manuscript_name, "points-2D", f"{page}_points.txt"
        )
        if not os.path.exists(POINTS_FILEPATH):
            return {"error": "Page not found"}, 404
        with open(POINTS_FILEPATH, "r") as f:
            points = [row.split() for row in f.readlines()]
        response["points"] = points
        return response, 200

    except Exception as e:
        return {"error": str(e)}, 500



# FULLY MANUAL LABELING

@bp.route("/segment/<string:manuscript_name>/<string:page>", methods=["POST"])
def make_segments(manuscript_name, page):
    MANUSCRIPTS_PATH = os.path.join(current_app.config['DATA_PATH'], 'manuscripts')
    segments = request.get_json()
    labels_file = os.path.join(
        MANUSCRIPTS_PATH, manuscript_name, "points-2D", f"{page}_labels.txt"
    )

    with open(labels_file, "w") as f:
        f.write("\n".join(map(str, segments)))

    segmentLinesFromPointClusters(manuscript_name, page)

    return {"message": f"succesfully saved labels for page {page}"}, 200














# GRAPH BASED LABELLING

@bp.route("/new-process-manuscript", methods=["POST"])
def new_process_manuscript():
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
    print("NEW PROCESS MANUSCRIPT")
    # TODO SHIFT THIS    
    # lines = recognise_characters(folder_path, model, manuscript_name)
    torch.cuda.empty_cache()
    gc.collect()
    # find_gpu_tensors()

    return Response(json.dumps({"message": "Files uploaded and points processing initiated."}), status=200, mimetype='application/json')





@bp.route("/semi-segment/<string:manuscript_name>/<string:page>", methods=["POST"])
def make_semi_segments(manuscript_name, page):
    print("REWIRED THE GRAPH")
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
        model_name_from_request = request_data.get("modelName")
        if not model_name_from_request:
            current_app.logger.error("Model name not provided in POST /semi-segment request.")
            return Response(json.dumps({"error": "Model name not provided"}), status=400, mimetype='application/json')

        # Extract graph data if available
        if 'graph' in request_data:
            graph_data = request_data['graph']
            
            # Save graph for GNN processing
            save_graph_for_gnn(graph_data, manuscript_name, page, output_dir=GRAPH_FILEPATH, update=True)
            
            # Generate labels from connected components in the graph
            labels = generate_labels_from_graph(graph_data)
            
            # Save the labels to the appropriate file
            with open(POINTS_FILEPATH, "w") as f:
                f.write("\n".join(map(str, labels)))
            
            # Also save the modifications log if present
            if 'modifications' in request_data:
                modifications_path = os.path.join(GRAPH_FILEPATH, f"{manuscript_name}_page{page}_modifications.json")
                with open(modifications_path, 'w') as f:
                    json.dump(request_data['modifications'], f, indent=2)
        
        # Process point segments if available
        if isinstance(request_data, list) or 'points' in request_data:
            segments_data = request_data if isinstance(request_data, list) else request_data['points']
            segments_path = os.path.join(GRAPH_FILEPATH, f"{manuscript_name}_page{page}_segments.json")
            with open(segments_path, 'w') as f:
                json.dump(segments_data, f, indent=2)

        # Run manual segmentation after saving labels
        segmentLinesFromPointClusters(manuscript_name, page)
        current_app.logger.info(f"Segmentation complete for {manuscript_name}/{page}.")

        # NOW, PERFORM CHARACTER RECOGNITION FOR THIS PAGE
        current_app.logger.info(f"Starting character recognition for {manuscript_name}/{page} with model {model_name_from_request}.")
        manuscript_folder_path = os.path.join(MANUSCRIPTS_PATH, manuscript_name)
        
        recognized_line_data = recognise_single_page_characters(
            manuscript_folder_path, model_name_from_request, manuscript_name, page
        )
        current_app.logger.info(f"Character recognition finished for {manuscript_name}/{page}.")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return Response(json.dumps({
            "message": f"Segmentation saved and page recognized for {manuscript_name} page {page}",
            "lines": recognized_line_data # Return the recognized lines for the current page
        }), status=200, mimetype='application/json')

    except Exception as e:
        current_app.logger.error(f"Error in POST /semi-segment: {str(e)}")
        # import traceback
        # traceback.print_exc() # For more detailed server logs during dev
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')



@bp.route("/semi-segment/<manuscript_name>/<page>", methods=["GET"])
def get_points_and_graph(manuscript_name, page):
    MANUSCRIPTS_PATH = os.path.join(current_app.config['DATA_PATH'], 'manuscripts')
    try:
        print("Getting points and checking for existing graph")
        filepath_jpg = os.path.join(MANUSCRIPTS_PATH, manuscript_name, "leaves", f"{page}.jpg")
        filepath_tif = os.path.join(MANUSCRIPTS_PATH, manuscript_name, "leaves", f"{page}.tif")
        filepath_png = os.path.join(MANUSCRIPTS_PATH, manuscript_name, "leaves", f"{page}.png")
        print("yooooooooooo")
        print(MANUSCRIPTS_PATH)
        print(filepath_jpg)
        # Check which file exists
        if os.path.exists(filepath_jpg):
            IMAGE_FILEPATH = filepath_jpg
        elif os.path.exists(filepath_tif):
            IMAGE_FILEPATH = filepath_tif
        elif os.path.exists(filepath_png):
            IMAGE_FILEPATH = filepath_png
        else:
            raise FileNotFoundError("Neither .jpg nor .tif image file found for the given page.")

        # Read the image using the appropriate method based on the file extension
        if IMAGE_FILEPATH.lower().endswith('.tif'):
            # Use Pillow to open TIFF images and convert to a NumPy array
            image = np.array(Image.open(IMAGE_FILEPATH))
        else:
            image = plt.imread(IMAGE_FILEPATH)
        
        image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
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
        
        
        POINTS_FILEPATH = os.path.join(
            MANUSCRIPTS_PATH, manuscript_name, "points-2D", f"{page}_points.txt"
        )
        GRAPH_FILEPATH = os.path.join(
            MANUSCRIPTS_PATH, manuscript_name, "points-2D"
        )

        if not os.path.exists(POINTS_FILEPATH):
            return {"error": "Page not found"}, 404
            
        # Load points from file
        with open(POINTS_FILEPATH, "r") as f:
            points_raw = [row.strip().split() for row in f.readlines()]
            
        # Convert to numeric values
        points = [[float(coord) for coord in point] for point in points_raw]

        # Always include points in response
        response["points"] = points

        # Build the expected file name (if old graph exists)
        graph_file_name = f"{manuscript_name}_page{page}_graph_updated.pt"
        full_file_path = os.path.join(GRAPH_FILEPATH, graph_file_name)
        
        # Check if the file exists and load it
        if os.path.exists(full_file_path):
            print(f"File exists: {full_file_path}")
            graph_data = load_graph_for_gnn(
                manuscript_name=manuscript_name,
                page_number=page,
                input_dir=GRAPH_FILEPATH,
                update=True  # we are loading previously updated graph
            )
            print("Loaded existing graph")
            response["graph"] = graph_data
        else:
            print(f"File not found: {full_file_path}, graph will be generated in frontend")
            # Don't include graph in response - frontend will generate it
            # response["graph"] will be None/undefined
        
        return response, 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}, 500









# Optional: Add endpoint to save graphs generated in frontend
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
        
        print(f"Graph saved for {manuscript_name}, page {page}")
        return {"success": True}, 200
        
    except Exception as e:
        print(f"Error saving graph: {str(e)}")
        return {"error": str(e)}, 500


# Example usage in a Flask/FastAPI endpoint:
# @app.route('/save-gnn-graph/<manuscript_name>/<page_number>', methods=['POST'])
# def save_gnn_graph_endpoint(manuscript_name, page_number):
#     graph_data = request.json
#     file_path = save_graph_for_gnn(graph_data, manuscript_name, page_number)
#     return jsonify({'success': True, 'file_path': file_path})