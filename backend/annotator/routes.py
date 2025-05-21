import os
import threading

from flask import Blueprint, request, send_from_directory, current_app, abort, send_file
from PIL import Image
import torch
import gc

from annotator.segmentation import segment_lines
from annotator.manual_segmentation import run_manual_segmentation
from annotator.recognition.recognition import recognise_characters
from annotator.finetune.finetune import finetune


#importing GNN libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from itertools import combinations
from scipy.spatial import cKDTree
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['grey', 'red'])
from sklearn.cluster import DBSCAN
import cv2
import networkx as nx
from collections import Counter
import random
import matplotlib
import matplotlib.colors as mcolors
import io
import base64



import numpy as np
import torch
from torch_geometric.data import Data
import json
import os


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


@bp.route("/segment/<string:manuscript_name>/<string:page>", methods=["POST"])
def make_segments(manuscript_name, page):
    MANUSCRIPTS_PATH = os.path.join(current_app.config['DATA_PATH'], 'manuscripts')
    segments = request.get_json()
    labels_file = os.path.join(
        MANUSCRIPTS_PATH, manuscript_name, "points-2D", f"{page}_labels.txt"
    )

    with open(labels_file, "w") as f:
        f.write("\n".join(map(str, segments)))

    run_manual_segmentation(manuscript_name, page)

    return {"message": f"succesfully saved labels for page {page}"}, 200






























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
        print("saving updated graph..")
        
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
        run_manual_segmentation(manuscript_name, page)
        
        return {"message": f"Graph and segmentation data saved for {manuscript_name} page {page}"}, 200

    except Exception as e:
        return {"error": str(e)}, 500

def generate_labels_from_graph(graph_data):
    """
    Generate labels for points based on connected components in the graph.
    Sort components from top to bottom and assign sequential labels.
    
    Args:
        graph_data (dict): Graph data containing nodes and edges
        
    Returns:
        list: Labels for each node/point
    """
    # Extract nodes and edges
    nodes = graph_data.get('nodes', [])
    edges = graph_data.get('edges', [])
    
    # Create an undirected graph using networkx
    import networkx as nx
    G = nx.Graph()
    
    # Add all nodes
    for i, node in enumerate(nodes):
        G.add_node(node['id'], x=node['x'], y=node['y'])
    
    # Add edges
    for edge in edges:
        G.add_edge(edge['source'], edge['target'])
    
    # Find connected components (each component is a line)
    components = list(nx.connected_components(G))
    
    # Calculate the average y-coordinate for each component
    component_y_avg = []
    for i, component in enumerate(components):
        y_coords = [nodes[n]['y'] for n in component if n < len(nodes)]
        avg_y = sum(y_coords) / len(y_coords) if y_coords else 0
        component_y_avg.append((i, avg_y, component))
    
    # Sort components by average y-coordinate (top to bottom)
    component_y_avg.sort(key=lambda x: x[1])
    
    # Create labels array (initialized with -1)
    labels = [-1] * len(nodes)
    
    # Assign labels to each node based on its component
    for label, (_, _, component) in enumerate(component_y_avg):
        for node_id in component:
            if node_id < len(labels):
                labels[node_id] = label
    
    return labels


@bp.route("/semi-segment/<manuscript_name>/<page>", methods=["GET"])
def get_points_and_graph(manuscript_name, page):
    MANUSCRIPTS_PATH = os.path.join(current_app.config['DATA_PATH'], 'manuscripts')
    try:
        print("Getting points and generating graph")
        # IMAGE_FILEPATH = os.path.join(
        #     MANUSCRIPTS_PATH, manuscript_name, "leaves", f"{page}.jpg"
        # )
        # image = plt.imread(IMAGE_FILEPATH)  # Replace with your image path
        # image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
        # Build potential file paths for jpg and tif files
        filepath_jpg = os.path.join(MANUSCRIPTS_PATH, manuscript_name, "leaves", f"{page}.jpg")
        filepath_tif = os.path.join(MANUSCRIPTS_PATH, manuscript_name, "leaves", f"{page}.tif")
        filepath_png = os.path.join(MANUSCRIPTS_PATH, manuscript_name, "leaves", f"{page}.png")

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
        
        # Apply the layout analysis logic to generate the graph
        graph_data = generate_layout_graph(points)
        save_graph_for_gnn(graph_data, manuscript_name, page, output_dir=GRAPH_FILEPATH)

        response["points"] = points
        response["graph"] = graph_data
        
        return response, 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}, 500

def generate_layout_graph(points):
    """
    Generate a graph representation of text layout based on points.
    This function implements the core layout analysis logic from the notebook.
    """
    NUM_NEIGHBOURS = 8
    cos_similarity_less_than = -0.8
    
    # Build a KD-tree for fast neighbor lookup
    tree = cKDTree(points)
    _, indices = tree.query(points, k=NUM_NEIGHBOURS)
    
    # Store graph edges and their properties
    edges = []
    edge_properties = []
    
    # Process nearest neighbors
    for current_point_index, nbr_indices in enumerate(indices):
        normalized_points = np.array(points)[nbr_indices] - np.array(points)[current_point_index]
        scaling_factor = np.max(np.abs(normalized_points))
        if scaling_factor == 0:
            scaling_factor = 1
        scaled_points = normalized_points / scaling_factor
        
        # Create a list of relative neighbors with their global indices
        relative_neighbours = [
            (global_idx, sp, np) 
            for global_idx, sp, np in zip(nbr_indices, scaled_points, normalized_points)
        ]
        
        filtered_neighbours = []
        for i, neighbor1 in enumerate(relative_neighbours):
            for neighbor2 in relative_neighbours[i+1:]:
                if np.linalg.norm(neighbor1[1]) * np.linalg.norm(neighbor2[1]) == 0:
                    cos_similarity = 0.0
                else:
                    cos_similarity = np.dot(neighbor1[1], neighbor2[1]) / (
                        np.linalg.norm(neighbor1[1]) * np.linalg.norm(neighbor2[1])
                    )
                
                # Calculate non-normalized distances
                norm1 = np.linalg.norm(neighbor1[2])
                norm2 = np.linalg.norm(neighbor2[2])
                total_length = norm1 + norm2
                
                # Select pairs with angles close to 180 degrees (opposite directions)
                if cos_similarity < cos_similarity_less_than:
                    filtered_neighbours.append((neighbor1, neighbor2, total_length, cos_similarity))
        
        if filtered_neighbours:
            # Find the shortest total length pair
            shortest_pair = min(filtered_neighbours, key=lambda x: x[2])
            
            connection_1, connection_2, total_length, cos_similarity = shortest_pair
            global_idx_connection_1 = connection_1[0]
            global_idx_connection_2 = connection_2[0]
            
            # Calculate angles with x-axis
            theta_a = np.degrees(np.arctan2(connection_1[2][1], connection_1[2][0]))
            theta_b = np.degrees(np.arctan2(connection_2[2][1], connection_2[2][0]))
            
            # Add edges to the graph
            edges.append([current_point_index, global_idx_connection_1])
            edges.append([current_point_index, global_idx_connection_2])
            
            # Calculate feature values for clustering
            y_diff1 = abs(connection_1[2][1])  # Vertical distance component
            y_diff2 = abs(connection_2[2][1])
            avg_y_diff = (y_diff1 + y_diff2) / 2
            
            x_diff1 = abs(connection_1[2][0])  # Horizontal distance component
            x_diff2 = abs(connection_2[2][0])
            avg_x_diff = (x_diff1 + x_diff2) / 2
            
            # Calculate aspect ratio (height/width)
            aspect_ratio = avg_y_diff / max(avg_x_diff, 0.001)  # Avoid division by zero
            
            # Calculate vertical alignment consistency
            vert_consistency = abs(y_diff1 - y_diff2)
            
            # Store edge properties for clustering
            edge_properties.append([
                total_length,
                np.abs(theta_a + theta_b),
                aspect_ratio,
                vert_consistency,
                avg_y_diff
            ])
    
    # USE THIS IF WE WANT TO KEEP THE OUTLIERS..FOR MAPS
    # Cluster the edges based on their properties
    # edge_labels = cluster_with_single_majority(np.array(edge_properties))
    
    # # Prepare the final graph structure
    # graph_data = {
    #     "nodes": [{"id": i, "x": float(point[0]), "y": float(point[1])} for i, point in enumerate(points)],
    #     "edges": []
    # }
    
    # # Add edges with their labels
    # for i, edge in enumerate(edges):
    #     # Determine edge label: 0 for correct (majority cluster), -1 for outliers
    #     edge_label = int(edge_labels[i // 2])  # Divide by 2 because we added each edge pair
        
    #     graph_data["edges"].append({
    #         "source": int(edge[0]),
    #         "target": int(edge[1]),
    #         "label": edge_label
    #     })    
    # return graph_data
    edge_labels = cluster_with_single_majority(np.array(edge_properties))

    # Create a mask for edges that are not outliers (label != -1)
    non_outlier_mask = np.array(edge_labels) != -1

    # Prepare the final graph structure
    graph_data = {
        "nodes": [{"id": i, "x": float(point[0]), "y": float(point[1])} for i, point in enumerate(points)],
        "edges": []
    }

    # Add edges with their labels, filtering out outliers
    for i, edge in enumerate(edges):
        # Determine the corresponding edge label using division by 2 (each edge appears twice)
        label_index = i // 2
        edge_label = int(edge_labels[label_index])
        
        # Only add the edge if it is not an outlier
        if non_outlier_mask[label_index]:
            graph_data["edges"].append({
                "source": int(edge[0]),
                "target": int(edge[1]),
                "label": edge_label
            })

    return graph_data


def cluster_with_single_majority(to_cluster, eps=10, min_samples=2):
    """
    Clusters data, identifying only one majority cluster and marking all other points as outliers.

    Args:
        to_cluster: List of data points to cluster.
        eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples: The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
        NumPy array of labels, where the majority cluster is labeled 0, and all other points are labeled -1.
    """
    to_cluster_array = np.array(to_cluster)

    if len(to_cluster_array) == 0:
      return np.array([])

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(to_cluster_array)

    # Count the occurrences of each label
    label_counts = Counter(labels)

    # Find the majority cluster label (excluding -1 outliers)
    majority_label = None
    max_count = 0
    for label, count in label_counts.items():
        if label != -1 and count > max_count:
            majority_label = label
            max_count = count

    # Create a new label array where the majority cluster is 0 and all others are -1
    new_labels = np.full(len(labels), -1)  # Initialize all as outliers

    if majority_label is not None:
        new_labels[labels == majority_label] = 0  # Assign 0 to the majority cluster

    return new_labels



def save_graph_for_gnn(graph_data, manuscript_name, page_number, output_dir='gnn_graphs',update=False):
    """
    Save a graph in a format compatible with Graph Neural Networks (PyTorch Geometric).
    
    Args:
        graph_data (dict): The graph data containing nodes and edges
        manuscript_name (str): Name of the manuscript
        page_number (int or str): Page number
        output_dir (str): Directory to save the graph data
    
    Returns:
        str: Path to the saved file
    """
    # Ensure output directory exists
    # os.makedirs(output_dir, exist_ok=True)
    
    # Extract node features (x and y coordinates)
    node_features = np.array([[node['x'], node['y']] for node in graph_data['nodes']], dtype=np.float32)
    
    # Extract edge indices in COO format
    edge_index = []
    edge_attr = []
    
    for edge in graph_data['edges']:
        source = edge['source']
        target = edge['target']
        label = edge['label']
        
        # Add edge in both directions for undirected graphs
        # (for directed graphs, remove the second append)
        edge_index.append([source, target])
        edge_attr.append([label])
    
    # Convert to numpy arrays
    edge_index = np.array(edge_index, dtype=np.int64).T  # Transpose to get 2 x num_edges
    edge_attr = np.array(edge_attr, dtype=np.float32)
    
    # Create PyTorch tensors
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # Create PyTorch Geometric Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(graph_data['nodes'])
    )
    
    # Add metadata
    data.manuscript = manuscript_name
    data.page = page_number
    
    # Save PyTorch Geometric data
    if not update:
        torch_path = os.path.join(output_dir, f"{manuscript_name}_page{page_number}_graph.pt")
    else:
        torch_path = os.path.join(output_dir, f"{manuscript_name}_page{page_number}_graph_updated.pt")
    torch.save(data, torch_path)
    
    # Also save as JSON for compatibility with other frameworks
    json_data = {
        "nodes": [{"id": i, "features": [float(f) for f in feat]} for i, feat in enumerate(node_features)],
        "edges": [{"source": int(edge_index[0, i]), 
                  "target": int(edge_index[1, i]), 
                  "features": [float(f) for f in edge_attr[i]]} 
                  for i in range(edge_index.shape[1])],
        "metadata": {
            "manuscript": manuscript_name,
            "page": page_number
        }
    }
    
    json_path = os.path.join(output_dir, f"{manuscript_name}_page{page_number}_graph.json")
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    return torch_path

# Example usage in a Flask/FastAPI endpoint:
# @app.route('/save-gnn-graph/<manuscript_name>/<page_number>', methods=['POST'])
# def save_gnn_graph_endpoint(manuscript_name, page_number):
#     graph_data = request.json
#     file_path = save_graph_for_gnn(graph_data, manuscript_name, page_number)
#     return jsonify({'success': True, 'file_path': file_path})