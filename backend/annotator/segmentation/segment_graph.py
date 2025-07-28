import os
import numpy as np
import torch
from torch_geometric.data import Data
import json
import cv2
from scipy.ndimage import maximum_filter
from scipy.ndimage import label
from annotator.segmentation.craft import CRAFT, copyStateDict, detect
from annotator.segmentation.utils import load_images_from_folder
from scipy.ndimage import maximum_filter, label
from skimage.draw import circle_perimeter
from pathlib import Path





# ------------------heatmap to point cloud---------

# def heatmap_to_pointcloud(heatmap, min_peak_value=0.3, min_distance=10):
#     """
#     Convert a 2D heatmap to a point cloud by identifying local maxima and generating
#     points with density proportional to the heatmap intensity.
    
#     Parameters:
#     -----------
#     heatmap : numpy.ndarray
#         2D array representing the heatmap
#     min_peak_value : float
#         Minimum value for a peak to be considered (normalized between 0 and 1)
#     min_distance : int
#         Minimum distance between peaks in pixels
        
#     Returns:
#     --------
#     points : numpy.ndarray
#         # TODO Each point represent a character. Now we want to get size (font size) along with the X,Y co-ordinates. To do this, caluclate a search window around each point dynamically based on the locations of the points.
#         Array of shape (N, 2) containing the generated points
#         #TODO add size of the blob as a third dimension. 

#     """
#     # Normalize heatmap to [0, 1]
#     heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
#     # Find local maxima
#     local_max = maximum_filter(heatmap_norm, size=min_distance)
#     peaks = (heatmap_norm == local_max) & (heatmap_norm > min_peak_value)
    
#     # Label connected components
#     labeled_peaks, num_peaks = label(peaks)
    
#     points = []
    
#     # For each peak, generate points
#     height = heatmap.shape[0]  # Get the height of the heatmap
#     for peak_idx in range(1, num_peaks + 1):
#         # Get peak location
#         peak_y, peak_x = np.where(labeled_peaks == peak_idx)[0][0], np.where(labeled_peaks == peak_idx)[1][0]
#         points.append([peak_x, peak_y])
#         #points.append([peak_x, height - 1 - peak_y])  # This line is modified

#     return np.array(points)


def heatmap_to_pointcloud(heatmap, min_peak_value=0.3, min_distance=5, max_growth_radius=50):
    """
    Convert a 2D heatmap to a point cloud (X, Y, Radius) by identifying local maxima
    and estimating a radius for each by growing a circle as long as the heatmap
    intensity along the circumference is decreasing.
    
    Parameters:
    -----------
    heatmap : numpy.ndarray
        2D array representing the heatmap.
    min_peak_value : float
        Minimum normalized value for a peak to be considered (normalized between 0 and 1).
        Peaks must have an intensity strictly greater than this value.
    min_distance : int
        Minimum distance between peaks in pixels. Used for `maximum_filter`.
    max_growth_radius : int, optional
        Maximum radius the circle is allowed to grow. If None, it defaults to
        half the minimum dimension of the heatmap.
        
    Returns:
    --------
    points_with_radius : numpy.ndarray
        Array of shape (N, 3) where N is the number of detected characters.
        Each row contains [Peak_X, Peak_Y, Estimated_Radius].
        Peak_X, Peak_Y are from the original peak detection.
        Estimated_Radius is the radius of the largest circle around the peak
        for which the average intensity on its circumference was still decreasing.
    """
    if heatmap.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max == h_min: # Handle flat heatmap
        return np.empty((0, 3), dtype=np.float64)
        
    # 1. Normalize heatmap to [0, 1]
    heatmap_norm = (heatmap - h_min) / (h_max - h_min)
    
    # 2. Find local maxima (Original logic)
    local_max_values = maximum_filter(heatmap_norm, size=min_distance)
    peaks_mask = (heatmap_norm == local_max_values) & (heatmap_norm > min_peak_value)
    
    # 3. Label connected components of these peak pixels
    labeled_individual_peaks, num_individual_peaks = label(peaks_mask)
    
    if num_individual_peaks == 0:
        return np.empty((0, 3), dtype=np.float64)

    points_and_radius = []
    
    H, W = heatmap_norm.shape
    if max_growth_radius is None:
        max_r_search = min(H, W) // 2
    else:
        max_r_search = max_growth_radius

    # 4. For each peak, grow a circle to estimate radius
    for peak_idx in range(1, num_individual_peaks + 1):
        peak_loc_y_arr, peak_loc_x_arr = np.where(labeled_individual_peaks == peak_idx)
        
        if peak_loc_y_arr.size == 0:
            continue
            
        peak_y, peak_x = peak_loc_y_arr[0], peak_loc_x_arr[0] # Use the first pixel of the peak area

        current_peak_intensity = heatmap_norm[peak_y, peak_x]
        last_ring_avg_intensity = current_peak_intensity
        estimated_radius = 0 # Radius 0 is the peak itself

        for r_test in range(1, max_r_search + 1):
            # Get coordinates of pixels on the circumference of radius r_test
            # skimage.draw.circle_perimeter ensures coordinates are within `shape` if provided.
            rr, cc = circle_perimeter(peak_y, peak_x, r_test, shape=heatmap_norm.shape)
            
            if rr.size == 0: # No pixels on this circumference (e.g., peak near edge, radius too large)
                break 

            current_ring_intensities = heatmap_norm[rr, cc]
            current_ring_avg_intensity = np.mean(current_ring_intensities)

            # Stop if slope is no longer strictly downward (i.e., current is flat or increasing)
            if current_ring_avg_intensity >= last_ring_avg_intensity:
                break 
            else:
                # Still decreasing, this radius is good. Update for next iteration.
                last_ring_avg_intensity = current_ring_avg_intensity
                estimated_radius = r_test # Update to this successful radius
        
        points_and_radius.append([float(peak_x), float(peak_y), float(estimated_radius)])

    return np.array(points_and_radius, dtype=np.float64)










# Assume these functions are defined elsewhere in your project
# from your_project.utils import load_images_from_folder, copyStateDict
# from your_project.detection import detect, CRAFT
# from your_project.pointcloud import heatmap_to_pointcloud

def images2points(folder_path):
    print(folder_path)
    m_name = os.path.basename(os.path.dirname(folder_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Model Loading ---
    _detector = CRAFT()
    _detector.load_state_dict(copyStateDict(torch.load("instance/models/segmentation/craft_mlt_25k.pth", map_location=device)))
    detector = torch.nn.DataParallel(_detector).to(device)
    detector.eval()

    # --- Data Loading ---
    inp_images, file_names = load_images_from_folder(folder_path)
    print("Current Working Directory:", os.getcwd())

    # --- Processing Loop ---
    out_images = []
    normalized_points_list = [] # List for normalized points
    unnormalized_points_list = [] # NEW: List for raw, unnormalized points
    page_dimensions = []
    
    for image, _filename in zip(inp_images, file_names):
        # 0. Store original page dimensions
        original_height, original_width, _ = image.shape
        page_dimensions.append((original_width, original_height))

        # 1. Get region score (heatmap)
        region_score, affinity_score = detect(image, detector, device)
        assert region_score.shape == affinity_score.shape
        
        # 2. Convert heatmap to raw point coordinates (unnormalized)
        raw_points = heatmap_to_pointcloud(region_score, min_peak_value=0.3, min_distance=10)
        
        # --- NEW: Store the unnormalized points first ---
        unnormalized_points_list.append(raw_points)

        # 3. Normalize the points
        height, width = region_score.shape
        longest_dim = max(height, width)
        
        if longest_dim > 0:
            normalized_points = raw_points / longest_dim
        else:
            normalized_points = raw_points

        # 4. Store the processed data
        normalized_points_list.append(normalized_points)
        out_images.append(np.copy(region_score))

    # --- Saving Results ---
    heatmap_dir = f'instance/manuscripts/{m_name}/heatmaps'
    base_data_dir = f'instance/manuscripts/{m_name}/base-dataset'
    frontend_graph_data_dir = f'instance/manuscripts/{m_name}/frontend-graph-data'

    os.makedirs(heatmap_dir, exist_ok=True)
    os.makedirs(base_data_dir, exist_ok=True)
    os.makedirs(frontend_graph_data_dir, exist_ok=True)

    # Save heatmaps
    for _img, _filename in zip(out_images, file_names):
        cv2.imwrite(os.path.join(heatmap_dir, _filename), 255 * _img)
    
    # --- Save NORMALIZED node features (for backward compatibility) ---
    for points, _filename in zip(normalized_points_list, file_names):
        output_filename = os.path.splitext(_filename)[0] + '_inputs_normalized.txt'
        output_path = os.path.join(base_data_dir, output_filename)
        np.savetxt(output_path, points, fmt='%f')

    # --- NEW: Save UNNORMALIZED node features to a separate file ---
    for raw_points, _filename in zip(unnormalized_points_list, file_names):
        raw_output_filename = os.path.splitext(_filename)[0] + '_inputs_unnormalized.txt'
        raw_output_path = os.path.join(base_data_dir, raw_output_filename)
        np.savetxt(raw_output_path, raw_points, fmt='%f')

    # Save the page dimensions
    for (width, height), _filename in zip(page_dimensions, file_names):
        dims_filename = os.path.splitext(_filename)[0] + '_dims.txt'
        dims_path = os.path.join(base_data_dir, dims_filename)
        with open(dims_path, 'w') as f:
            f.write(f"{width} {height}")


    # --- Cleanup ---
    del detector
    del _detector
    torch.cuda.empty_cache()

    print(f"Finished processing. All data saved to: {base_data_dir}")
    


def handle_save_graph(graph_data, manuscript_name, page_number, output_dir='gnn_graphs',update=False):
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
    node_features = np.array([[node['x'], node['y'], node['s']] for node in graph_data['nodes']], dtype=np.float32)
    
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
        torch_path = os.path.join(output_dir, f"{page_number}_graph.pt")
    else:
        torch_path = os.path.join(output_dir, f"{page_number}_graph_updated.pt")
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
    
    json_path = os.path.join(output_dir, f"{page_number}_graph.json")
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    return torch_path

def handle_load_graph(page_number,
                       input_dir='gnn_graphs',
                       update=False):
    """
    Load a previously saved PyTorch Geometric graph Data object.
    
    Args:
        manuscript_name (str): Name of the manuscript
        page_number (int or str): Page number
        input_dir (str): Directory where the graph files live
        update (bool): If True, look for the "_graph_updated.pt" version
    
    Returns:
        Data: The loaded PyG Data object
    
    Raises:
        FileNotFoundError: If the expected .pt file is not found
    """
    # Choose filename suffix based on update flag
    suffix = "_graph_updated.pt" if update else "_graph.pt"
    filename = f"{page_number}{suffix}"
    full_path = os.path.join(input_dir, filename)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"No graph file found at: {full_path}")
    
    # Load and return the Data object
    data = torch.load(full_path)
    return data_to_serializable_graph_dict(data)

def data_to_serializable_graph_dict(data):
    """
    Convert a PyTorch Geometric Data object into your JSON-serializable graph structure.

    Args:
        data (Data): The PyG Data object

    Returns:
        dict: JSON-serializable dictionary in desired format
    """
    # Nodes
    nodes = [
        {"id": i, "x": float(coord[0]), "y": float(coord[1]), "s": float(coord[2])}
        for i, coord in enumerate(data.x.tolist())
    ]

    # Edges
    edges = []
    edge_index = data.edge_index.tolist()
    edge_attr = data.edge_attr.tolist()

    for i in range(len(edge_attr)):
        source = int(edge_index[0][i])
        target = int(edge_index[1][i])
        label = int(edge_attr[i][0])  # Assuming edge_attr is shape [num_edges, 1]
        edges.append({"source": source, "target": target, "label": label})

    graph_data = {
        "nodes": nodes,
        "edges": edges,
        "num_nodes": data.num_nodes,
        "manuscript": getattr(data, 'manuscript', None),
        "page": getattr(data, 'page', None)
    }

    return graph_data

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





