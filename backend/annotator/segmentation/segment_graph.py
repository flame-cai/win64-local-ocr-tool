import os
import numpy as np
# from scipy.spatial import cKDTree
# from sklearn.cluster import DBSCAN
# from collections import Counter
import torch
from torch_geometric.data import Data
import json
import cv2
from scipy.ndimage import maximum_filter
from scipy.ndimage import label

from annotator.segmentation.craft import CRAFT, copyStateDict, detect
from annotator.segmentation.utils import load_images_from_folder


# ------------------heatmap to point cloud---------
# TODO get size (font size) of blob along with the X,Y co-ordinates
def heatmap_to_pointcloud(heatmap, min_peak_value=0.3, min_distance=10):
    """
    Convert a 2D heatmap to a point cloud by identifying local maxima and generating
    points with density proportional to the heatmap intensity.
    
    Parameters:
    -----------
    heatmap : numpy.ndarray
        2D array representing the heatmap
    min_peak_value : float
        Minimum value for a peak to be considered (normalized between 0 and 1)
    min_distance : int
        Minimum distance between peaks in pixels
        
    Returns:
    --------
    points : numpy.ndarray
        Array of shape (N, 2) containing the generated points
    """
    # Normalize heatmap to [0, 1]
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    # Find local maxima
    local_max = maximum_filter(heatmap_norm, size=min_distance)
    peaks = (heatmap_norm == local_max) & (heatmap_norm > min_peak_value)
    
    # Label connected components
    labeled_peaks, num_peaks = label(peaks)
    
    points = []
    
    # For each peak, generate points
    height = heatmap.shape[0]  # Get the height of the heatmap
    for peak_idx in range(1, num_peaks + 1):
        # Get peak location
        peak_y, peak_x = np.where(labeled_peaks == peak_idx)[0][0], np.where(labeled_peaks == peak_idx)[1][0]
        points.append([peak_x, peak_y])
        #points.append([peak_x, height - 1 - peak_y])  # This line is modified

    return np.array(points)



def images2points(folder_path):
    print(folder_path)
    #m_name = folder_path.split('/')[-2]
    m_name = os.path.basename(os.path.dirname(folder_path))
    device = torch.device('cuda') #change to cpu if no gpu


    # HEATMAP
    inp_images, file_names = load_images_from_folder(folder_path)
    print("Current Working Directory:", os.getcwd())

    _detector = CRAFT()
    _detector.load_state_dict(copyStateDict(torch.load("instance/models/segmentation/craft_mlt_25k.pth",map_location=device)))
    detector = torch.nn.DataParallel(_detector).to(device)
    detector.eval()


    out_images=[]
    points = []
    for image,_filename in zip(inp_images, file_names):
        # get region score and affinity score
        region_score, affinity_score = detect(image,detector, device)
        assert region_score.shape == affinity_score.shape
        points_twoD = heatmap_to_pointcloud(region_score, min_peak_value=0.3, min_distance=10)

        points.append(points_twoD)
        out_images.append(np.copy(region_score))


    if os.path.exists(f'instance/manuscripts/{m_name}/heatmaps') == False:
        os.makedirs(f'instance/manuscripts/{m_name}/heatmaps')

    if os.path.exists(f'instance/manuscripts/{m_name}/points-2D') == False:
        os.makedirs(f'instance/manuscripts/{m_name}/points-2D')

    for _img,_filename in zip(out_images,file_names):
        cv2.imwrite(f"instance/manuscripts/{m_name}/heatmaps/{_filename.replace('.tif','.jpg')}",255*_img)
        
    for points_twoD,_filename in zip(points,file_names):
        np.savetxt(f'instance/manuscripts/{m_name}/points-2D/{os.path.splitext(_filename)[0]}_points.txt', points_twoD, fmt='%d')

        # clear GPU memory
    del detector
    del _detector
    torch.cuda.empty_cache()
    


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

def load_graph_for_gnn(manuscript_name,
                       page_number,
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
    filename = f"{manuscript_name}_page{page_number}{suffix}"
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
        {"id": i, "x": float(coord[0]), "y": float(coord[1])}
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





