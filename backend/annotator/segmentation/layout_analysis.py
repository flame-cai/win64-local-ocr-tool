import os

#importing GNN libraries
import numpy as np
from scipy.spatial import cKDTree
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['grey', 'red'])
from sklearn.cluster import DBSCAN
from collections import Counter
import numpy as np
import torch
from torch_geometric.data import Data
import json
import os


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