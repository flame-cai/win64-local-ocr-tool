import os
import json
import logging
import argparse
from collections import defaultdict, Counter
from itertools import combinations
from sklearn.decomposition import PCA

import numpy as np
from scipy.spatial import KDTree
from sklearn.mixture import GaussianMixture # New import
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches # New import needed for custom legend

# --- Configuration ---
# K for K-Nearest Neighbors in inter-line spacing calculation
INTER_LINE_K = 10
# K for Heuristic Graph neighbor search
HEURISTIC_GRAPH_K = 10
# Cosine similarity threshold for opposite neighbors
OPPOSITE_NEIGHBOR_COS_SIM_THRESHOLD = -0.8
# Number of bins for font size analysis
NUM_FONT_BINS = 8

# --- Utility Functions ---

def setup_logging():
    """Configures the logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def describe_distribution(data: np.ndarray, name: str) -> dict:
    """Computes descriptive statistics for a 1D numpy array."""
    if data.size == 0:
        logging.warning(f"Distribution '{name}' is empty. Returning NaN statistics.")
        return {
            "count": 0, "mean": float('nan'), "std": float('nan'),
            "min": float('nan'), "25%": float('nan'), "50%": float('nan'),
            "75%": float('nan'), "max": float('nan')
        }
    
    stats = {
        "count": len(data),
        "mean": np.mean(data),
        "std": np.std(data),
        "min": np.min(data),
        "25%": np.percentile(data, 25),
        "50%": np.percentile(data, 50),
        "75%": np.percentile(data, 75),
        "max": np.max(data)
    }
    return stats

def load_page_data(page_id: int, base_path: str) -> (dict, np.ndarray, np.ndarray):
    """Loads all data files for a single page."""
    logging.info(f"Loading data for page {page_id}...")
    try:
        dims_path = os.path.join(base_path, f"{page_id}_dims.txt")
        inputs_path = os.path.join(base_path, f"{page_id}_inputs_unnormalized.txt")
        labels_path = os.path.join(base_path, f"{page_id}_labels_textline.txt")

        dims_arr = np.loadtxt(dims_path)
        dims = {"width": dims_arr[0], "height": dims_arr[1]}
        
        points = np.loadtxt(inputs_path) # [x, y, s]
        labels = np.loadtxt(labels_path, dtype=int)

        assert points.shape[0] == labels.shape[0], \
            f"Page {page_id}: Mismatch between number of points ({points.shape[0]}) and labels ({labels.shape[0]})"
        assert points.shape[1] == 3, \
            f"Page {page_id}: Points data should have 3 columns (x, y, s)"

        return dims, points, labels
    except FileNotFoundError as e:
        logging.error(f"File not found for page {page_id}: {e}")
        return None, None, None
    except Exception as e:
        logging.error(f"Error loading data for page {page_id}: {e}")
        return None, None, None


# --- Core Statistical Calculation Functions ---



def calculate_intra_line_stats(lines_data: dict) -> dict:
    """
    Calculates statistics that exist *within* each text line.
    - MODIFIED: Vertical Baseline Jitter now uses PCA for rotation invariance.
    - MODIFIED: Now returns RHS paired with its associated font size.
    """
    logging.info("Calculating intra-line statistics...")
    rhs_with_font_size = []
    all_jitter, all_font_size_cv, all_angles = [], [], []

    for line_label, line_data in lines_data.items():
        points = line_data['points']
        n_points = len(points)
        
        if n_points < 2:
            logging.warning(f"Line {line_label} has < 2 points. Skipping all calculations.")
            continue

        # 1. Intra-Line Font Size Variation
        font_sizes = points[:, 2]
        if np.mean(font_sizes) > 0:
            cv = np.std(font_sizes) / np.mean(font_sizes)
            all_font_size_cv.append(cv)

        # Build KD-Tree for efficient intra-line neighbor search
        line_kdtree = KDTree(points[:, :2])
        distances, indices = line_kdtree.query(points[:, :2], k=2)

        # 2. Relative Horizontal Spacing & 3. Local Writing Angle
        for i in range(n_points):
            neighbor_idx = indices[i, 1]
            p1 = points[i]
            p2 = points[neighbor_idx]
            dist = distances[i, 1]
            s_avg = (p1[2] + p2[2]) / 2.0
            
            if s_avg > 0:
                rhs_with_font_size.append((dist / s_avg, s_avg))

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            all_angles.append(np.arctan2(dy, dx))
            
        # 4. Vertical Baseline Jitter (using PCA for rotation invariance)
        xy_coords = points[:, :2]
        try:
            # Fit PCA to find the principal axis of the text line
            pca = PCA(n_components=2)
            pca.fit(xy_coords)
            
            # The second principal component is the direction of the jitter (perpendicular to the line)
            jitter_vector = pca.components_[1]
            
            # Center the data
            centered_coords = xy_coords - pca.mean_
            
            # Project the centered coordinates onto the jitter vector.
            # The result is the signed perpendicular distance from the baseline.
            perpendicular_distances = np.dot(centered_coords, jitter_vector)
            
            # Normalize jitter by font size
            # Use np.maximum to avoid division by zero
            jitter = perpendicular_distances / np.maximum(points[:, 2], 1e-6)
            all_jitter.extend(jitter.tolist())
            
        except Exception as e:
            # PCA can fail if all points are identical, which is an edge case.
            logging.warning(f"Could not perform PCA for line {line_label} (n_points={n_points}): {e}")

    return {
        "rhs_with_font_size": np.array(rhs_with_font_size),
        "vertical_baseline_jitter": np.array(all_jitter),
        "intra_line_font_size_cv": np.array(all_font_size_cv),
        "local_writing_angle_rad": np.array(all_angles)
    }


def calculate_inter_line_stats(all_points: np.ndarray, lines_data: dict, page_dims: dict) -> dict:
    # This function remains unchanged
    logging.info("Calculating inter-line statistics...")
    all_rvs = []
    
    if len(all_points) > INTER_LINE_K:
        all_labels = np.array([ld['label'] for ld in lines_data.values() for _ in range(len(ld['points']))])
        page_kdtree = KDTree(all_points[:, :2])
        distances, indices = page_kdtree.query(all_points[:, :2], k=INTER_LINE_K)
        
        for i in range(len(all_points)):
            current_label = all_labels[i]
            neighbor_indices = indices[i, 1:]
            neighbor_labels = all_labels[neighbor_indices]
            other_line_mask = neighbor_labels != current_label
            
            if np.any(other_line_mask):
                other_line_neighbors = all_points[neighbor_indices[other_line_mask]]
                vertical_distances = np.abs(all_points[i, 1] - other_line_neighbors[:, 1])
                min_vd = np.min(vertical_distances)
                if all_points[i, 2] > 0:
                    all_rvs.append(min_vd / all_points[i, 2])
    else:
        logging.warning("Not enough points on page to calculate inter-line spacing.")

    line_x_mins, line_x_maxs, line_centers = [], [], []
    page_width = page_dims['width']
    
    for line_label, line_data in lines_data.items():
        points = line_data['points']
        if len(points) > 0:
            x_coords = points[:, 0]
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            line_x_mins.append(x_min / page_width)
            line_x_maxs.append(x_max / page_width)
            line_centers.append(((x_min + x_max) / 2.0) / page_width)

    return {
        "relative_vertical_spacing": np.array(all_rvs),
        "line_alignment_left_normalized": np.array(line_x_mins),
        "line_alignment_right_normalized": np.array(line_x_maxs),
        "line_alignment_center_normalized": np.array(line_centers)
    }


def calculate_page_level_stats(all_points: np.ndarray, page_dims: dict) -> dict:
    # This function remains unchanged
    logging.info("Calculating page-level statistics...")
    width = page_dims.get('width', 0)
    height = page_dims.get('height', 0)
    aspect_ratio = width / height if height > 0 else float('nan')
    
    n_chars = len(all_points)
    if n_chars == 0:
        return {"aspect_ratio": aspect_ratio, "character_density": 0, "ink_density": 0}
        
    x_coords, y_coords, sizes = all_points[:, 0], all_points[:, 1], all_points[:, 2]
    text_block_width = np.max(x_coords) - np.min(x_coords)
    text_block_height = np.max(y_coords) - np.min(y_coords)
    text_block_area = text_block_width * text_block_height

    if text_block_area == 0:
        char_density, ink_density = float('inf'), float('inf')
    else:
        ink_area = np.sum(np.pi * (sizes / 2.0)**2)
        char_density = n_chars / text_block_area
        ink_density = ink_area / text_block_area

    return {
        "aspect_ratio": aspect_ratio,
        "character_density": char_density,
        "ink_density": ink_density
    }




def calculate_graph_based_stats(all_points: np.ndarray, page_dims: dict) -> dict:
    # This function remains unchanged
    logging.info("Calculating graph-based statistics...")
    n_points = len(all_points)
    if n_points < HEURISTIC_GRAPH_K:
        return {"heuristic_degree": np.array([]), "overlap": np.array([])}

    max_dim = max(page_dims['width'], page_dims['height'])
    max_s = np.max(all_points[:, 2])
    normalized_points = np.copy(all_points)
    normalized_points[:, :2] /= max_dim
    if max_s > 0: normalized_points[:, 2] /= max_s

    kdtree = KDTree(normalized_points[:, :2])
    heuristic_directed_edges = []
    for i in range(n_points):
        _, neighbor_indices = kdtree.query(normalized_points[i, :2], k=HEURISTIC_GRAPH_K)
        neighbor_indices = neighbor_indices[1:]
        best_pair, min_dist_sum = None, float('inf')
        for n1_idx, n2_idx in combinations(neighbor_indices, 2):
            vec1 = normalized_points[n1_idx, :2] - normalized_points[i, :2]
            vec2 = normalized_points[n2_idx, :2] - normalized_points[i, :2]
            norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0: continue
            cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
            if cosine_sim < OPPOSITE_NEIGHBOR_COS_SIM_THRESHOLD:
                dist_sum = norm1 + norm2
                if dist_sum < min_dist_sum:
                    min_dist_sum, best_pair = dist_sum, (n1_idx, n2_idx)
        if best_pair:
            heuristic_directed_edges.extend([(i, best_pair[0]), (i, best_pair[1])])
            
    degrees = np.zeros(n_points, dtype=int)
    for u, v in heuristic_directed_edges:
        degrees[u] += 1
        degrees[v] += 1
    
    edge_counts = Counter(heuristic_directed_edges)
    overlaps = []
    processed_edges = set()
    for u, v in heuristic_directed_edges:
        edge_key = tuple(sorted((u, v)))
        if edge_key not in processed_edges:
            overlap_val = edge_counts.get((u, v), 0) + edge_counts.get((v, u), 0)
            overlaps.append(overlap_val)
            processed_edges.add(edge_key)

    return {"heuristic_degree": degrees, "overlap": np.array(overlaps)}, heuristic_directed_edges


COLORS = [
    '#4363d8', '#f58231', '#ffe119', '#3cb44b', '#e6194B',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
    '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000',
    '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9'
]


def visualize_graph_stats(page_id: int, points: np.ndarray, graph_stats: dict, edges: list, page_dims: dict, output_dir: str):
    """
    Visualizes the graph structure for a single page and saves it as a PNG.
    - Node color represents heuristic degree (discrete).
    - Edge color represents overlap count (discrete).
    - Node size represents font size.
    - A clear, non-obstructing legend is generated.
    """
    logging.info(f"Visualizing graph for page {page_id} with new style...")
    n_points = len(points)
    if n_points == 0 or not edges:
        logging.warning(f"Skipping visualization for page {page_id} due to no points or edges.")
        return

    degrees = graph_stats.get('heuristic_degree', np.array([]))
    if degrees.size == 0:
        logging.warning(f"Skipping visualization for page {page_id} as no degrees were computed.")
        return

    fig, ax = plt.subplots(figsize=(12, 12 * (page_dims['height'] / page_dims['width'])))
    
    # --- 1. Calculate Overlaps and Map Colors ---
    edge_counts = Counter(edges)
    undirected_edges = defaultdict(lambda: {'points': [], 'overlap': 0})
    for u, v in edges:
        key = tuple(sorted((u, v)))
        undirected_edges[key]['points'] = [points[u, :2], points[v, :2]]
    for key in undirected_edges:
        u, v = key
        undirected_edges[key]['overlap'] = edge_counts.get((u, v), 0) + edge_counts.get((v, u), 0)

    # Create discrete color maps for overlaps and degrees
    unique_overlaps = sorted(np.unique([d['overlap'] for d in undirected_edges.values()]))
    unique_degrees = sorted(np.unique(degrees))
    
    # Use one iterator for the COLORS list to ensure colors are distinct
    color_iterator = 0
    overlap_color_map = {}
    for overlap in unique_overlaps:
        overlap_color_map[overlap] = COLORS[color_iterator % len(COLORS)]
        color_iterator += 1

    degree_color_map = {}
    for degree in unique_degrees:
        degree_color_map[degree] = COLORS[color_iterator % len(COLORS)]
        color_iterator += 1
        
    # --- 2. Plot Edges in Groups by Overlap Color ---
    edges_by_color = defaultdict(list)
    for edge in undirected_edges.values():
        color = overlap_color_map[edge['overlap']]
        edges_by_color[color].append(edge['points'])

    for color, line_segments in edges_by_color.items():
        lc = LineCollection(line_segments, colors=color, linewidths=1.2, zorder=1)
        ax.add_collection(lc)

    # --- 3. Plot Nodes in Groups by Degree Color ---
    for degree_val, color in degree_color_map.items():
        mask = (degrees == degree_val)
        ax.scatter(
            points[mask, 0], points[mask, 1],
            s=np.maximum(points[mask, 2] * 1.5, 5.0), # Scale font size, ensure min size
            c=color,
            zorder=2,
            edgecolors='k',
            linewidth=0.3
        )

    # --- 4. Create and Place a Custom Legend ---
    legend_handles = []
    # Add handles for node degrees
    if degree_color_map:
        legend_handles.append(mpatches.Patch(color='none', label='Node Degree:'))
        for degree, color in sorted(degree_color_map.items()):
            legend_handles.append(mpatches.Patch(color=color, label=f'{degree}'))
    
    # Add handles for edge overlaps
    if overlap_color_map:
        legend_handles.append(mpatches.Patch(color='none', label='')) # Spacer
        legend_handles.append(mpatches.Patch(color='none', label='Edge Overlap:'))
        for overlap, color in sorted(overlap_color_map.items()):
            legend_handles.append(mpatches.Patch(color=color, label=f'{overlap}'))

    # Place legend outside the plot area to prevent obstruction
    fig.legend(handles=legend_handles, 
               loc="upper left", 
               bbox_to_anchor=(1.01, 1.0),
               title="Legend",
               fontsize='small')

    # --- 5. Final Touches ---
    ax.set_title(f"Page {page_id} - Graph-Based Statistics (Discrete Colors)")
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    ax.set_xlim(0, page_dims['width'])
    ax.set_ylim(page_dims['height'], 0)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Save figure, ensuring the legend is not cut off
    output_path = os.path.join(output_dir, f"page_{page_id}_graph.png")
    # Use bbox_inches='tight' to make sure the external legend is saved
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved styled graph visualization to '{output_path}'")


def aggregate_statistics(all_page_stats: list) -> dict:
    """
    HEAVILY UPGRADED: Aggregates statistics and calculates derived ratios.
    1. Aggregates basic stats.
    2. Calculates derived global ratios (Line/Char spacing).
    3. Analyzes horizontal spacing bimodal distribution (Word/Char spacing).
    4. Analyzes horizontal spacing vs. font size (Binned stats).
    """
    logging.info("Aggregating statistics and calculating derived ratios...")
    if not all_page_stats:
        logging.warning("No page statistics to aggregate.")
        return {}

    # --- Step 1: Collate all raw data from all pages ---
    collated = defaultdict(list)
    for page_stat in all_page_stats:
        for category, stats in page_stat.items():
            if not isinstance(stats, dict): continue
            for stat_name, values in stats.items():
                if isinstance(values, np.ndarray) and values.size > 0:
                    collated[f"{category}.{stat_name}"].append(values)
                elif not isinstance(values, np.ndarray):
                    collated[f"{category}.{stat_name}"].append(np.array([values]))
    
    # --- Step 2: Compute basic summary stats for all distributions ---
    aggregated_summary = defaultdict(dict)
    for key, list_of_arrays in collated.items():
        category, stat_name = key.split('.', 1)
        full_distribution = np.concatenate(list_of_arrays)
        aggregated_summary[category][stat_name] = describe_distribution(full_distribution, key)
    
    # --- Step 3: Calculate derived global ratios ---
    logging.info("Calculating derived global ratios...")
    derived_ratios = {}
    mean_rhs = aggregated_summary['intra_line']['rhs_with_font_size']['mean']
    mean_rvs = aggregated_summary['inter_line']['relative_vertical_spacing']['mean']
    
    if mean_rhs > 0:
        derived_ratios['line_spacing_to_char_spacing_ratio'] = mean_rvs / mean_rhs
    else:
        derived_ratios['line_spacing_to_char_spacing_ratio'] = float('nan')
    
    # --- Step 4: Analyze bimodal distribution of horizontal spacing (Word vs. Char) ---
    logging.info("Analyzing word vs. character spacing...")
    full_rhs_data = np.concatenate(collated['intra_line.rhs_with_font_size'])
    # We only need the spacing values for this analysis
    full_rhs_dist = full_rhs_data[:, 0].reshape(-1, 1)

    try:
        if len(full_rhs_dist) > 20: # Need enough data for GMM
            gmm = GaussianMixture(n_components=2, random_state=42).fit(full_rhs_dist)
            
            # Identify which component is char and which is word based on mean
            means = gmm.means_.flatten()
            variances = gmm.covariances_.flatten()
            weights = gmm.weights_.flatten()
            
            char_idx, word_idx = (0, 1) if means[0] < means[1] else (1, 0)

            char_stats = {"mean": means[char_idx], "std": np.sqrt(variances[char_idx]), "weight": weights[char_idx]}
            word_stats = {"mean": means[word_idx], "std": np.sqrt(variances[word_idx]), "weight": weights[word_idx]}

            derived_ratios['word_spacing_analysis'] = {
                'char_spacing_stats': char_stats,
                'word_spacing_stats': word_stats,
                'word_to_char_spacing_ratio': word_stats['mean'] / char_stats['mean']
            }
        else:
            logging.warning("Not enough RHS data points to perform GMM analysis.")
            derived_ratios['word_spacing_analysis'] = "Not enough data"
    except Exception as e:
        logging.error(f"GMM for word/char spacing failed: {e}")
        derived_ratios['word_spacing_analysis'] = f"GMM failed: {e}"
        
    aggregated_summary['derived_ratios'] = derived_ratios
    
    # --- Step 5: Binned analysis of horizontal spacing vs. font size ---
    logging.info("Performing binned analysis of RHS vs. font size...")
    # Use full_rhs_data from Step 4: col 0 is RHS, col 1 is font size
    font_sizes = full_rhs_data[:, 1]
    rhs_values = full_rhs_data[:, 0]
    
    min_s, max_s = np.min(font_sizes), np.max(font_sizes)
    bin_edges = np.linspace(min_s, max_s, NUM_FONT_BINS + 1)
    
    binned_stats = []
    for i in range(NUM_FONT_BINS):
        bin_start, bin_end = bin_edges[i], bin_edges[i+1]
        
        # Find indices of points where font size is in the current bin
        mask = (font_sizes >= bin_start) & (font_sizes < bin_end)
        # For the last bin, include the max value
        if i == NUM_FONT_BINS - 1:
            mask = (font_sizes >= bin_start) & (font_sizes <= bin_end)
            
        rhs_in_bin = rhs_values[mask]
        
        bin_summary = {
            "font_size_bin_start": bin_start,
            "font_size_bin_end": bin_end,
            "font_size_bin_center": (bin_start + bin_end) / 2.0,
            "stats": describe_distribution(rhs_in_bin, f"RHS for font bin {i+1}")
        }
        binned_stats.append(bin_summary)
        
    aggregated_summary['intra_line']['binned_rhs_by_font_size'] = binned_stats

    # Final metadata
    aggregated_summary['metadata'] = {'num_pages_processed': len(all_page_stats)}
    
    return aggregated_summary


def main():
    """Main execution function."""
    setup_logging()
    parser = argparse.ArgumentParser(description="Calculate typographical and layout statistics from manuscript data.")
    
    # --- Argument Parsing ---
    parser.add_argument("dataset_path", type=str, help="Path to the base dataset directory.")
    parser.add_argument("--output_dir", type=str, default="stats", help="Main directory to save all output files and subdirectories.")
    parser.add_argument("--output_per_page", type=str, default="stats_per_page.json", help="Filename for the per-page statistics JSON.")
    parser.add_argument("--output_aggregated", type=str, default="stats_aggregated.json", help="Filename for the aggregated statistics JSON.")
    parser.add_argument("--visualize_graphs", action="store_true", help="Generate and save graph visualizations for each page.")
    parser.add_argument("--viz_dir", type=str, default="graph_visualizations", help="Subdirectory name for graph visualizations.")
    
    args = parser.parse_args()

    # --- Directory Setup ---
    # Create the main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"All outputs will be saved in the '{args.output_dir}' directory.")

    # Construct the full path for the visualization subdirectory and create it if needed
    viz_output_dir = os.path.join(args.output_dir, args.viz_dir)
    if args.visualize_graphs:
        os.makedirs(viz_output_dir, exist_ok=True)
        logging.info(f"Graph visualizations will be saved to '{viz_output_dir}'")
        
    # --- Data Loading ---
    try:
        files = os.listdir(args.dataset_path)
        page_ids = sorted(list(set([int(f.split('_')[0]) for f in files if f.split('_')[0].isdigit()])))
        logging.info(f"Found {len(page_ids)} pages in '{args.dataset_path}': {page_ids}")
    except Exception as e:
        logging.critical(f"Could not read dataset directory '{args.dataset_path}': {e}")
        return

    # --- Processing Loop ---
    all_page_raw_stats_for_json = []
    stats_for_aggregation = []
    
    for page_id in page_ids:
        dims, points, labels = load_page_data(page_id, args.dataset_path)
        if dims is None: continue
        
        # Prepare line data
        lines_data = defaultdict(lambda: {'points': [], 'indices': [], 'label': None})
        for i, (point, label) in enumerate(zip(points, labels)):
            lines_data[label]['points'].append(point)
            lines_data[label]['indices'].append(i)
            lines_data[label]['label'] = label
        
        for label in lines_data:
            lines_data[label]['points'] = np.array(lines_data[label]['points'])

        # Calculate all stats
        graph_stats, graph_edges = calculate_graph_based_stats(points, dims)
        page_stats_data = {
            'intra_line': calculate_intra_line_stats(lines_data),
            'inter_line': calculate_inter_line_stats(points, lines_data, dims),
            'page_level': calculate_page_level_stats(points, dims),
            'graph_based': graph_stats,
        }
        stats_for_aggregation.append(page_stats_data)
        
        # Visualize if requested
        if args.visualize_graphs:
            visualize_graph_stats(
                page_id=page_id,
                points=points,
                graph_stats=graph_stats,
                edges=graph_edges,
                page_dims=dims,
                output_dir=viz_output_dir  # Use the fully constructed path
            )
        
        page_stats_with_id = {'page_id': page_id, **page_stats_data}
        all_page_raw_stats_for_json.append(page_stats_with_id)
    
    # --- Save JSON Outputs ---
    # Construct full paths for JSON output files
    per_page_path = os.path.join(args.output_dir, args.output_per_page)
    aggregated_path = os.path.join(args.output_dir, args.output_aggregated)

    logging.info(f"Saving per-page statistics to '{per_page_path}'...")
    with open(per_page_path, 'w') as f:
        json.dump(all_page_raw_stats_for_json, f, cls=NumpyEncoder, indent=4)

    aggregated_stats = aggregate_statistics(stats_for_aggregation)
    
    logging.info(f"Saving aggregated statistics to '{aggregated_path}'...")
    with open(aggregated_path, 'w') as f:
        json.dump(aggregated_stats, f, cls=NumpyEncoder, indent=4)

    logging.info("Processing complete.")

if __name__ == '__main__':
    main()