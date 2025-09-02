    You are an expert software developer in machine learning, vue js frontend development, and flask backend development.
    You job is to add a section to the already existing sections called 'edit manuscript', which will allow us to edit the layout (add/delete edges of previously processed manuscripts/maps)

    THe existing three options on the main page are:

        <RouterLink :to="{ name: 'new-manuscript' }" class="btn btn-primary m-2">
        New Manuscript / Map
        </RouterLink>

        <RouterLink :to="{ name: 'upload-manuscript' }" class="btn btn-sm btn-secondary text-gray-700 bg-gray-200 m-2">
        old version -- Annotate
        </RouterLink>

        <RouterLink :to="{ name: 'uploaded-manuscripts' }" class="btn btn-sm btn-secondary text-gray-700 bg-gray-200 m-2">
        old version -- Uploaded Manuscripts
        </RouterLink>

    and we ant to add another section called 'edit manuscript'. Our primary goal is to allow the user to edit (add/delete edges) and update a previously processed manuscript. 

    Make sure the 'edit manuscript section will be able to update the previously saved data of each manuscript, which contains:
    - gnn-dataset
        - _dims.txt (these won't need updating)
        - _inputs_normalized.txt (these won't need updating)
        _ _inputs_unnormalized.txt (these won't need updating)
        - _labels_textline.txt (this will need updating)
    - frontend-graph-data (this will need updating)
    - heatmaps (these won't need updating)
    - leaves (these won't need updating)
    - lines (these won't need updating)

    We essentially want to update the graph data (_labels_textline.txt and frontend-graph-data). 
    The GUI frontend of this new view will be very similar to the what it is when we add/delete edges of a new manuscript (including the hotkeys)

    And if the layout graph is updated, the lines segmentation will also change - so pay attention to this too.

    Do not make unnecessary changes
    Please write robust code, which will handle edge case, have lots of logging and debugging. Have assert statements where required. Pay special attention to update base-data and frontend-graph-data (they are the same data in different format)
    Please write entire files which need to be changed, or new files which need to be added. 
    Please make sure none of the adjacent and downstream code breaks with this new addition.


    ================================================
    FILE: README.md
    ================================================


    Digitizing text from historical manuscripts yields historians multiple benefits. The digitization process consists of three steps: text-line-image segmentation, text recognition from the text-line-images (and post-correction). 

    This tool enables segmenting text-line-images from pages with diverse layouts. It represents text-lines as graphs, with characters as the nodes, and with edges connecting each character of a text-line to it's previous and next neighbour. In other words, we use nodes and edges as units of comparison and data collection instead of dense pixel-level metrics. This enables easier layout annotation, and improved performance compared to existing methods (as tested on a set of 15 pages with layouts of varying complexity, ranging from simple single-column and double-column layouts to layouts with pictures, footnotes, tables, interlinear writing, marginalia, text bleeding, staining, coloring, and irregular font sizes)

    To recognise text content from the segmented text-line-images, we use a pre-trained text recognition model for the Devanāgarī script. The tools enables fine-tuning of the pre-trained model on specific manuscripts, which results in the model's predictions getting progressively better with more annotated data, thus also making the subsequent annotation easier - similar to active learning.

    Contact kartik.niszoig at gmail for questions, comments and reporting bugs..


    ![Demo](demo.gif)

    **Step 1**: Automatically Segment Text Line Images from Document, with the ability to manually ADD or DELETE edges for tricky edge-case page layouts. **Step 2**: Recognize the text content from the Text Line Images, make corrections, and fine tune the IMG2TEXT model

    ## News    

    - [2025/05/30] Code Released!

    ## Environment Setup
    The code is tested on Windows 11 (x64) machine with NVIDIA GeForce RTX 4050 Laptop GPU with CUDA 12.8 Driver. 

    ```
    # Download/Clone this repository
    git clone https://github.com/flame-cai/win64-local-ocr-tool.git

    # go to folder win64-local-ocr-tool
    cd win64-local-ocr-tool
    ```

    The application uses two AI models: [CRAFT](https://github.com/clovaai/CRAFT-pytorch) and [EasyOCR's](https://github.com/JaidedAI) Devanagari pretrained model. CRAFT detects the locations of the characters in a page, which is used to crop out text-line-images from pages with diverse layouts. The Devanagari pretrained model is then used to detect the text-content from the cropped text-line-images, and can also be fine-tuned for a specific manuscript. 

    - Download craft_mlt_25k.pth from [here](https://huggingface.co/amitesh863/craft/resolve/main/craft_mlt_25k.pth?download=true). Put this file in the `backend/instance/models/segmentation/` folder. 

    - Download devanagari.pth from [here](https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/devanagari.zip). Make sure to unzip the devanagari.zip file to get devanagari.pth file. Put this file in the `backend/instance/models/recognition/` folder. 


    ### Setup the backend
    Please follow the following steps to create the backend conda environment:
    ```
    # open terminal (or miniconda prompt) and go to the backend folder
    cd backend

    # create the conda environment
    conda env create -f environment.yml

    # activate the conda environment named 'ocr-tool'
    conda activate ocr-tool

    # run the backend
    flask run --debug

    # OR run the backend using:  
    python app.py
    ```

    ### Setup the frontend
    Install [Node.js](https://nodejs.org/en) if not installed.
    ```
    # open a new terminal, and go to frontend folder
    cd frontend

    # Install the node packages using
    npm install

    # Run the development server using 
    npm run dev
    ```

    



    ================================================
    FILE: calculate_stats.py
    ================================================
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

    def load_page_data(page_id: str, base_path: str) -> (dict, np.ndarray, np.ndarray):
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


    def visualize_graph_stats(page_id: str, points: np.ndarray, graph_stats: dict, edges: list, page_dims: dict, output_dir: str):
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
                s=np.maximum(points[mask, 2]/2, 2.0), # Scale font size, ensure min size
                c=color,
                zorder=2,
                edgecolors='k',
                alpha=0.5,
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

        # --- Directory Setup (TASK 2) ---
        # Get the basename of the dataset path (e.g., 'synthetic-dataset' from 'data/synthetic-dataset')
        dataset_name = os.path.basename(os.path.normpath(args.dataset_path))
        # The final output directory will be <output_dir>/<dataset_name>
        final_output_dir = os.path.join(args.output_dir, dataset_name)
        
        os.makedirs(final_output_dir, exist_ok=True)
        logging.info(f"All outputs will be saved in the '{final_output_dir}' directory.")

        # Construct the full path for the visualization subdirectory and create it if needed
        viz_output_dir = os.path.join(final_output_dir, args.viz_dir)
        if args.visualize_graphs:
            os.makedirs(viz_output_dir, exist_ok=True)
            logging.info(f"Graph visualizations will be saved to '{viz_output_dir}'")
            
        # --- Data Loading (TASK 1) ---
        try:
            files = os.listdir(args.dataset_path)
            # Identify page IDs by a common suffix, e.g., '_dims.txt'.
            # This supports string-based IDs like 'A_12'. Using .removesuffix for clarity (Python 3.9+).
            dims_suffix = "_dims.txt"
            page_ids = sorted([f.removesuffix(dims_suffix) for f in files if f.endswith(dims_suffix)])
            
            if not page_ids:
                logging.warning(f"Could not find any pages in '{args.dataset_path}'. "
                                f"Expected to find files ending with '{dims_suffix}'.")
            else:
                logging.info(f"Found {len(page_ids)} pages in '{args.dataset_path}'.")
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
        
        # --- Save JSON Outputs (TASK 2) ---
        # Construct full paths for JSON output files inside the new final_output_dir
        per_page_path = os.path.join(final_output_dir, args.output_per_page)
        aggregated_path = os.path.join(final_output_dir, args.output_aggregated)

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


    ================================================
    FILE: backend/app.py
    ================================================
    from annotator import create_app

    if __name__ == '__main__':
        app = create_app()
        app.run(debug=True)



    ================================================
    FILE: backend/environment.yml
    ================================================
    name: ocr-tool

    channels:
    - conda-forge  
    - pytorch
    - pyg
    - nvidia       
    - defaults    

    dependencies:
    - python=3.11
    - numpy
    - pandas
    - scipy
    - scikit-learn
    - scikit-image
    - matplotlib
    - pytorch==2.4.1
    - torchvision
    - pytorch-cuda=12.1 
    - pyg
    - flask
    - flask-cors
    - flask-sqlalchemy
    - werkzeug
    - pillow
    - opencv
    - python-dotenv
    - packaging
    - six
    - natsort
    - pyyaml
    - pip:
        - lmdb
        - nltk
        - python-json-logger
        - regex




    ================================================
    FILE: backend/annotator/__init__.py
    ================================================
    from flask import Flask
    from flask_cors import CORS

    from annotator.config import Config
    from annotator.models import db


    def create_app():
        app = Flask(__name__)

        app.config.from_object(Config())

        db.init_app(app)
        with app.app_context():
            db.create_all()

        CORS(app)

        from annotator import routes

        app.register_blueprint(routes.bp)

        return app



    ================================================
    FILE: backend/annotator/config.py
    ================================================
    import os
    from dotenv import load_dotenv

    load_dotenv('.env')

    class Config(object):
        SQLALCHEMY_DATABASE_URI = "sqlite:///project.db"
        DATA_PATH = os.environ.get('DATA_PATH', default='instance')


    ================================================
    FILE: backend/annotator/models.py
    ================================================
    from datetime import datetime

    from flask_sqlalchemy import SQLAlchemy
    from sqlalchemy.orm import DeclarativeBase
    from sqlalchemy.orm import Mapped, mapped_column

    class Base(DeclarativeBase):
        pass

    db = SQLAlchemy(model_class=Base)

    class RecognitionLog(db.Model):
        id: Mapped[int] = mapped_column(primary_key=True)
        predicted_label: Mapped[str]
        confidence_score: Mapped[float]
        manuscript_name: Mapped[str]
        page: Mapped[str]
        line: Mapped[str]
        image_path: Mapped[str]
        timestamp: Mapped[datetime]

    class UserAnnotationLog(db.Model):
        id: Mapped[int] = mapped_column(primary_key=True)
        manuscript_name: Mapped[str]
        page: Mapped[str]
        line: Mapped[str]
        ground_truth: Mapped[str]
        levenshtein_distance: Mapped[int]
        image_path: Mapped[str]
        timestamp: Mapped[datetime]


    ================================================
    FILE: backend/annotator/routes.py
    ================================================
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
                
                # Downscale the image using a high-quality resampling filter
                # Note: In newer versions of Pillow (9.0.0+), Image.LANCZOS is aliased
                # to Image.Resampling.LANCZOS. Using the latter is preferred.
                try:
                    # For Pillow 9.0.0 and newer
                    from PIL import Image as PILImage
                    resampling_filter = PILImage.Resampling.LANCZOS
                except AttributeError:
                    # For older versions of Pillow
                    resampling_filter = Image.LANCZOS

                image = image.resize((new_width, new_height), resampling_filter)
            # --- MODIFICATION END ---

            # Convert to RGB if needed (JPEG doesn't support some modes like RGBA)
            if image.mode in ("RGBA", "P", "LA"):
                image = image.convert("RGB")

            # Build new filename with .jpg extension
            new_filename = f"{base_filename}.jpg"

            # Save image as JPEG in leaves_folder_path
            image.save(os.path.join(leaves_folder_path, new_filename), "JPEG")

            print(f"Saved: {new_filename}")

        # It's assumed that images2points is defined elsewhere in your project
        images2points(os.path.join(folder_path, "leaves")) 
        torch.cuda.empty_cache()
        gc.collect()

        return Response(json.dumps({"message": "Files uploaded and points processing initiated."}), status=200, mimetype='application/json')



    # AUTO GENERATE GRAPH or load previously UPDATED GRAPH
    @bp.route("/semi-segment/<manuscript_name>/<page>", methods=["GET"])
    def get_node_features_and_graph(manuscript_name, page):
        current_app.logger.info("Getting Manuscript Page, Points and previously updated graph (if available)")
        MANUSCRIPTS_PATH = os.path.join(current_app.config['DATA_PATH'], 'manuscripts')
        IMAGE_FILEPATH= os.path.join(MANUSCRIPTS_PATH, manuscript_name, "leaves", f"{page}.jpg")
        POINTS_FILEPATH = os.path.join(
            MANUSCRIPTS_PATH, manuscript_name, "gnn-dataset", f"{page}_inputs_unnormalized.txt"
        )
        GRAPH_FILEPATH = os.path.join(
            MANUSCRIPTS_PATH, manuscript_name, "frontend-graph-data"
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
            points = [[float(coord) for coord in point] for point in points_raw] ##TODO ADD FEATURES
            # Always include points in response
            response["points"] = points

            # If graph already exist before, load it, else create a new graph in frontend
            graph_file_name = f"{page}_graph_updated.pt"
            full_file_path = os.path.join(GRAPH_FILEPATH, graph_file_name)
            # Check if the file exists and load it
            if os.path.exists(full_file_path):
                graph_data = handle_load_graph(
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
                MANUSCRIPTS_PATH, manuscript_name, "frontend-graph-data"
            )
            
            # Save the graph using existing save function
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
            POINTS_FILEPATH = os.path.join(
                MANUSCRIPTS_PATH, manuscript_name, "gnn-dataset", f"{page}_labels_textline.txt"
            )
            GRAPH_FILEPATH = os.path.join(
                MANUSCRIPTS_PATH, manuscript_name, "frontend-graph-data"
            )
            
            # Parse request data
            request_data = request.json

            # Extract graph data if available
            if 'graph' in request_data:
                graph_data = request_data['graph']
                
                # Save graph for GNN processing
                current_app.logger.info(f"Saving updated Graph for: {manuscript_name}/{page}.")
                handle_save_graph(graph_data, manuscript_name, page, output_dir=GRAPH_FILEPATH, update=True)
                
                # Generate labels from connected components in the graph
                current_app.logger.info(f"Generating Labels from updated Graph for: {manuscript_name}/{page}.")
                labels = generate_labels_from_graph(graph_data)
                
                # Save the labels to the appropriate file
                with open(POINTS_FILEPATH, "w") as f:
                    f.write("\n".join(map(str, labels)))
                
                # Also save the modifications log if present
                # if 'modifications' in request_data:
                #     modifications_path = os.path.join(GRAPH_FILEPATH, f"{page}_modifications.json")
                #     with open(modifications_path, 'w') as f:
                #         json.dump(request_data['modifications'], f, indent=2)
            

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


    # FULLY AUTOMATIC AND RECOGNIZE TEXT CONTENTS (OLD METHOD)
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




    ================================================
    FILE: backend/annotator/finetune/dataset.py
    ================================================
    import os
    import sys
    import re
    import six
    import math
    import torch
    import pandas  as pd

    from natsort import natsorted
    from PIL import Image
    import numpy as np
    from torch.utils.data import Dataset, ConcatDataset, Subset
    # from torch._utils import _accumulate
    from itertools import accumulate as _accumulate
    import torchvision.transforms as transforms

    def contrast_grey(img):
        high = np.percentile(img, 90)
        low  = np.percentile(img, 10)
        return (high-low)/(high+low), high, low

    def adjust_contrast_grey(img, target = 0.4):
        contrast, high, low = contrast_grey(img)
        if contrast < target:
            img = img.astype(int)
            ratio = 200./(high-low)
            img = (img - low + 25)*ratio
            img = np.maximum(np.full(img.shape, 0) ,np.minimum(np.full(img.shape, 255), img)).astype(np.uint8)
        return img


    class Batch_Balanced_Dataset(object):

        def __init__(self, opt):
            """
            Modulate the data ratio in the batch.
            For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
            the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.
            """
            log = open(f'./saved_models/{opt.model_name}/log_dataset.txt', 'a')
            dashed_line = '-' * 80
            print(dashed_line)
            log.write(dashed_line + '\n')
            print(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
            log.write(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}\n')
            assert len(opt.select_data) == len(opt.batch_ratio)

            _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, contrast_adjust = opt.contrast_adjust)
            self.data_loader_list = []
            self.dataloader_iter_list = []
            batch_size_list = []
            Total_batch_size = 0
            for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
                _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
                print(dashed_line)
                log.write(dashed_line + '\n')
                _dataset, _dataset_log = hierarchical_dataset(root=opt.train_data, opt=opt, select_data=[selected_d])
                total_number_dataset = len(_dataset)
                log.write(_dataset_log)

                """
                The total number of data can be modified with opt.total_data_usage_ratio.
                ex) opt.total_data_usage_ratio = 1 indicates 100% usage, and 0.2 indicates 20% usage.
                See 4.2 section in our paper.
                """
                number_dataset = int(total_number_dataset * float(opt.total_data_usage_ratio))
                dataset_split = [number_dataset, total_number_dataset - number_dataset]
                indices = range(total_number_dataset)
                _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                            for offset, length in zip(_accumulate(dataset_split), dataset_split)]
                selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
                selected_d_log += f'num samples of {selected_d} per batch: {opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}'
                print(selected_d_log)
                log.write(selected_d_log + '\n')
                batch_size_list.append(str(_batch_size))
                Total_batch_size += _batch_size

                _data_loader = torch.utils.data.DataLoader(
                    _dataset, batch_size=_batch_size,
                    prefetch_factor=None,
                    shuffle=True,
                    num_workers=int(opt.workers), #prefetch_factor=2,persistent_workers=True,
                    collate_fn=_AlignCollate, pin_memory=True)
                self.data_loader_list.append(_data_loader)
                self.dataloader_iter_list.append(iter(_data_loader))

            Total_batch_size_log = f'{dashed_line}\n'
            batch_size_sum = '+'.join(batch_size_list)
            Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
            Total_batch_size_log += f'{dashed_line}'
            opt.batch_size = Total_batch_size

            print(Total_batch_size_log)
            log.write(Total_batch_size_log + '\n')
            log.close()

        def get_batch(self):
            balanced_batch_images = []
            balanced_batch_texts = []

            for i, data_loader_iter in enumerate(self.dataloader_iter_list):
                try:
                    print(i,data_loader_iter)
                    image, text = next(data_loader_iter)
                    balanced_batch_images.append(image)
                    balanced_batch_texts += text
                except StopIteration:
                    self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                    image, text = next(self.dataloader_iter_list[i])
                    balanced_batch_images.append(image)
                    balanced_batch_texts += text
                except ValueError:
                    pass

            balanced_batch_images = torch.cat(balanced_batch_images, 0)

            return balanced_batch_images, balanced_batch_texts


    def hierarchical_dataset(root, opt, select_data='/'):
        """ select_data='/' contains all sub-directory of root directory """
        dataset_list = []
        dataset_log = f'dataset_root:    {root}\t dataset: {select_data[0]}'
        print(dataset_log)
        dataset_log += '\n'
        for dirpath, dirnames, filenames in os.walk(root+'/'):
            if not dirnames:
                select_flag = False
                for selected_d in select_data:
                    if selected_d in dirpath:
                        select_flag = True
                        break

                if select_flag:
                    dataset = OCRDataset(dirpath, opt)
                    sub_dataset_log = f'sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}'
                    print(sub_dataset_log)
                    dataset_log += f'{sub_dataset_log}\n'
                    dataset_list.append(dataset)

        concatenated_dataset = ConcatDataset(dataset_list)

        return concatenated_dataset, dataset_log

    class OCRDataset(Dataset):

        def __init__(self, root, opt):

            self.root = root
            self.opt = opt
            print(root)
            self.df = pd.read_csv(os.path.join(root,'labels.csv'), sep='^([^,]+),', engine='python', usecols=['filename', 'words'], keep_default_na=False)
            self.nSamples = len(self.df)

            if self.opt.data_filtering_off:
                self.filtered_index_list = [index for index in range(self.nSamples)]
            else:
                self.filtered_index_list = []
                for index in range(self.nSamples):
                    label = self.df.at[index,'words']
                    try:
                        if len(label) > self.opt.batch_max_length:
                            continue
                    except:
                        print(label)
                    out_of_char = f'[^{self.opt.character}]'
                    if re.search(out_of_char, label.lower()):
                        continue
                    self.filtered_index_list.append(index)
                self.nSamples = len(self.filtered_index_list)

        def __len__(self):
            return self.nSamples

        def __getitem__(self, index):
            index = self.filtered_index_list[index]
            print(self.df)
            img_fname = self.df.at[index,'filename']
            img_fpath = os.path.join(self.root, img_fname)
            label = self.df.at[index,'words']

            if self.opt.rgb:
                img = Image.open(img_fpath).convert('RGB')  # for color image
            else:
                img = Image.open(img_fpath).convert('L')

            if not self.opt.sensitive:
                label = label.lower()

            # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
            out_of_char = f'[^{self.opt.character}]'
            label = re.sub(out_of_char, '', label)

            return (img, label)

    class ResizeNormalize(object):

        def __init__(self, size, interpolation=Image.BICUBIC):
            self.size = size
            self.interpolation = interpolation
            self.toTensor = transforms.ToTensor()

        def __call__(self, img):
            img = img.resize(self.size, self.interpolation)
            img = self.toTensor(img)
            img.sub_(0.5).div_(0.5)
            return img


    class NormalizePAD(object):

        def __init__(self, max_size, PAD_type='right'):
            self.toTensor = transforms.ToTensor()
            self.max_size = max_size
            self.max_width_half = math.floor(max_size[2] / 2)
            self.PAD_type = PAD_type

        def __call__(self, img):
            img = self.toTensor(img)
            img.sub_(0.5).div_(0.5)
            c, h, w = img.size()
            Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
            Pad_img[:, :, :w] = img  # right pad
            if self.max_size[2] != w:  # add border Pad
                Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

            return Pad_img


    class AlignCollate(object):

        def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False, contrast_adjust = 0.):
            self.imgH = imgH
            self.imgW = imgW
            self.keep_ratio_with_pad = keep_ratio_with_pad
            self.contrast_adjust = contrast_adjust

        def __call__(self, batch):
            batch = filter(lambda x: x is not None, batch)
            images, labels = zip(*batch)

            if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
                resized_max_w = self.imgW
                input_channel = 3 if images[0].mode == 'RGB' else 1
                transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

                resized_images = []
                for image in images:
                    w, h = image.size

                    #### augmentation here - change contrast
                    if self.contrast_adjust > 0:
                        image = np.array(image.convert("L"))
                        image = adjust_contrast_grey(image, target = self.contrast_adjust)
                        image = Image.fromarray(image, 'L')

                    ratio = w / float(h)
                    if math.ceil(self.imgH * ratio) > self.imgW:
                        resized_w = self.imgW
                    else:
                        resized_w = math.ceil(self.imgH * ratio)

                    resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                    resized_images.append(transform(resized_image))
                    # resized_image.save('./image_test/%d_test.jpg' % w)

                image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

            else:
                transform = ResizeNormalize((self.imgW, self.imgH))
                image_tensors = [transform(image) for image in images]
                image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

            return image_tensors, labels


    def tensor2im(image_tensor, imtype=np.uint8):
        image_numpy = image_tensor.cpu().float().numpy()
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        return image_numpy.astype(imtype)


    def save_image(image_numpy, image_path):
        image_pil = Image.fromarray(image_numpy)
        image_pil.save(image_path)



    ================================================
    FILE: backend/annotator/finetune/finetune.py
    ================================================
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

    import os
    import csv
    import shutil
    import random
    import yaml
    import pandas as pd

    from datetime import datetime
    from flask import current_app

    from annotator.finetune.utils import AttrDict
    from annotator.finetune.train import train
    from annotator.models import db, UserAnnotationLog


    def get_config(file_path, manuscript_name, selected_model, model_name):
        with open(file_path, "r", encoding="utf-8") as stream:
            opt = yaml.safe_load(stream)
        opt = AttrDict(opt)
        opt.character = opt.number + opt.symbol + opt.lang_char
        opt.manuscript_name = manuscript_name
        opt.saved_model = os.path.join(
            os.path.join(current_app.config['DATA_PATH']), "models", "recognition", selected_model,
        )
        opt.model_name = model_name
        # This creates place for logfile. Need to keep this in mind when transitioning logs from .txt to database
        os.makedirs(f"./saved_models/{opt.model_name}", exist_ok=True)
        return opt


    def finetune(data):
        MANUSCRIPTS_PATH = os.path.join(current_app.config['DATA_PATH'], 'manuscripts')

        manuscript_name = data[0]["manuscript_name"]
        annotations = data[0]["annotations"]
        selected_model = data[0]["selected_model"]
        model_name = data[0].get("model_name", f"{manuscript_name}.pth")

        opt = get_config(
            os.path.join("annotator", "finetune", "config_files", "config.yml"),
            manuscript_name,
            selected_model,
            model_name,
        )

        TEMP_FOLDER = "temp"
        TRAIN_FOLDER = os.path.join(TEMP_FOLDER, "train")
        VAL_FOLDER = os.path.join(TEMP_FOLDER, "val")
        TRAIN_CSV_FILE = os.path.join(TRAIN_FOLDER, "labels.csv")
        VAL_CSV_FILE = os.path.join(VAL_FOLDER, "labels.csv")

        # Ensure the temp folder exists
        os.makedirs(TRAIN_FOLDER, exist_ok=True)
        os.makedirs(VAL_FOLDER, exist_ok=True)

        # Initialize the CSV files with headers if they don't exist
        for csv_file in [TRAIN_CSV_FILE, VAL_CSV_FILE]:
            if not os.path.exists(csv_file):
                with open(csv_file, mode="w", encoding="utf-8", newline="") as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(["filename", "words"])

        for page in annotations:
            for line in annotations[page]:
                ground_truth = annotations[page][line]["ground_truth"]
                image_path = os.path.join(
                    MANUSCRIPTS_PATH, manuscript_name, "lines", page, line + ".jpg"
                )
                filename = os.path.basename(image_path)

                # Create log entry
                log_entry = UserAnnotationLog(
                    manuscript_name=manuscript_name,
                    page=page,
                    line=line,
                    ground_truth=ground_truth,
                    levenshtein_distance=annotations[page][line]["levenshtein_distance"],
                    image_path=image_path,
                    timestamp=datetime.now(),
                )
                db.session.add(log_entry)

                # Randomly assign to train or val (80% train, 20% val)
                if random.random() < 0.8:
                    target_folder = TRAIN_FOLDER
                    target_csv = TRAIN_CSV_FILE
                else:
                    target_folder = VAL_FOLDER
                    target_csv = VAL_CSV_FILE

                # Copy the image to the appropriate folder
                try:
                    shutil.copy(image_path, target_folder)
                except FileNotFoundError:
                    print(f"Image not found: {image_path}")

                # Append to the appropriate CSV file
                with open(target_csv, mode="a", encoding="utf-8", newline="") as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow([filename, ground_truth])
        db.session.commit()

        train(opt, manuscript_name, amp=False)

        shutil.rmtree("temp")



    ================================================
    FILE: backend/annotator/finetune/model.py
    ================================================
    import torch.nn as nn
    from annotator.finetune.modules.transformation import TPS_SpatialTransformerNetwork
    from annotator.finetune.modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
    from annotator.finetune.modules.sequence_modeling import BidirectionalLSTM
    from annotator.finetune.modules.prediction import Attention

    class Model(nn.Module):

        def __init__(self, opt):
            super(Model, self).__init__()
            self.opt = opt
            self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                        'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

            """ Transformation """
            if opt.Transformation == 'TPS':
                self.Transformation = TPS_SpatialTransformerNetwork(
                    F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
            else:
                print('No Transformation module specified')

            """ FeatureExtraction """
            if opt.FeatureExtraction == 'VGG':
                self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
            elif opt.FeatureExtraction == 'RCNN':
                self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
            elif opt.FeatureExtraction == 'ResNet':
                self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
            else:
                raise Exception('No FeatureExtraction module specified')
            self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
            self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

            """ Sequence modeling"""
            if opt.SequenceModeling == 'BiLSTM':
                self.SequenceModeling = nn.Sequential(
                    BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                    BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
                self.SequenceModeling_output = opt.hidden_size
            else:
                print('No SequenceModeling module specified')
                self.SequenceModeling_output = self.FeatureExtraction_output

            """ Prediction """
            if opt.Prediction == 'CTC':
                self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
            elif opt.Prediction == 'Attn':
                self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)
            else:
                raise Exception('Prediction is neither CTC or Attn')

        def forward(self, input, text, is_train=True):
            """ Transformation stage """
            if not self.stages['Trans'] == "None":
                input = self.Transformation(input)

            """ Feature extraction stage """
            visual_feature = self.FeatureExtraction(input)
            visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
            visual_feature = visual_feature.squeeze(3)

            """ Sequence modeling stage """
            if self.stages['Seq'] == 'BiLSTM':
                contextual_feature = self.SequenceModeling(visual_feature)
            else:
                contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

            """ Prediction stage """
            if self.stages['Pred'] == 'CTC':
                prediction = self.Prediction(contextual_feature.contiguous())
            else:
                prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)

            return prediction



    ================================================
    FILE: backend/annotator/finetune/test.py
    ================================================
    import os
    import time
    import string
    import argparse

    import torch
    import torch.backends.cudnn as cudnn
    import torch.utils.data
    import torch.nn.functional as F
    import numpy as np
    from nltk.metrics.distance import edit_distance

    from annotator.finetune.utils import CTCLabelConverter, AttnLabelConverter, Averager
    from annotator.finetune.dataset import hierarchical_dataset, AlignCollate
    from annotator.finetune.model import Model

    def validation(model, criterion, evaluation_loader, converter, opt, device):
        """ validation or evaluation """
        n_correct = 0
        norm_ED = 0
        length_of_data = 0
        infer_time = 0
        valid_loss_avg = Averager()

        for i, (image_tensors, labels) in enumerate(evaluation_loader):
            batch_size = image_tensors.size(0)
            length_of_data = length_of_data + batch_size
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)
            
            start_time = time.time()
            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)
                forward_time = time.time() - start_time

                # Calculate evaluation loss for CTC decoder.
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                # permute 'preds' to use CTCloss format
                cost = criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)

                if opt.decode == 'greedy':
                    # Select max probabilty (greedy decoding) then decode index to character
                    _, preds_index = preds.max(2)
                    preds_index = preds_index.view(-1)
                    preds_str = converter.decode_greedy(preds_index.data, preds_size.data)
                elif opt.decode == 'beamsearch':
                    preds_str = converter.decode_beamsearch(preds, beamWidth=2)

            else:
                preds = model(image, text_for_pred, is_train=False)
                forward_time = time.time() - start_time

                preds = preds[:, :text_for_loss.shape[1] - 1, :]
                target = text_for_loss[:, 1:]  # without [GO] Symbol
                cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)
                labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

            infer_time += forward_time
            valid_loss_avg.add(cost)

            # calculate accuracy & confidence score
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            confidence_score_list = []
            
            for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    gt = gt[:gt.find('[s]')]
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                if pred == gt:
                    n_correct += 1

                '''
                (old version) ICDAR2017 DOST Normalized Edit Distance https://rrc.cvc.uab.es/?ch=7&com=tasks
                "For each word we calculate the normalized edit distance to the length of the ground truth transcription." 
                if len(gt) == 0:
                    norm_ED += 1
                else:
                    norm_ED += edit_distance(pred, gt) / len(gt)
                '''
                
                # ICDAR2019 Normalized Edit Distance 
                if len(gt) == 0 or len(pred) ==0:
                    norm_ED += 0
                elif len(gt) > len(pred):
                    norm_ED += 1 - edit_distance(pred, gt) / len(gt)
                else:
                    norm_ED += 1 - edit_distance(pred, gt) / len(pred)

                # calculate confidence score (= multiply of pred_max_prob)
                try:
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                except:
                    confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
                confidence_score_list.append(confidence_score)
                # print(pred, gt, pred==gt, confidence_score)

        accuracy = n_correct / float(length_of_data) * 100
        norm_ED = norm_ED / float(length_of_data) # ICDAR2019 Normalized Edit Distance

        return valid_loss_avg.val(), accuracy, norm_ED, preds_str, confidence_score_list, labels, infer_time, length_of_data



    ================================================
    FILE: backend/annotator/finetune/train.py
    ================================================
    import os
    import sys
    import time
    import random
    import torch
    import torch.backends.cudnn as cudnn
    import torch.nn as nn
    import torch.nn.init as init
    import torch.optim as optim
    import torch.utils.data
    from torch.cuda.amp import autocast, GradScaler
    import numpy as np

    from annotator.finetune.utils import CTCLabelConverter, AttnLabelConverter, Averager
    from annotator.finetune.dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
    from annotator.finetune.model import Model
    from annotator.finetune.test import validation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def count_parameters(model):
        print("Modules, Parameters")
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            #table.add_row([name, param])
            total_params+=param
            print(name, param)
        print(f"Total Trainable Params: {total_params}")
        return total_params

    def train(opt, manuscript_name, show_number = 2, amp=False ):
        """ dataset preparation """
        if not opt.data_filtering_off:
            print('Filtering the images containing characters which are not in opt.character')
            print('Filtering the images whose label is longer than opt.batch_max_length')

        opt.select_data = opt.select_data.split('-')
        opt.batch_ratio = opt.batch_ratio.split('-')
        train_dataset = Batch_Balanced_Dataset(opt)

        log = open(f'./saved_models/{opt.model_name}/log_dataset.txt', 'a', encoding="utf8")
        AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, contrast_adjust=opt.contrast_adjust)
        valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=min(32, opt.batch_size),
            shuffle=True,  # 'True' to check training progress with validation function.
            num_workers=int(opt.workers), prefetch_factor=None,
            collate_fn=AlignCollate_valid, pin_memory=True)
        log.write(valid_dataset_log)
        print('-' * 80)
        log.write('-' * 80 + '\n')
        log.close()
        
        """ model configuration """
        if 'CTC' in opt.Prediction:
            converter = CTCLabelConverter(opt.character)
        else:
            converter = AttnLabelConverter(opt.character)
        opt.num_class = len(converter.character)

        if opt.rgb:
            opt.input_channel = 3
        model = Model(opt)
        print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
            opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
            opt.SequenceModeling, opt.Prediction)

        if opt.saved_model != '':
            pretrained_dict = torch.load(opt.saved_model)
            if opt.new_prediction:
                model.Prediction = nn.Linear(model.SequenceModeling_output, len(pretrained_dict['module.Prediction.weight']))  
            
            model = torch.nn.DataParallel(model).to(device) 
            print(f'loading pretrained model from {opt.saved_model}')
            if opt.FT:
                model.load_state_dict(pretrained_dict, strict=False)
            else:
                model.load_state_dict(pretrained_dict)
            if opt.new_prediction:
                model.module.Prediction = nn.Linear(model.module.SequenceModeling_output, opt.num_class)  
                for name, param in model.module.Prediction.named_parameters():
                    if 'bias' in name:
                        init.constant_(param, 0.0)
                    elif 'weight' in name:
                        init.kaiming_normal_(param)
                model = model.to(device) 
        else:
            # weight initialization
            for name, param in model.named_parameters():
                if 'localization_fc2' in name:
                    print(f'Skip {name} as it is already initialized')
                    continue
                try:
                    if 'bias' in name:
                        init.constant_(param, 0.0)
                    elif 'weight' in name:
                        init.kaiming_normal_(param)
                except Exception as e:  # for batchnorm.
                    if 'weight' in name:
                        param.data.fill_(1)
                    continue
            model = torch.nn.DataParallel(model).to(device)
        
        model.train() 
        print("Model:")
        print(model)
        count_parameters(model)
        
        """ setup loss """
        if 'CTC' in opt.Prediction:
            criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
        else:
            criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
        # loss averager
        loss_avg = Averager()

        # freeze some layers
        try:
            if opt.freeze_FeatureFxtraction:
                for param in model.module.FeatureExtraction.parameters():
                    param.requires_grad = False
            if opt.freeze_SequenceModeling:
                for param in model.module.SequenceModeling.parameters():
                    param.requires_grad = False
        except:
            pass
        
        # filter that only require gradient decent
        filtered_parameters = []
        params_num = []
        for p in filter(lambda p: p.requires_grad, model.parameters()):
            filtered_parameters.append(p)
            params_num.append(np.prod(p.size()))
        print('Trainable params num : ', sum(params_num))
        # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

        # setup optimizer
        if opt.optim=='adam':
            #optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
            optimizer = optim.Adam(filtered_parameters)
        else:
            optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
        print("Optimizer:")
        print(optimizer)

        """ final options """
        # print(opt)
        with open(f'./saved_models/{opt.model_name}/opt.txt', 'a', encoding="utf8") as opt_file:
            opt_log = '------------ Options -------------\n'
            args = vars(opt)
            for k, v in args.items():
                opt_log += f'{str(k)}: {str(v)}\n'
            opt_log += '---------------------------------------\n'
            print(opt_log)
            opt_file.write(opt_log)

        """ start training """
        start_iter = 0
        if opt.saved_model != '':
            try:
                start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
                print(f'continue to train, start_iter: {start_iter}')
            except:
                pass

        start_time = time.time()
        best_accuracy = -1
        best_norm_ED = -1
        i = start_iter

        scaler = GradScaler()
        t1= time.time()
            
        while(True):
            # train part
            optimizer.zero_grad(set_to_none=True)
            
            if amp:
                with autocast():
                    image_tensors, labels = train_dataset.get_batch()
                    image = image_tensors.to(device)
                    text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
                    batch_size = image.size(0)

                    if 'CTC' in opt.Prediction:
                        preds = model(image, text).log_softmax(2)
                        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                        preds = preds.permute(1, 0, 2)
                        torch.backends.cudnn.enabled = False
                        cost = criterion(preds, text.to(device), preds_size.to(device), length.to(device))
                        torch.backends.cudnn.enabled = True
                    else:
                        preds = model(image, text[:, :-1])  # align with Attention.forward
                        target = text[:, 1:]  # without [GO] Symbol
                        cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
                scaler.scale(cost).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                image_tensors, labels = train_dataset.get_batch()
                image = image_tensors.to(device)
                text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
                batch_size = image.size(0)
                if 'CTC' in opt.Prediction:
                    preds = model(image, text).log_softmax(2)
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    preds = preds.permute(1, 0, 2)
                    torch.backends.cudnn.enabled = False
                    cost = criterion(preds, text.to(device), preds_size.to(device), length.to(device))
                    torch.backends.cudnn.enabled = True
                else:
                    preds = model(image, text[:, :-1])  # align with Attention.forward
                    target = text[:, 1:]  # without [GO] Symbol
                    cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
                cost.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip) 
                optimizer.step()
            loss_avg.add(cost)

            # validation part
            if (i % opt.valInterval == 0) and (i!=0):
                print('training time: ', time.time()-t1)
                t1=time.time()
                elapsed_time = time.time() - start_time
                # for log
                with open(f'./saved_models/{opt.model_name}/log_train.txt', 'a', encoding="utf8") as log:
                    model.eval()
                    with torch.no_grad():
                        valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels,\
                        infer_time, length_of_data = validation(model, criterion, valid_loader, converter, opt, device)
                    model.train()

                    # training loss and validation loss
                    loss_log = f'[{i}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                    loss_avg.reset()

                    current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.4f}'

                    # keep best accuracy model (on valid dataset)
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                        torch.save(model.state_dict(), f'instance/models/recognition/{opt.model_name}_{manuscript_name}_best_accuracy.pth')
                    if current_norm_ED > best_norm_ED:
                        best_norm_ED = current_norm_ED
                        torch.save(model.state_dict(), f'instance/models/recognition/{opt.model_name}_{manuscript_name}_best_norm_ED.pth')
                    best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.4f}'

                    loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                    print(loss_model_log)
                    log.write(loss_model_log + '\n')

                    # show some predicted results
                    dashed_line = '-' * 80
                    head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                    predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                    
                    #show_number = min(show_number, len(labels))
                    
                    start = random.randint(0,len(labels) - show_number )    
                    for gt, pred, confidence in zip(labels[start:start+show_number], preds[start:start+show_number], confidence_score[start:start+show_number]):
                        if 'Attn' in opt.Prediction:
                            gt = gt[:gt.find('[s]')]
                            pred = pred[:pred.find('[s]')]

                        predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                    predicted_result_log += f'{dashed_line}'
                    print(predicted_result_log)
                    log.write(predicted_result_log + '\n')
                    print('validation time: ', time.time()-t1)
                    t1=time.time()
            # save model per 1e+4 iter.
            if (i + 1) % 1e+4 == 0:
                torch.save(
                    model.state_dict(), f'instance/models/recognition/{opt.model_name}/iter_{i+1}.pth')

            if i == opt.num_iter:
                print('end the training')
                del model
                torch.cuda.empty_cache()
                sys.exit()
            i += 1



    ================================================
    FILE: backend/annotator/finetune/utils.py
    ================================================
    import torch
    import pickle
    import numpy as np
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self

    ##### https://github.com/githubharald/CTCDecoder/blob/master/src/BeamSearch.py
    class BeamEntry:
        "information about one single beam at specific time-step"
        def __init__(self):
            self.prTotal = 0 # blank and non-blank
            self.prNonBlank = 0 # non-blank
            self.prBlank = 0 # blank
            self.prText = 1 # LM score
            self.lmApplied = False # flag if LM was already applied to this beam
            self.labeling = () # beam-labeling

    class BeamState:
        "information about the beams at specific time-step"
        def __init__(self):
            self.entries = {}

        def norm(self):
            "length-normalise LM score"
            for (k, _) in self.entries.items():
                labelingLen = len(self.entries[k].labeling)
                self.entries[k].prText = self.entries[k].prText ** (1.0 / (labelingLen if labelingLen else 1.0))

        def sort(self):
            "return beam-labelings, sorted by probability"
            beams = [v for (_, v) in self.entries.items()]
            sortedBeams = sorted(beams, reverse=True, key=lambda x: x.prTotal*x.prText)
            return [x.labeling for x in sortedBeams]

        def wordsearch(self, classes, ignore_idx, beamWidth, dict_list):
            beams = [v for (_, v) in self.entries.items()]
            sortedBeams = sorted(beams, reverse=True, key=lambda x: x.prTotal*x.prText)[:beamWidth]

            for j, candidate in enumerate(sortedBeams):
                idx_list = candidate.labeling
                text = ''
                for i,l in enumerate(idx_list):
                    if l not in ignore_idx and (not (i > 0 and idx_list[i - 1] == idx_list[i])):  # removing repeated characters and blank.
                        text += classes[l]

                if j == 0: best_text = text
                if text in dict_list:
                    print('found text: ', text)
                    best_text = text
                    break
                else:
                    print('not in dict: ', text)
            return best_text

    def applyLM(parentBeam, childBeam, classes, lm):
        "calculate LM score of child beam by taking score from parent beam and bigram probability of last two chars"
        if lm and not childBeam.lmApplied:
            c1 = classes[parentBeam.labeling[-1] if parentBeam.labeling else classes.index(' ')] # first char
            c2 = classes[childBeam.labeling[-1]] # second char
            lmFactor = 0.01 # influence of language model
            bigramProb = lm.getCharBigram(c1, c2) ** lmFactor # probability of seeing first and second char next to each other
            childBeam.prText = parentBeam.prText * bigramProb # probability of char sequence
            childBeam.lmApplied = True # only apply LM once per beam entry

    def addBeam(beamState, labeling):
        "add beam if it does not yet exist"
        if labeling not in beamState.entries:
            beamState.entries[labeling] = BeamEntry()

    def ctcBeamSearch(mat, classes, ignore_idx, lm, beamWidth=25, dict_list = []):
        "beam search as described by the paper of Hwang et al. and the paper of Graves et al."

        #blankIdx = len(classes)
        blankIdx = 0
        maxT, maxC = mat.shape

        # initialise beam state
        last = BeamState()
        labeling = ()
        last.entries[labeling] = BeamEntry()
        last.entries[labeling].prBlank = 1
        last.entries[labeling].prTotal = 1

        # go over all time-steps
        for t in range(maxT):
            curr = BeamState()

            # get beam-labelings of best beams
            bestLabelings = last.sort()[0:beamWidth]

            # go over best beams
            for labeling in bestLabelings:

                # probability of paths ending with a non-blank
                prNonBlank = 0
                # in case of non-empty beam
                if labeling:
                    # probability of paths with repeated last char at the end
                    prNonBlank = last.entries[labeling].prNonBlank * mat[t, labeling[-1]]

                # probability of paths ending with a blank
                prBlank = (last.entries[labeling].prTotal) * mat[t, blankIdx]

                # add beam at current time-step if needed
                addBeam(curr, labeling)

                # fill in data
                curr.entries[labeling].labeling = labeling
                curr.entries[labeling].prNonBlank += prNonBlank
                curr.entries[labeling].prBlank += prBlank
                curr.entries[labeling].prTotal += prBlank + prNonBlank
                curr.entries[labeling].prText = last.entries[labeling].prText # beam-labeling not changed, therefore also LM score unchanged from
                curr.entries[labeling].lmApplied = True # LM already applied at previous time-step for this beam-labeling

                # extend current beam-labeling
                for c in range(maxC - 1):
                    # add new char to current beam-labeling
                    newLabeling = labeling + (c,)

                    # if new labeling contains duplicate char at the end, only consider paths ending with a blank
                    if labeling and labeling[-1] == c:
                        prNonBlank = mat[t, c] * last.entries[labeling].prBlank
                    else:
                        prNonBlank = mat[t, c] * last.entries[labeling].prTotal

                    # add beam at current time-step if needed
                    addBeam(curr, newLabeling)

                    # fill in data
                    curr.entries[newLabeling].labeling = newLabeling
                    curr.entries[newLabeling].prNonBlank += prNonBlank
                    curr.entries[newLabeling].prTotal += prNonBlank

                    # apply LM
                    #applyLM(curr.entries[labeling], curr.entries[newLabeling], classes, lm)

            # set new beam state
            last = curr

        # normalise LM scores according to beam-labeling-length
        last.norm()

        # sort by probability
        #bestLabeling = last.sort()[0] # get most probable labeling

        # map labels to chars
        #res = ''
        #for idx,l in enumerate(bestLabeling):
        #    if l not in ignore_idx and (not (idx > 0 and bestLabeling[idx - 1] == bestLabeling[idx])):  # removing repeated characters and blank.
        #        res += classes[l]

        if dict_list == []:
            bestLabeling = last.sort()[0] # get most probable labeling
            res = ''
            for i,l in enumerate(bestLabeling):
                if l not in ignore_idx and (not (i > 0 and bestLabeling[i - 1] == bestLabeling[i])):  # removing repeated characters and blank.
                    res += classes[l]
        else:
            res = last.wordsearch(classes, ignore_idx, beamWidth, dict_list)

        return res
    #####

    def consecutive(data, mode ='first', stepsize=1):
        group = np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
        group = [item for item in group if len(item)>0]

        if mode == 'first': result = [l[0] for l in group]
        elif mode == 'last': result = [l[-1] for l in group]
        return result

    def word_segmentation(mat, separator_idx =  {'th': [1,2],'en': [3,4]}, separator_idx_list = [1,2,3,4]):
        result = []
        sep_list = []
        start_idx = 0
        for sep_idx in separator_idx_list:
            if sep_idx % 2 == 0: mode ='first'
            else: mode ='last'
            a = consecutive( np.argwhere(mat == sep_idx).flatten(), mode)
            new_sep = [ [item, sep_idx] for item in a]
            sep_list += new_sep
        sep_list = sorted(sep_list, key=lambda x: x[0])

        for sep in sep_list:
            for lang in separator_idx.keys():
                if sep[1] == separator_idx[lang][0]: # start lang
                    sep_lang = lang
                    sep_start_idx = sep[0]
                elif sep[1] == separator_idx[lang][1]: # end lang
                    if sep_lang == lang: # check if last entry if the same start lang
                        new_sep_pair = [lang, [sep_start_idx+1, sep[0]-1]]
                        if sep_start_idx > start_idx:
                            result.append( ['', [start_idx, sep_start_idx-1] ] )
                        start_idx = sep[0]+1
                        result.append(new_sep_pair)
                    else: # reset
                        sep_lang = ''

        if start_idx <= len(mat)-1:
            result.append( ['', [start_idx, len(mat)-1] ] )
        return result

    class CTCLabelConverter(object):
        """ Convert between text-label and text-index """

        #def __init__(self, character, separator = []):
        def __init__(self, character, separator_list = {}, dict_pathlist = {}):
            # character (str): set of the possible characters.
            dict_character = list(character)

            #special_character = ['\xa2', '\xa3', '\xa4','\xa5']
            #self.separator_char = special_character[:len(separator)]

            self.dict = {}
            #for i, char in enumerate(self.separator_char + dict_character):
            for i, char in enumerate(dict_character):
                # NOTE: 0 is reserved for 'blank' token required by CTCLoss
                self.dict[char] = i + 1

            self.character = ['[blank]'] + dict_character  # dummy '[blank]' token for CTCLoss (index 0)
            #self.character = ['[blank]']+ self.separator_char + dict_character  # dummy '[blank]' token for CTCLoss (index 0)
            self.separator_list = separator_list

            separator_char = []
            for lang, sep in separator_list.items():
                separator_char += sep

            self.ignore_idx = [0] + [i+1 for i,item in enumerate(separator_char)]

            dict_list = {}
            for lang, dict_path in dict_pathlist.items():
                with open(dict_path, "rb") as input_file:
                    word_count = pickle.load(input_file)
                dict_list[lang] = word_count
            self.dict_list = dict_list

        def encode(self, text, batch_max_length=25):
            """convert text-label into text-index.
            input:
                text: text labels of each image. [batch_size]

            output:
                text: concatenated text index for CTCLoss.
                        [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
                length: length of each text. [batch_size]
            """
            length = [len(s) for s in text]
            text = ''.join(text)
            text = [self.dict[char] for char in text]

            return (torch.IntTensor(text), torch.IntTensor(length))

        def decode_greedy(self, text_index, length):
            """ convert text-index into text-label. """
            texts = []
            index = 0
            for l in length:
                t = text_index[index:index + l]

                char_list = []
                for i in range(l):
                    if t[i] not in self.ignore_idx and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank (and separator).
                    #if (t[i] != 0) and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank (and separator).
                        char_list.append(self.character[t[i]])
                text = ''.join(char_list)

                texts.append(text)
                index += l
            return texts

        def decode_beamsearch(self, mat, beamWidth=5):
            texts = []

            for i in range(mat.shape[0]):
                t = ctcBeamSearch(mat[i], self.character, self.ignore_idx, None, beamWidth=beamWidth)
                texts.append(t)
            return texts

        def decode_wordbeamsearch(self, mat, beamWidth=5):
            texts = []
            argmax = np.argmax(mat, axis = 2)
            for i in range(mat.shape[0]):
                words = word_segmentation(argmax[i])
                string = ''
                for word in words:
                    matrix = mat[i, word[1][0]:word[1][1]+1,:]
                    if word[0] == '': dict_list = []
                    else: dict_list = self.dict_list[word[0]]
                    t = ctcBeamSearch(matrix, self.character, self.ignore_idx, None, beamWidth=beamWidth, dict_list=dict_list)
                    string += t
                texts.append(string)
            return texts

    class AttnLabelConverter(object):
        """ Convert between text-label and text-index """

        def __init__(self, character):
            # character (str): set of the possible characters.
            # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
            list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
            list_character = list(character)
            self.character = list_token + list_character

            self.dict = {}
            for i, char in enumerate(self.character):
                # print(i, char)
                self.dict[char] = i

        def encode(self, text, batch_max_length=25):
            """ convert text-label into text-index.
            input:
                text: text labels of each image. [batch_size]
                batch_max_length: max length of text label in the batch. 25 by default

            output:
                text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                    text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
                length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
            """
            length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
            # batch_max_length = max(length) # this is not allowed for multi-gpu setting
            batch_max_length += 1
            # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
            batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
            for i, t in enumerate(text):
                text = list(t)
                text.append('[s]')
                text = [self.dict[char] for char in text]
                batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
            return (batch_text.to(device), torch.IntTensor(length).to(device))

        def decode(self, text_index, length):
            """ convert text-index into text-label. """
            texts = []
            for index, l in enumerate(length):
                text = ''.join([self.character[i] for i in text_index[index, :]])
                texts.append(text)
            return texts


    class Averager(object):
        """Compute average for torch.Tensor, used for loss average."""

        def __init__(self):
            self.reset()

        def add(self, v):
            count = v.data.numel()
            v = v.data.sum()
            self.n_count += count
            self.sum += v

        def reset(self):
            self.n_count = 0
            self.sum = 0

        def val(self):
            res = 0
            if self.n_count != 0:
                res = self.sum / float(self.n_count)
            return res



    ================================================
    FILE: backend/annotator/finetune/config_files/config.yml
    ================================================
    number: '0123456789०१२३४५६७८९'
    symbol: "~!@#`$%^&*()-_+=[]\\{}|;':\",./<>?॰। "
    lang_char: 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.ँंःअअंअःआइईउऊऋएऐऑओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळवशषसह़ािीुूृॅेैॉोौ्ॐ॒क़ख़ग़ज़ड़ढ़फ़ॠ'
    experiment_name: 'devanagari_filtered'
    train_data: 'temp'
    valid_data: 'temp/val'
    manualSeed: 1111
    workers: 0
    batch_size: 32 #32
    num_iter: 5
    valInterval: 5
    saved_model: '' #'saved_models/en_filtered/iter_300000.pth'
    FT: False
    optim: False # default is Adadelta
    lr: 1.
    beta1: 0.9
    rho: 0.95
    eps: 0.00000001
    grad_clip: 5
    #Data processing
    select_data: 'train' # this is dataset folder in train_data
    batch_ratio: '1' 
    total_data_usage_ratio: 1.0
    batch_max_length: 250 
    imgH: 50
    imgW: 1150
    rgb: False
    contrast_adjust: False
    sensitive: False
    PAD: True
    contrast_adjust: 0.0
    data_filtering_off: True
    # Model Architecture
    Transformation: 'None'
    FeatureExtraction: 'ResNet'
    SequenceModeling: 'BiLSTM'
    Prediction: 'CTC'
    num_fiducial: 20
    input_channel: 1
    output_channel: 512
    hidden_size: 512
    decode: 'greedy'
    new_prediction: True
    freeze_FeatureFxtraction: False
    freeze_SequenceModeling: False


    ================================================
    FILE: backend/annotator/finetune/modules/feature_extraction.py
    ================================================
    import torch.nn as nn
    import torch.nn.functional as F


    class VGG_FeatureExtractor(nn.Module):
        """ FeatureExtractor of CRNN (https://arxiv.org/pdf/1507.05717.pdf) """

        def __init__(self, input_channel, output_channel=512):
            super(VGG_FeatureExtractor, self).__init__()
            self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                                int(output_channel / 2), output_channel]  # [64, 128, 256, 512]
            self.ConvNet = nn.Sequential(
                nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
                nn.MaxPool2d(2, 2),  # 64x16x50
                nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
                nn.MaxPool2d(2, 2),  # 128x8x25
                nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True),  # 256x8x25
                nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
                nn.MaxPool2d((2, 1), (2, 1)),  # 256x4x25
                nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),  # 512x4x25
                nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
                nn.MaxPool2d((2, 1), (2, 1)),  # 512x2x25
                nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0), nn.ReLU(True))  # 512x1x24

        def forward(self, input):
            return self.ConvNet(input)


    class RCNN_FeatureExtractor(nn.Module):
        """ FeatureExtractor of GRCNN (https://papers.nips.cc/paper/6637-gated-recurrent-convolution-neural-network-for-ocr.pdf) """

        def __init__(self, input_channel, output_channel=512):
            super(RCNN_FeatureExtractor, self).__init__()
            self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                                int(output_channel / 2), output_channel]  # [64, 128, 256, 512]
            self.ConvNet = nn.Sequential(
                nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
                nn.MaxPool2d(2, 2),  # 64 x 16 x 50
                GRCL(self.output_channel[0], self.output_channel[0], num_iteration=5, kernel_size=3, pad=1),
                nn.MaxPool2d(2, 2),  # 64 x 8 x 25
                GRCL(self.output_channel[0], self.output_channel[1], num_iteration=5, kernel_size=3, pad=1),
                nn.MaxPool2d(2, (2, 1), (0, 1)),  # 128 x 4 x 26
                GRCL(self.output_channel[1], self.output_channel[2], num_iteration=5, kernel_size=3, pad=1),
                nn.MaxPool2d(2, (2, 1), (0, 1)),  # 256 x 2 x 27
                nn.Conv2d(self.output_channel[2], self.output_channel[3], 2, 1, 0, bias=False),
                nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True))  # 512 x 1 x 26

        def forward(self, input):
            return self.ConvNet(input)


    class ResNet_FeatureExtractor(nn.Module):
        """ FeatureExtractor of FAN (http://openaccess.thecvf.com/content_ICCV_2017/papers/Cheng_Focusing_Attention_Towards_ICCV_2017_paper.pdf) """

        def __init__(self, input_channel, output_channel=512):
            super(ResNet_FeatureExtractor, self).__init__()
            self.ConvNet = ResNet(input_channel, output_channel, BasicBlock, [1, 2, 5, 3])

        def forward(self, input):
            return self.ConvNet(input)


    # For Gated RCNN
    class GRCL(nn.Module):

        def __init__(self, input_channel, output_channel, num_iteration, kernel_size, pad):
            super(GRCL, self).__init__()
            self.wgf_u = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False)
            self.wgr_x = nn.Conv2d(output_channel, output_channel, 1, 1, 0, bias=False)
            self.wf_u = nn.Conv2d(input_channel, output_channel, kernel_size, 1, pad, bias=False)
            self.wr_x = nn.Conv2d(output_channel, output_channel, kernel_size, 1, pad, bias=False)

            self.BN_x_init = nn.BatchNorm2d(output_channel)

            self.num_iteration = num_iteration
            self.GRCL = [GRCL_unit(output_channel) for _ in range(num_iteration)]
            self.GRCL = nn.Sequential(*self.GRCL)

        def forward(self, input):
            """ The input of GRCL is consistant over time t, which is denoted by u(0)
            thus wgf_u / wf_u is also consistant over time t.
            """
            wgf_u = self.wgf_u(input)
            wf_u = self.wf_u(input)
            x = F.relu(self.BN_x_init(wf_u))

            for i in range(self.num_iteration):
                x = self.GRCL[i](wgf_u, self.wgr_x(x), wf_u, self.wr_x(x))

            return x


    class GRCL_unit(nn.Module):

        def __init__(self, output_channel):
            super(GRCL_unit, self).__init__()
            self.BN_gfu = nn.BatchNorm2d(output_channel)
            self.BN_grx = nn.BatchNorm2d(output_channel)
            self.BN_fu = nn.BatchNorm2d(output_channel)
            self.BN_rx = nn.BatchNorm2d(output_channel)
            self.BN_Gx = nn.BatchNorm2d(output_channel)

        def forward(self, wgf_u, wgr_x, wf_u, wr_x):
            G_first_term = self.BN_gfu(wgf_u)
            G_second_term = self.BN_grx(wgr_x)
            G = F.sigmoid(G_first_term + G_second_term)

            x_first_term = self.BN_fu(wf_u)
            x_second_term = self.BN_Gx(self.BN_rx(wr_x) * G)
            x = F.relu(x_first_term + x_second_term)

            return x


    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, inplanes, planes, stride=1, downsample=None):
            super(BasicBlock, self).__init__()
            self.conv1 = self._conv3x3(inplanes, planes)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = self._conv3x3(planes, planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride

        def _conv3x3(self, in_planes, out_planes, stride=1):
            "3x3 convolution with padding"
            return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                            padding=1, bias=False)

        def forward(self, x):
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
            out = self.relu(out)

            return out


    class ResNet(nn.Module):

        def __init__(self, input_channel, output_channel, block, layers):
            super(ResNet, self).__init__()

            self.output_channel_block = [int(output_channel / 4), int(output_channel / 2), output_channel, output_channel]

            self.inplanes = int(output_channel / 8)
            self.conv0_1 = nn.Conv2d(input_channel, int(output_channel / 16),
                                    kernel_size=3, stride=1, padding=1, bias=False)
            self.bn0_1 = nn.BatchNorm2d(int(output_channel / 16))
            self.conv0_2 = nn.Conv2d(int(output_channel / 16), self.inplanes,
                                    kernel_size=3, stride=1, padding=1, bias=False)
            self.bn0_2 = nn.BatchNorm2d(self.inplanes)
            self.relu = nn.ReLU(inplace=True)

            self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
            self.conv1 = nn.Conv2d(self.output_channel_block[0], self.output_channel_block[
                                0], kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])

            self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.layer2 = self._make_layer(block, self.output_channel_block[1], layers[1], stride=1)
            self.conv2 = nn.Conv2d(self.output_channel_block[1], self.output_channel_block[
                                1], kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(self.output_channel_block[1])

            self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
            self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
            self.conv3 = nn.Conv2d(self.output_channel_block[2], self.output_channel_block[
                                2], kernel_size=3, stride=1, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])

            self.layer4 = self._make_layer(block, self.output_channel_block[3], layers[3], stride=1)
            self.conv4_1 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[
                                    3], kernel_size=2, stride=(2, 1), padding=(0, 1), bias=False)
            self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
            self.conv4_2 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[
                                    3], kernel_size=2, stride=1, padding=0, bias=False)
            self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])

        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv0_1(x)
            x = self.bn0_1(x)
            x = self.relu(x)
            x = self.conv0_2(x)
            x = self.bn0_2(x)
            x = self.relu(x)

            x = self.maxpool1(x)
            x = self.layer1(x)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.maxpool2(x)
            x = self.layer2(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)

            x = self.maxpool3(x)
            x = self.layer3(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)

            x = self.layer4(x)
            x = self.conv4_1(x)
            x = self.bn4_1(x)
            x = self.relu(x)
            x = self.conv4_2(x)
            x = self.bn4_2(x)
            x = self.relu(x)

            return x



    ================================================
    FILE: backend/annotator/finetune/modules/prediction.py
    ================================================
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    class Attention(nn.Module):

        def __init__(self, input_size, hidden_size, num_classes):
            super(Attention, self).__init__()
            self.attention_cell = AttentionCell(input_size, hidden_size, num_classes)
            self.hidden_size = hidden_size
            self.num_classes = num_classes
            self.generator = nn.Linear(hidden_size, num_classes)

        def _char_to_onehot(self, input_char, onehot_dim=38):
            input_char = input_char.unsqueeze(1)
            batch_size = input_char.size(0)
            one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(device)
            one_hot = one_hot.scatter_(1, input_char, 1)
            return one_hot

        def forward(self, batch_H, text, is_train=True, batch_max_length=25):
            """
            input:
                batch_H : contextual_feature H = hidden state of encoder. [batch_size x num_steps x num_classes]
                text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [GO] token. text[:, 0] = [GO].
            output: probability distribution at each step [batch_size x num_steps x num_classes]
            """
            batch_size = batch_H.size(0)
            num_steps = batch_max_length + 1  # +1 for [s] at end of sentence.

            output_hiddens = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0).to(device)
            hidden = (torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device),
                    torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device))

            if is_train:
                for i in range(num_steps):
                    # one-hot vectors for a i-th char. in a batch
                    char_onehots = self._char_to_onehot(text[:, i], onehot_dim=self.num_classes)
                    # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                    hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                    output_hiddens[:, i, :] = hidden[0]  # LSTM hidden index (0: hidden, 1: Cell)
                probs = self.generator(output_hiddens)

            else:
                targets = torch.LongTensor(batch_size).fill_(0).to(device)  # [GO] token
                probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0).to(device)

                for i in range(num_steps):
                    char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                    hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                    probs_step = self.generator(hidden[0])
                    probs[:, i, :] = probs_step
                    _, next_input = probs_step.max(1)
                    targets = next_input

            return probs  # batch_size x num_steps x num_classes


    class AttentionCell(nn.Module):

        def __init__(self, input_size, hidden_size, num_embeddings):
            super(AttentionCell, self).__init__()
            self.i2h = nn.Linear(input_size, hidden_size, bias=False)
            self.h2h = nn.Linear(hidden_size, hidden_size)  # either i2i or h2h should have bias
            self.score = nn.Linear(hidden_size, 1, bias=False)
            self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
            self.hidden_size = hidden_size

        def forward(self, prev_hidden, batch_H, char_onehots):
            # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
            batch_H_proj = self.i2h(batch_H)
            prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
            e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj))  # batch_size x num_encoder_step * 1

            alpha = F.softmax(e, dim=1)
            context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)  # batch_size x num_channel
            concat_context = torch.cat([context, char_onehots], 1)  # batch_size x (num_channel + num_embedding)
            cur_hidden = self.rnn(concat_context, prev_hidden)
            return cur_hidden, alpha



    ================================================
    FILE: backend/annotator/finetune/modules/sequence_modeling.py
    ================================================
    import torch.nn as nn


    class BidirectionalLSTM(nn.Module):

        def __init__(self, input_size, hidden_size, output_size):
            super(BidirectionalLSTM, self).__init__()
            self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
            self.linear = nn.Linear(hidden_size * 2, output_size)

        def forward(self, input):
            """
            input : visual feature [batch_size x T x input_size]
            output : contextual feature [batch_size x T x output_size]
            """
            try:
                self.rnn.flatten_parameters()
            except:
                pass
            recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
            output = self.linear(recurrent)  # batch_size x T x output_size
            return output



    ================================================
    FILE: backend/annotator/finetune/modules/transformation.py
    ================================================
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    class TPS_SpatialTransformerNetwork(nn.Module):
        """ Rectification Network of RARE, namely TPS based STN """

        def __init__(self, F, I_size, I_r_size, I_channel_num=1):
            """ Based on RARE TPS
            input:
                batch_I: Batch Input Image [batch_size x I_channel_num x I_height x I_width]
                I_size : (height, width) of the input image I
                I_r_size : (height, width) of the rectified image I_r
                I_channel_num : the number of channels of the input image I
            output:
                batch_I_r: rectified image [batch_size x I_channel_num x I_r_height x I_r_width]
            """
            super(TPS_SpatialTransformerNetwork, self).__init__()
            self.F = F
            self.I_size = I_size
            self.I_r_size = I_r_size  # = (I_r_height, I_r_width)
            self.I_channel_num = I_channel_num
            self.LocalizationNetwork = LocalizationNetwork(self.F, self.I_channel_num)
            self.GridGenerator = GridGenerator(self.F, self.I_r_size)

        def forward(self, batch_I):
            batch_C_prime = self.LocalizationNetwork(batch_I)  # batch_size x K x 2
            build_P_prime = self.GridGenerator.build_P_prime(batch_C_prime)  # batch_size x n (= I_r_width x I_r_height) x 2
            build_P_prime_reshape = build_P_prime.reshape([build_P_prime.size(0), self.I_r_size[0], self.I_r_size[1], 2])
            batch_I_r = F.grid_sample(batch_I, build_P_prime_reshape, padding_mode='border')

            return batch_I_r


    class LocalizationNetwork(nn.Module):
        """ Localization Network of RARE, which predicts C' (K x 2) from I (I_width x I_height) """

        def __init__(self, F, I_channel_num):
            super(LocalizationNetwork, self).__init__()
            self.F = F
            self.I_channel_num = I_channel_num
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=self.I_channel_num, out_channels=64, kernel_size=3, stride=1, padding=1,
                        bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
                nn.MaxPool2d(2, 2),  # batch_size x 64 x I_height/2 x I_width/2
                nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
                nn.MaxPool2d(2, 2),  # batch_size x 128 x I_height/4 x I_width/4
                nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
                nn.MaxPool2d(2, 2),  # batch_size x 256 x I_height/8 x I_width/8
                nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),
                nn.AdaptiveAvgPool2d(1)  # batch_size x 512
            )

            self.localization_fc1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True))
            self.localization_fc2 = nn.Linear(256, self.F * 2)

            # Init fc2 in LocalizationNetwork
            self.localization_fc2.weight.data.fill_(0)
            """ see RARE paper Fig. 6 (a) """
            ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
            ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
            ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
            ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
            ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
            initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
            self.localization_fc2.bias.data = torch.from_numpy(initial_bias).float().view(-1)

        def forward(self, batch_I):
            """
            input:     batch_I : Batch Input Image [batch_size x I_channel_num x I_height x I_width]
            output:    batch_C_prime : Predicted coordinates of fiducial points for input batch [batch_size x F x 2]
            """
            batch_size = batch_I.size(0)
            features = self.conv(batch_I).view(batch_size, -1)
            batch_C_prime = self.localization_fc2(self.localization_fc1(features)).view(batch_size, self.F, 2)
            return batch_C_prime


    class GridGenerator(nn.Module):
        """ Grid Generator of RARE, which produces P_prime by multiplying T with P """

        def __init__(self, F, I_r_size):
            """ Generate P_hat and inv_delta_C for later """
            super(GridGenerator, self).__init__()
            self.eps = 1e-6
            self.I_r_height, self.I_r_width = I_r_size
            self.F = F
            self.C = self._build_C(self.F)  # F x 2
            self.P = self._build_P(self.I_r_width, self.I_r_height)
            ## for multi-gpu, you need register buffer
            self.register_buffer("inv_delta_C", torch.tensor(self._build_inv_delta_C(self.F, self.C)).float())  # F+3 x F+3
            self.register_buffer("P_hat", torch.tensor(self._build_P_hat(self.F, self.C, self.P)).float())  # n x F+3
            ## for fine-tuning with different image width, you may use below instead of self.register_buffer
            #self.inv_delta_C = torch.tensor(self._build_inv_delta_C(self.F, self.C)).float().cuda()  # F+3 x F+3
            #self.P_hat = torch.tensor(self._build_P_hat(self.F, self.C, self.P)).float().cuda()  # n x F+3

        def _build_C(self, F):
            """ Return coordinates of fiducial points in I_r; C """
            ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
            ctrl_pts_y_top = -1 * np.ones(int(F / 2))
            ctrl_pts_y_bottom = np.ones(int(F / 2))
            ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
            ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
            C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
            return C  # F x 2

        def _build_inv_delta_C(self, F, C):
            """ Return inv_delta_C which is needed to calculate T """
            hat_C = np.zeros((F, F), dtype=float)  # F x F
            for i in range(0, F):
                for j in range(i, F):
                    r = np.linalg.norm(C[i] - C[j])
                    hat_C[i, j] = r
                    hat_C[j, i] = r
            np.fill_diagonal(hat_C, 1)
            hat_C = (hat_C ** 2) * np.log(hat_C)
            # print(C.shape, hat_C.shape)
            delta_C = np.concatenate(  # F+3 x F+3
                [
                    np.concatenate([np.ones((F, 1)), C, hat_C], axis=1),  # F x F+3
                    np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1),  # 2 x F+3
                    np.concatenate([np.zeros((1, 3)), np.ones((1, F))], axis=1)  # 1 x F+3
                ],
                axis=0
            )
            inv_delta_C = np.linalg.inv(delta_C)
            return inv_delta_C  # F+3 x F+3

        def _build_P(self, I_r_width, I_r_height):
            I_r_grid_x = (np.arange(-I_r_width, I_r_width, 2) + 1.0) / I_r_width  # self.I_r_width
            I_r_grid_y = (np.arange(-I_r_height, I_r_height, 2) + 1.0) / I_r_height  # self.I_r_height
            P = np.stack(  # self.I_r_width x self.I_r_height x 2
                np.meshgrid(I_r_grid_x, I_r_grid_y),
                axis=2
            )
            return P.reshape([-1, 2])  # n (= self.I_r_width x self.I_r_height) x 2

        def _build_P_hat(self, F, C, P):
            n = P.shape[0]  # n (= self.I_r_width x self.I_r_height)
            P_tile = np.tile(np.expand_dims(P, axis=1), (1, F, 1))  # n x 2 -> n x 1 x 2 -> n x F x 2
            C_tile = np.expand_dims(C, axis=0)  # 1 x F x 2
            P_diff = P_tile - C_tile  # n x F x 2
            rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)  # n x F
            rbf = np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))  # n x F
            P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
            return P_hat  # n x F+3

        def build_P_prime(self, batch_C_prime):
            """ Generate Grid from batch_C_prime [batch_size x F x 2] """
            batch_size = batch_C_prime.size(0)
            batch_inv_delta_C = self.inv_delta_C.repeat(batch_size, 1, 1)
            batch_P_hat = self.P_hat.repeat(batch_size, 1, 1)
            batch_C_prime_with_zeros = torch.cat((batch_C_prime, torch.zeros(
                batch_size, 3, 2).float().to(device)), dim=1)  # batch_size x F+3 x 2
            batch_T = torch.bmm(batch_inv_delta_C, batch_C_prime_with_zeros)  # batch_size x F+3 x 2
            batch_P_prime = torch.bmm(batch_P_hat, batch_T)  # batch_size x n x 2
            return batch_P_prime  # batch_size x n x 2



    ================================================
    FILE: backend/annotator/recognition/__init__.py
    ================================================
    [Empty file]


    ================================================
    FILE: backend/annotator/recognition/dataset.py
    ================================================
    import os
    import sys
    import re
    import six
    import math
    import lmdb
    import torch

    from natsort import natsorted
    from PIL import Image
    import numpy as np
    from torch.utils.data import Dataset, ConcatDataset, Subset
    # from torch._utils import _accumulate
    from itertools import accumulate as _accumulate
    import torchvision.transforms as transforms


    class Batch_Balanced_Dataset(object):

        def __init__(self, opt):
            """
            Modulate the data ratio in the batch.
            For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
            the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.
            """
            log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')
            dashed_line = '-' * 80
            print(dashed_line)
            log.write(dashed_line + '\n')
            print(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
            log.write(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}\n')
            assert len(opt.select_data) == len(opt.batch_ratio)
            
            _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
            self.data_loader_list = []
            self.dataloader_iter_list = []
            batch_size_list = []
            Total_batch_size = 0
            for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
                _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
                print(dashed_line)
                log.write(dashed_line + '\n')
                _dataset, _dataset_log = hierarchical_dataset(root=opt.train_data, opt=opt, select_data=[selected_d])
                total_number_dataset = len(_dataset)
                log.write(_dataset_log)

                """
                The total number of data can be modified with opt.total_data_usage_ratio.
                ex) opt.total_data_usage_ratio = 1 indicates 100% usage, and 0.2 indicates 20% usage.
                See 4.2 section in our paper.
                """
                number_dataset = int(total_number_dataset * float(opt.total_data_usage_ratio))
                dataset_split = [number_dataset, total_number_dataset - number_dataset]
                indices = range(total_number_dataset)
                _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                            for offset, length in zip(_accumulate(dataset_split), dataset_split)]
                selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
                selected_d_log += f'num samples of {selected_d} per batch: {opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}'
                print(selected_d_log)
                log.write(selected_d_log + '\n')
                batch_size_list.append(str(_batch_size))
                Total_batch_size += _batch_size

                _data_loader = torch.utils.data.DataLoader(
                    _dataset, batch_size=_batch_size,
                    shuffle=True,
                    num_workers=int(opt.workers),
                    collate_fn=_AlignCollate, pin_memory=True)
                self.data_loader_list.append(_data_loader)
                self.dataloader_iter_list.append(iter(_data_loader))

            Total_batch_size_log = f'{dashed_line}\n'
            batch_size_sum = '+'.join(batch_size_list)
            Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
            Total_batch_size_log += f'{dashed_line}'
            opt.batch_size = Total_batch_size

            print(Total_batch_size_log)
            log.write(Total_batch_size_log + '\n')
            log.close()

        def get_batch(self):
            balanced_batch_images = []
            balanced_batch_texts = []

            for i, data_loader_iter in enumerate(self.dataloader_iter_list):
                try:
                    image, text = next(data_loader_iter)
                    balanced_batch_images.append(image)
                    balanced_batch_texts += text
                except StopIteration:
                    self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                    image, text = next(self.dataloader_iter_list[i])
                    balanced_batch_images.append(image)
                    balanced_batch_texts += text
                except ValueError:
                    pass

            balanced_batch_images = torch.cat(balanced_batch_images, 0)

            return balanced_batch_images, balanced_batch_texts


    def hierarchical_dataset(root, opt, select_data='/'):
        """ select_data='/' contains all sub-directory of root directory """
        dataset_list = []
        dataset_log = f'dataset_root:    {root}\t dataset: {select_data[0]}'
        print(dataset_log)
        dataset_log += '\n'

        for dirpath, dirnames, filenames in os.walk(root+'/'):
            
            if not dirnames:
                select_flag = False
                for selected_d in select_data:
                    if selected_d in dirpath:
                        select_flag = True
                        break

                if select_flag:
                    dataset = LmdbDataset(dirpath, opt)
                    sub_dataset_log = f'sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}'
                    print(sub_dataset_log)
                    dataset_log += f'{sub_dataset_log}\n'
                    dataset_list.append(dataset)
        
        concatenated_dataset = ConcatDataset(dataset_list)

        return concatenated_dataset, dataset_log


    class LmdbDataset(Dataset):

        def __init__(self, root, opt):

            self.root = root
            self.opt = opt
            self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
            if not self.env:
                print('cannot create lmdb from %s' % (root))
                sys.exit(0)

            with self.env.begin(write=False) as txn:
                nSamples = int(txn.get('num-samples'.encode()))
                self.nSamples = nSamples

                if self.opt.data_filtering_off:
                    # for fast check or benchmark evaluation with no filtering
                    self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
                else:
                    """ Filtering part
                    If you want to evaluate IC15-2077 & CUTE datasets which have special character labels,
                    use --data_filtering_off and only evaluate on alphabets and digits.
                    see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L190-L192

                    And if you want to evaluate them with the model trained with --sensitive option,
                    use --sensitive and --data_filtering_off,
                    see https://github.com/clovaai/deep-text-recognition-benchmark/blob/dff844874dbe9e0ec8c5a52a7bd08c7f20afe704/test.py#L137-L144
                    """
                    self.filtered_index_list = []
                    for index in range(self.nSamples):
                        index += 1  # lmdb starts with 1
                        label_key = 'label-%09d'.encode() % index
                        label = txn.get(label_key).decode('utf-8')

                        if len(label) > self.opt.batch_max_length:
                            # print(f'The length of the label is longer than max_length: length
                            # {len(label)}, {label} in dataset {self.root}')
                            continue

                        # By default, images containing characters which are not in opt.character are filtered.
                        # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
                        out_of_char = f'[^{self.opt.character}]'
                        if re.search(out_of_char, label.lower()):
                            continue

                        self.filtered_index_list.append(index)

                    self.nSamples = len(self.filtered_index_list)

        def __len__(self):
            return self.nSamples

        def __getitem__(self, index):
            assert index <= len(self), 'index range error'
            index = self.filtered_index_list[index]

            with self.env.begin(write=False) as txn:
                label_key = 'label-%09d'.encode() % index
                label = txn.get(label_key).decode('utf-8')
                img_key = 'image-%09d'.encode() % index
                imgbuf = txn.get(img_key)

                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                try:
                    if self.opt.rgb:
                        img = Image.open(buf).convert('RGB')  # for color image
                    else:
                        img = Image.open(buf).convert('L')

                except IOError:
                    print(f'Corrupted image for {index}')
                    # make dummy image and dummy label for corrupted image.
                    if self.opt.rgb:
                        img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                    else:
                        img = Image.new('L', (self.opt.imgW, self.opt.imgH))
                    label = '[dummy_label]'

                if not self.opt.sensitive:
                    label = label.lower()

                # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
                out_of_char = f'[^{self.opt.character}]'
                label = re.sub(out_of_char, '', label)

            return (img, label)


    class RawDataset(Dataset):

        def __init__(self, root, opt):
            self.opt = opt
            self.image_path_list = []
            for dirpath, dirnames, filenames in os.walk(root):
                for name in filenames:
                    _, ext = os.path.splitext(name)
                    ext = ext.lower()
                    if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                        self.image_path_list.append(os.path.join(dirpath, name))

            self.image_path_list = natsorted(self.image_path_list)
            self.nSamples = len(self.image_path_list)

        def __len__(self):
            return self.nSamples

        def __getitem__(self, index):

            try:
                if self.opt.rgb:
                    img = Image.open(self.image_path_list[index]).convert('RGB')  # for color image
                else:
                    img = Image.open(self.image_path_list[index]).convert('L')

            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                if self.opt.rgb:
                    img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                else:
                    img = Image.new('L', (self.opt.imgW, self.opt.imgH))

            return (img, self.image_path_list[index])


    class ResizeNormalize(object):

        def __init__(self, size, interpolation=Image.BICUBIC):
            self.size = size
            self.interpolation = interpolation
            self.toTensor = transforms.ToTensor()

        def __call__(self, img):
            img = img.resize(self.size, self.interpolation)
            img = self.toTensor(img)
            img.sub_(0.5).div_(0.5)
            return img


    class NormalizePAD(object):

        def __init__(self, max_size, PAD_type='right'):
            self.toTensor = transforms.ToTensor()
            self.max_size = max_size
            self.max_width_half = math.floor(max_size[2] / 2)
            self.PAD_type = PAD_type

        def __call__(self, img):
            img = self.toTensor(img)
            img.sub_(0.5).div_(0.5)
            c, h, w = img.size()
            Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
            Pad_img[:, :, :w] = img  # right pad
            if self.max_size[2] != w:  # add border Pad
                Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

            return Pad_img


    class AlignCollate(object):

        def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
            self.imgH = imgH
            self.imgW = imgW
            self.keep_ratio_with_pad = keep_ratio_with_pad

        def __call__(self, batch):
            batch = filter(lambda x: x is not None, batch)
            images, labels = zip(*batch)

            if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
                resized_max_w = self.imgW
                input_channel = 3 if images[0].mode == 'RGB' else 1
                transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

                resized_images = []
                for image in images:
                    w, h = image.size
                    ratio = w / float(h)
                    if math.ceil(self.imgH * ratio) > self.imgW:
                        resized_w = self.imgW
                    else:
                        resized_w = math.ceil(self.imgH * ratio)

                    resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                    resized_images.append(transform(resized_image))
                    # resized_image.save('./image_test/%d_test.jpg' % w)

                image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

            else:
                transform = ResizeNormalize((self.imgW, self.imgH))
                image_tensors = [transform(image) for image in images]
                image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

            return image_tensors, labels


    def tensor2im(image_tensor, imtype=np.uint8):
        image_numpy = image_tensor.cpu().float().numpy()
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        return image_numpy.astype(imtype)


    def save_image(image_numpy, image_path):
        image_pil = Image.fromarray(image_numpy)
        image_pil.save(image_path)



    ================================================
    FILE: backend/annotator/recognition/demo.py
    ================================================
    import string
    import torch
    import torch.backends.cudnn as cudnn
    import torch.utils.data
    import torch.nn.functional as F

    from annotator.recognition.utils import CTCLabelConverter, AttnLabelConverter
    from annotator.recognition.dataset import RawDataset, AlignCollate
    from annotator.recognition.model import Model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class OCRConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)


    def recognise_lines(
        image_folder,
        saved_model,
        transformation,
        feature_extraction,
        sequence_modeling,
        prediction,
        batch_size=192,
        workers=4,
        batch_max_length=25,
        imgH=32,
        imgW=100,
        rgb=False,
        character=None,
        sensitive=False,
        pad=False,
        num_fiducial=20,
        input_channel=1,
        output_channel=512,
        hidden_size=256,
    ):
        """
        Recognise text lines from images in the specified folder using the specified model.

        Parameters:
            image_folder (str): Path to the folder containing images.
            saved_model (str): Path to the pretrained model.
            transformation (str): Transformation stage. Options: None, TPS.
            feature_extraction (str): Feature extraction stage. Options: VGG, RCNN, ResNet.
            sequence_modeling (str): Sequence modeling stage. Options: None, BiLSTM.
            prediction (str): Prediction stage. Options: CTC, Attn.
            batch_size (int): Batch size for processing images.
            workers (int): Number of workers for data loading.
            batch_max_length (int): Maximum label length.
            imgH (int): Height of input images.
            imgW (int): Width of input images.
            rgb (bool): Whether to use RGB input.
            character (str): Character set for labels.
            sensitive (bool): Use sensitive character mode.
            pad (bool): Whether to pad resized images to maintain aspect ratio.
            num_fiducial (int): Number of fiducial points for TPS-STN.
            input_channel (int): Number of input channels for the feature extractor.
            output_channel (int): Number of output channels for the feature extractor.
            hidden_size (int): Size of the LSTM hidden state.

        Returns:
            results (list): List of dictionaries containing image paths, predicted labels, and confidence scores.
        """

        # Configure the character set
        if sensitive:
            character = string.printable[:-6] if character is None else character
        elif character is None:
            character = "0123456789abcdefghijklmnopqrstuvwxyz"

        # Initialize label converter
        if "CTC" in prediction:
            converter = CTCLabelConverter(character)
        else:
            converter = AttnLabelConverter(character)
        num_class = len(converter.character)

        # Set input channel for RGB images
        input_channel = 3 if rgb else input_channel

        opt = OCRConfig(
            image_folder=image_folder,
            saved_model=saved_model,
            Transformation=transformation,
            FeatureExtraction=feature_extraction,
            SequenceModeling=sequence_modeling,
            Prediction=prediction,
            batch_size=batch_size,
            workers=workers,
            batch_max_length=batch_max_length,
            imgH=imgH,
            imgW=imgW,
            rgb=rgb,
            character=character,
            sensitive=sensitive,
            PAD=pad,
            num_fiducial=num_fiducial,
            input_channel=input_channel,
            output_channel=output_channel,
            hidden_size=hidden_size,
            num_class=num_class,
        )

        # Load the model
        model = Model(opt)
        model = torch.nn.DataParallel(model).to(device)

        print(f"Loading pretrained model from {saved_model}")
        model.load_state_dict(torch.load(saved_model, map_location=device))

        # Prepare data loader
        AlignCollate_demo = AlignCollate(imgH=imgH, imgW=imgW, keep_ratio_with_pad=pad)
        demo_data = RawDataset(root=image_folder, opt=opt)  # Use RawDataset
        demo_loader = torch.utils.data.DataLoader(
            demo_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            collate_fn=AlignCollate_demo,
            pin_memory=True,
        )

        # Perform prediction
        model.eval()
        results = []
        with torch.no_grad():
            for image_tensors, image_path_list in demo_loader:
                batch_size = image_tensors.size(0)
                image = image_tensors.to(device)
                length_for_pred = torch.IntTensor([batch_max_length] * batch_size).to(
                    device
                )
                text_for_pred = (
                    torch.LongTensor(batch_size, batch_max_length + 1).fill_(0).to(device)
                )

                if "CTC" in prediction:
                    preds = model(image, text_for_pred)
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    _, preds_index = preds.max(2)
                    preds_str = converter.decode(preds_index, preds_size)
                    del preds_size, preds_index
                else:
                    preds = model(image, text_for_pred, is_train=False)
                    _, preds_index = preds.max(2)
                    preds_str = converter.decode(preds_index, length_for_pred)

                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)

                for img_name, pred, pred_max_prob in zip(
                    image_path_list, preds_str, preds_max_prob
                ):
                    if "Attn" in prediction:
                        pred_EOS = pred.find("[s]")
                        pred = pred[:pred_EOS]
                        pred_max_prob = pred_max_prob[:pred_EOS]

                    confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                    results.append(
                        {
                            "image_path": img_name,
                            "predicted_label": pred,
                            "confidence_score": confidence_score.item(),
                        }
                    )
                del image, length_for_pred, text_for_pred, preds, preds_prob, preds_max_prob
                if 'preds_size' in locals(): del preds_size
                if 'preds_index' in locals(): del preds_index
                torch.cuda.empty_cache()

        
        # clear GPU memory
        del model
        del demo_loader, AlignCollate_demo, demo_data
        torch.cuda.empty_cache()

        return results



    ================================================
    FILE: backend/annotator/recognition/model.py
    ================================================
    """
    Copyright (c) 2019-present NAVER Corp.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    """

    import torch.nn as nn

    from annotator.recognition.modules.transformation import TPS_SpatialTransformerNetwork
    from annotator.recognition.modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
    from annotator.recognition.modules.sequence_modeling import BidirectionalLSTM
    from annotator.recognition.modules.prediction import Attention


    class Model(nn.Module):

        def __init__(self, opt):
            super(Model, self).__init__()
            self.opt = opt
            self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                        'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

            """ Transformation """
            if opt.Transformation == 'TPS':
                self.Transformation = TPS_SpatialTransformerNetwork(
                    F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
            else:
                print('No Transformation module specified')

            """ FeatureExtraction """
            if opt.FeatureExtraction == 'VGG':
                self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
            elif opt.FeatureExtraction == 'RCNN':
                self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
            elif opt.FeatureExtraction == 'ResNet':
                self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
            else:
                raise Exception('No FeatureExtraction module specified')
            self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
            self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

            """ Sequence modeling"""
            if opt.SequenceModeling == 'BiLSTM':
                self.SequenceModeling = nn.Sequential(
                    BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                    BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
                self.SequenceModeling_output = opt.hidden_size
            else:
                print('No SequenceModeling module specified')
                self.SequenceModeling_output = self.FeatureExtraction_output

            """ Prediction """
            if opt.Prediction == 'CTC':
                self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
            elif opt.Prediction == 'Attn':
                self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)
            else:
                raise Exception('Prediction is neither CTC or Attn')

        def forward(self, input, text, is_train=True):
            """ Transformation stage """
            if not self.stages['Trans'] == None:
                input = self.Transformation(input)

            """ Feature extraction stage """
            visual_feature = self.FeatureExtraction(input)
            visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
            visual_feature = visual_feature.squeeze(3)

            """ Sequence modeling stage """
            if self.stages['Seq'] == 'BiLSTM':
                contextual_feature = self.SequenceModeling(visual_feature)
            else:
                contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

            """ Prediction stage """
            if self.stages['Pred'] == 'CTC':
                prediction = self.Prediction(contextual_feature.contiguous())
            else:
                prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)

            return prediction



    ================================================
    FILE: backend/annotator/recognition/recognition.py
    ================================================
    import os
    import subprocess
    import torch

    from datetime import datetime
    from flask import current_app

    from annotator.recognition.demo import recognise_lines
    from annotator.models import db, RecognitionLog

    def get_filename_without_extension(file_path):
        """
        Extracts the filename without extension from a given file path.

        :param file_path: str - The full file path.
        :return: str - The filename without the extension.
        """
        # Extract the base name of the file
        base_name = os.path.basename(file_path)
        # Remove the file extension
        file_name, _ = os.path.splitext(base_name)
        return file_name

    def get_subfolders(folder_path):
        return [subfolder for subfolder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subfolder))]

    def recognise_characters(folder_path, model, manuscript_name):
        lines_of_all_pages = {}
        lines_folder_path = os.path.join(folder_path, "lines")
        page_subfolders = get_subfolders(lines_folder_path)
        for page_subfolder in page_subfolders:
            lines_of_one_page = recognise_lines(
                image_folder=os.path.join(lines_folder_path, page_subfolder),
                saved_model=os.path.join(current_app.config['DATA_PATH'], 'models', 'recognition', model),
                transformation=None,
                feature_extraction="ResNet",
                sequence_modeling="BiLSTM",
                prediction="CTC",
                workers=0,
                batch_max_length=250,
                imgH=50,
                imgW=2000,
                pad=True,
                character="""`0123456789~!@#$%^&*()-_+=[]\\{}|;':",./<>? abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.ँंःअअंअःआइईउऊऋएऐऑओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळवशषसह़ािीुूृॅेैॉोौ्ॐ॒क़ख़ग़ज़ड़ढ़फ़ॠ।०१२३४५६७८९॰""",
                hidden_size=512,
                output_channel=512,
            )
            for line in lines_of_one_page:
                line["manuscript_name"] = manuscript_name
                line["selected_model"] = model
                line["page"] = page_subfolder
                line["line"] = get_filename_without_extension(line["image_path"])

                # Add model name to Log
                log_entry = RecognitionLog(
                    image_path=line["image_path"],
                    predicted_label=line["predicted_label"],
                    confidence_score=line["confidence_score"],
                    manuscript_name=manuscript_name,
                    page=page_subfolder,
                    line=line["line"],
                    timestamp=datetime.now()
                )
                db.session.add(log_entry)
            db.session.commit()
            lines_of_all_pages[page_subfolder] = lines_of_one_page
        
        # clear GPU memory
        del lines_of_one_page
        torch.cuda.empty_cache()
        
        return lines_of_all_pages


    # In annotator/recognition/recognition.py (or wherever recognise_characters is)
    # Make sure to import: os, current_app, datetime, db, RecognitionLog, torch, get_filename_without_extension
    # And from annotator.recognition.demo import recognise_lines

    def recognise_single_page_characters(manuscript_folder_path, model_name, manuscript_name, page_to_process):
        """
        Recognises characters for a single specified page of a manuscript.
        Returns a dictionary of line data for that page.
        """
        lines_data_for_page = {}
        # Path to the specific page's line images
        page_lines_folder = os.path.join(manuscript_folder_path, "lines", page_to_process)

        if not os.path.isdir(page_lines_folder):
            current_app.logger.warning(f"Lines folder not found for page: {page_lines_folder}")
            return {} # Return empty if no lines found for the page

        # Recognise lines in the specified page_lines_folder
        # Note: ensure recognise_lines correctly processes images in this folder
        recognized_lines_list = recognise_lines(
            image_folder=page_lines_folder,
            saved_model=os.path.join(current_app.config['DATA_PATH'], 'models', 'recognition', model_name),
            transformation=None, # Add other params as they are in recognise_characters
            feature_extraction="ResNet",
            sequence_modeling="BiLSTM",
            prediction="CTC",
            workers=0,
            batch_max_length=250,
            imgH=50,
            imgW=2000,
            pad=True,
            character="""`0123456789~!@#$%^&*()-_+=[]\\{}|;':",./<>? abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.ँंःअअंअःआइईउऊऋएऐऑओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळवशषसह़ािीुूृॅेैॉोौ्ॐ॒क़ख़ग़ज़ड़ढ़फ़ॠ।०१२३४५६७८९॰""",
            hidden_size=512,
            output_channel=512,
        )

        for line_info in recognized_lines_list:
            # 'image_path' from recognise_lines is the full fs path to the line image.
            # We need the line name (filename without ext) for the store and URL construction.
            line_name = get_filename_without_extension(line_info["image_path"])

            lines_data_for_page[line_name] = {
                "predicted_label": line_info["predicted_label"],
                # This 'image_path' will be used by frontend to construct the request to /line-images endpoint.
                # So, it should be the line identifier (filename without extension).
                "image_path": line_name,
                "confidence_score": line_info["confidence_score"],
                # Optionally include these if frontend store needs them per line directly,
                # though typically they are known from manuscript_name and modelName in store.
                # "manuscript_name": manuscript_name,
                # "selected_model": model_name
            }

            # Log to DB
            log_entry = RecognitionLog(
                image_path=line_info["image_path"], # Log the actual filesystem path
                predicted_label=line_info["predicted_label"],
                confidence_score=line_info["confidence_score"],
                manuscript_name=manuscript_name,
                page=page_to_process,
                line=line_name, # Log the line name
                timestamp=datetime.now()
            )
            db.session.add(log_entry)
        
        db.session.commit()
        # GPU memory clear can be done after this call in the route handler
        return lines_data_for_page


    ================================================
    FILE: backend/annotator/recognition/test.py
    ================================================
    import os
    import time
    import string
    import argparse
    import re

    import torch
    import torch.backends.cudnn as cudnn
    import torch.utils.data
    import torch.nn.functional as F
    import numpy as np
    from nltk.metrics.distance import edit_distance

    from annotator.recognition.utils import CTCLabelConverter, AttnLabelConverter, Averager
    from annotator.recognition.dataset import hierarchical_dataset, AlignCollate
    from annotator.recognition.model import Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def benchmark_all_eval(model, criterion, converter, opt, calculate_infer_time=False):
        """ evaluation with 10 benchmark evaluation datasets """
        # The evaluation datasets, dataset order is same with Table 1 in our paper.
        eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867', 'IC13_857',
                        'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80']

        # # To easily compute the total accuracy of our paper.
        # eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_867', 
        #                   'IC13_1015', 'IC15_2077', 'SVTP', 'CUTE80']

        if calculate_infer_time:
            evaluation_batch_size = 1  # batch_size should be 1 to calculate the GPU inference time per image.
        else:
            evaluation_batch_size = opt.batch_size

        list_accuracy = []
        total_forward_time = 0
        total_evaluation_data_number = 0
        total_correct_number = 0
        log = open(f'./result/{opt.exp_name}/log_all_evaluation.txt', 'a')
        dashed_line = '-' * 80
        print(dashed_line)
        log.write(dashed_line + '\n')
        for eval_data in eval_data_list:
            eval_data_path = os.path.join(opt.eval_data, eval_data)
            AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
            eval_data, eval_data_log = hierarchical_dataset(root=eval_data_path, opt=opt)
            evaluation_loader = torch.utils.data.DataLoader(
                eval_data, batch_size=evaluation_batch_size,
                shuffle=False,
                num_workers=int(opt.workers),
                collate_fn=AlignCollate_evaluation, pin_memory=True)

            _, accuracy_by_best_model, norm_ED_by_best_model, _, _, _, infer_time, length_of_data = validation(
                model, criterion, evaluation_loader, converter, opt)
            list_accuracy.append(f'{accuracy_by_best_model:0.3f}')
            total_forward_time += infer_time
            total_evaluation_data_number += len(eval_data)
            total_correct_number += accuracy_by_best_model * length_of_data
            log.write(eval_data_log)
            print(f'Acc {accuracy_by_best_model:0.3f}\t normalized_ED {norm_ED_by_best_model:0.3f}')
            log.write(f'Acc {accuracy_by_best_model:0.3f}\t normalized_ED {norm_ED_by_best_model:0.3f}\n')
            print(dashed_line)
            log.write(dashed_line + '\n')

        averaged_forward_time = total_forward_time / total_evaluation_data_number * 1000
        total_accuracy = total_correct_number / total_evaluation_data_number
        params_num = sum([np.prod(p.size()) for p in model.parameters()])

        evaluation_log = 'accuracy: '
        for name, accuracy in zip(eval_data_list, list_accuracy):
            evaluation_log += f'{name}: {accuracy}\t'
        evaluation_log += f'total_accuracy: {total_accuracy:0.3f}\t'
        evaluation_log += f'averaged_infer_time: {averaged_forward_time:0.3f}\t# parameters: {params_num/1e6:0.3f}'
        print(evaluation_log)
        log.write(evaluation_log + '\n')
        log.close()

        return None


    def validation(model, criterion, evaluation_loader, converter, opt):
        """ validation or evaluation """
        n_correct = 0
        norm_ED = 0
        length_of_data = 0
        infer_time = 0
        valid_loss_avg = Averager()

        for i, (image_tensors, labels) in enumerate(evaluation_loader):
            batch_size = image_tensors.size(0)
            length_of_data = length_of_data + batch_size
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)

            start_time = time.time()
            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)
                forward_time = time.time() - start_time

                # Calculate evaluation loss for CTC deocder.
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                # permute 'preds' to use CTCloss format
                if opt.baiduCTC:
                    cost = criterion(preds.permute(1, 0, 2), text_for_loss, preds_size, length_for_loss) / batch_size
                else:
                    cost = criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)

                # Select max probabilty (greedy decoding) then decode index to character
                if opt.baiduCTC:
                    _, preds_index = preds.max(2)
                    preds_index = preds_index.view(-1)
                else:
                    _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index.data, preds_size.data)
            
            else:
                preds = model(image, text_for_pred, is_train=False)
                forward_time = time.time() - start_time

                preds = preds[:, :text_for_loss.shape[1] - 1, :]
                target = text_for_loss[:, 1:]  # without [GO] Symbol
                cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)
                labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

            infer_time += forward_time
            valid_loss_avg.add(cost)

            # calculate accuracy & confidence score
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            confidence_score_list = []
            for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    gt = gt[:gt.find('[s]')]
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
                if opt.sensitive and opt.data_filtering_off:
                    pred = pred.lower()
                    gt = gt.lower()
                    alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
                    out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
                    pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
                    gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)

                if pred == gt:
                    n_correct += 1

                '''
                (old version) ICDAR2017 DOST Normalized Edit Distance https://rrc.cvc.uab.es/?ch=7&com=tasks
                "For each word we calculate the normalized edit distance to the length of the ground truth transcription."
                if len(gt) == 0:
                    norm_ED += 1
                else:
                    norm_ED += edit_distance(pred, gt) / len(gt)
                '''

                # ICDAR2019 Normalized Edit Distance
                if len(gt) == 0 or len(pred) == 0:
                    norm_ED += 0
                elif len(gt) > len(pred):
                    norm_ED += 1 - edit_distance(pred, gt) / len(gt)
                else:
                    norm_ED += 1 - edit_distance(pred, gt) / len(pred)

                # calculate confidence score (= multiply of pred_max_prob)
                try:
                    confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                except:
                    confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
                confidence_score_list.append(confidence_score)
                # print(pred, gt, pred==gt, confidence_score)

        accuracy = n_correct / float(length_of_data) * 100
        norm_ED = norm_ED / float(length_of_data)  # ICDAR2019 Normalized Edit Distance

        return valid_loss_avg.val(), accuracy, norm_ED, preds_str, confidence_score_list, labels, infer_time, length_of_data


    def test(opt):
        """ model configuration """
        if 'CTC' in opt.Prediction:
            converter = CTCLabelConverter(opt.character)
        else:
            converter = AttnLabelConverter(opt.character)
        opt.num_class = len(converter.character)

        if opt.rgb:
            opt.input_channel = 3
        model = Model(opt)
        print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
            opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
            opt.SequenceModeling, opt.Prediction)
        model = torch.nn.DataParallel(model).to(device)

        # load model
        print('loading pretrained model from %s' % opt.saved_model)
        model.load_state_dict(torch.load(opt.saved_model, map_location=device))
        opt.exp_name = '_'.join(opt.saved_model.split('/')[1:])
        # print(model)

        """ keep evaluation model and result logs """
        os.makedirs(f'./result/{opt.exp_name}', exist_ok=True)
        os.system(f'cp {opt.saved_model} ./result/{opt.exp_name}/')

        """ setup loss """
        if 'CTC' in opt.Prediction:
            criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
        else:
            criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0

        """ evaluation """
        model.eval()
        with torch.no_grad():
            if opt.benchmark_all_eval:  # evaluation with 10 benchmark evaluation datasets
                benchmark_all_eval(model, criterion, converter, opt)
            else:
                log = open(f'./result/{opt.exp_name}/log_evaluation.txt', 'a')
                AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
                eval_data, eval_data_log = hierarchical_dataset(root=opt.eval_data, opt=opt)
                evaluation_loader = torch.utils.data.DataLoader(
                    eval_data, batch_size=opt.batch_size,
                    shuffle=False,
                    num_workers=int(opt.workers),
                    collate_fn=AlignCollate_evaluation, pin_memory=True)
                _, accuracy_by_best_model, _, _, _, _, _, _ = validation(
                    model, criterion, evaluation_loader, converter, opt)
                log.write(eval_data_log)
                print(f'{accuracy_by_best_model:0.3f}')
                log.write(f'{accuracy_by_best_model:0.3f}\n')
                log.close()


    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--eval_data', required=True, help='path to evaluation dataset')
        parser.add_argument('--benchmark_all_eval', action='store_true', help='evaluate 10 benchmark evaluation datasets')
        parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
        parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
        parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
        """ Data processing """
        parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
        parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
        parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
        parser.add_argument('--rgb', action='store_true', help='use rgb input')
        parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
        parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
        parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
        parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
        parser.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')
        """ Model Architecture """
        parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
        parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
        parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
        parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
        parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
        parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
        parser.add_argument('--output_channel', type=int, default=512,
                            help='the number of output channel of Feature extractor')
        parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

        opt = parser.parse_args()

        """ vocab / character number configuration """
        if opt.sensitive:
            opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

        cudnn.benchmark = True
        cudnn.deterministic = True
        opt.num_gpu = torch.cuda.device_count()

        test(opt)



    ================================================
    FILE: backend/annotator/recognition/train.py
    ================================================
    import os
    import sys
    import time
    import random
    import string
    import argparse

    import torch
    import torch.backends.cudnn as cudnn
    import torch.nn.init as init
    import torch.optim as optim
    import torch.utils.data
    import numpy as np

    from annotator.recognition.utils import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter, Averager
    from annotator.recognition.dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
    from annotator.recognition.model import Model
    from annotator.recognition.test import validation

    if torch.cuda.is_available():
        device = torch.device('cuda')
        map_location=torch.device('cuda')
    else:
        device = torch.device('cpu')
        map_location=torch.device('cpu')

    def train(opt):
        """ dataset preparation """
        if not opt.data_filtering_off:
            print('Filtering the images containing characters which are not in opt.character')
            print('Filtering the images whose label is longer than opt.batch_max_length')
            # see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L130

        opt.select_data = opt.select_data.split('-')
        opt.batch_ratio = opt.batch_ratio.split('-')
        train_dataset = Batch_Balanced_Dataset(opt)

        log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')
        AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=opt.batch_size,
            shuffle=True,  # 'True' to check training progress with validation function.
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_valid, pin_memory=True)
        log.write(valid_dataset_log)
        print('-' * 80)
        log.write('-' * 80 + '\n')
        log.close()
        
        """ model configuration """
        if 'CTC' in opt.Prediction:
            if opt.baiduCTC:
                converter = CTCLabelConverterForBaiduWarpctc(opt.character)
            else:
                converter = CTCLabelConverter(opt.character)
        else:
            converter = AttnLabelConverter(opt.character)
        opt.num_class = len(converter.character)

        if opt.rgb:
            opt.input_channel = 3
        model = Model(opt)
        print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
            opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
            opt.SequenceModeling, opt.Prediction)

        # weight initialization
        for name, param in model.named_parameters():
            if 'localization_fc2' in name:
                print(f'Skip {name} as it is already initialized')
                continue
            try:
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            except Exception as e:  # for batchnorm.
                if 'weight' in name:
                    param.data.fill_(1)
                continue

        # data parallel for multi-GPU
        model = torch.nn.DataParallel(model).to(device)
        model.train()
        if opt.saved_model != '':
            print(f'loading pretrained model from {opt.saved_model}')
            if opt.FT:
                model.module.load_state_dict(torch.load(opt.saved_model, map_location = map_location), strict=False)
            else:
                model.load_state_dict(torch.load(opt.saved_model, map_location = map_location))
        print("Model:")
        print(model)

        """ setup loss """
        if 'CTC' in opt.Prediction:
            if opt.baiduCTC:
                # need to install warpctc. see our guideline.
                from warpctc_pytorch import CTCLoss 
                criterion = CTCLoss()
            else:
                criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
        else:
            criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
        # loss averager
        loss_avg = Averager()

        # filter that only require gradient decent
        filtered_parameters = []
        params_num = []
        for p in filter(lambda p: p.requires_grad, model.parameters()):
            filtered_parameters.append(p)
            params_num.append(np.prod(p.size()))
        print('Trainable params num : ', sum(params_num))
        # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

        # setup optimizer
        if opt.adam:
            optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
        else:
            optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
        print("Optimizer:")
        print(optimizer)

        """ final options """
        # print(opt)
        with open(f'./saved_models/{opt.exp_name}/opt.txt', 'a') as opt_file:
            opt_log = '------------ Options -------------\n'
            args = vars(opt)
            for k, v in args.items():
                opt_log += f'{str(k)}: {str(v)}\n'
            opt_log += '---------------------------------------\n'
            print(opt_log)
            opt_file.write(opt_log)

        """ start training """
        start_iter = 0
        if opt.saved_model != '':
            try:
                start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
                print(f'continue to train, start_iter: {start_iter}')
            except:
                pass

        start_time = time.time()
        best_accuracy = -1
        best_norm_ED = -1
        iteration = start_iter

        while(True):
            # train part
            image_tensors, labels = train_dataset.get_batch()
            image = image_tensors.to(device)
            text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
            batch_size = image.size(0)

            if 'CTC' in opt.Prediction:
                preds = model(image, text)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                if opt.baiduCTC:
                    preds = preds.permute(1, 0, 2)  # to use CTCLoss format
                    cost = criterion(preds, text, preds_size, length) / batch_size
                else:
                    preds = preds.log_softmax(2).permute(1, 0, 2)
                    cost = criterion(preds, text, preds_size, length)

            else:
                preds = model(image, text[:, :-1])  # align with Attention.forward
                target = text[:, 1:]  # without [GO] Symbol
                cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

            model.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
            optimizer.step()

            loss_avg.add(cost)

            # validation part
            if (iteration + 1) % opt.valInterval == 0 or iteration == 0: # To see training progress, we also conduct validation when 'iteration == 0' 
                elapsed_time = time.time() - start_time
                # for log
                with open(f'./saved_models/{opt.exp_name}/log_train.txt', 'a') as log:
                    model.eval()
                    with torch.no_grad():
                        valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, infer_time, length_of_data = validation(
                            model, criterion, valid_loader, converter, opt)
                    model.train()

                    # training loss and validation loss
                    loss_log = f'[{iteration+1}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                    loss_avg.reset()

                    current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.2f}'

                    # keep best accuracy model (on valid dataset)
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                        torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_accuracy.pth')
                    if current_norm_ED > best_norm_ED:
                        best_norm_ED = current_norm_ED
                        torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_norm_ED.pth')
                    best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.2f}'

                    loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                    print(loss_model_log)
                    log.write(loss_model_log + '\n')

                    # show some predicted results
                    dashed_line = '-' * 80
                    head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                    predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                    for gt, pred, confidence in zip(labels[:5], preds[:5], confidence_score[:5]):
                        if 'Attn' in opt.Prediction:
                            gt = gt[:gt.find('[s]')]
                            pred = pred[:pred.find('[s]')]

                        predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                    predicted_result_log += f'{dashed_line}'
                    print(predicted_result_log)
                    log.write(predicted_result_log + '\n')

            # save model per 1e+5 iter.
            if (iteration + 1) % 1e+5 == 0:
                torch.save(
                    model.state_dict(), f'./saved_models/{opt.exp_name}/iter_{iteration+1}.pth')

            if (iteration + 1) == opt.num_iter:
                print('end the training')
                sys.exit()
            iteration += 1


    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--exp_name', help='Where to store logs and models')
        parser.add_argument('--train_data', required=True, help='path to training dataset')
        parser.add_argument('--valid_data', required=True, help='path to validation dataset')
        parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
        parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
        parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
        parser.add_argument('--num_iter', type=int, default=300000, help='number of iterations to train for')
        parser.add_argument('--valInterval', type=int, default=2000, help='Interval between each validation')
        parser.add_argument('--saved_model', default='', help="path to model to continue training")
        parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
        parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
        parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
        parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
        parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
        parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
        parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
        parser.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')
        """ Data processing """
        parser.add_argument('--select_data', type=str, default='/', 
                        help='select training data (default is MJ-ST, which means MJ and ST used as training data)') 
        parser.add_argument('--batch_ratio', type=str, default='1', 
                        help='assign ratio for each selected data in the batch') 
        # parser.add_argument('--select_data', type=str, default='MJ-ST',
        #                     help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
        # parser.add_argument('--batch_ratio', type=str, default='0.5-0.5',
        #                     help='assign ratio for each selected data in the batch')
        parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                            help='total data usage ratio, this ratio is multiplied to total number of data.')
        parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
        parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
        parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
        parser.add_argument('--rgb', action='store_true', help='use rgb input')
        parser.add_argument('--character', type=str,
                            default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
        parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
        parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
        parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
        """ Model Architecture """
        parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
        parser.add_argument('--FeatureExtraction', type=str, required=True,
                            help='FeatureExtraction stage. VGG|RCNN|ResNet')
        parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
        parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
        parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
        parser.add_argument('--input_channel', type=int, default=1,
                            help='the number of input channel of Feature extractor')
        parser.add_argument('--output_channel', type=int, default=512,
                            help='the number of output channel of Feature extractor')
        parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

        opt = parser.parse_args()
        # opt.character = '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.ँंःअअंअःआइईउऊऋएऐऑओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळवशषसह़ािीुूृॅेैॉोौ्ॐ॒क़ख़ग़ज़ड़ढ़फ़ॠ।०१२३४५६७८९॰'  

        if not opt.exp_name:
            opt.exp_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
            opt.exp_name += f'-Seed{opt.manualSeed}'
            # print(opt.exp_name)

        os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)

        """ vocab / character number configuration """
        if opt.sensitive:
            # opt.character += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

        """ Seed and GPU setting """
        # print("Random Seed: ", opt.manualSeed)
        random.seed(opt.manualSeed)
        np.random.seed(opt.manualSeed)
        torch.manual_seed(opt.manualSeed)
        torch.cuda.manual_seed(opt.manualSeed)

        cudnn.benchmark = True
        cudnn.deterministic = True
        opt.num_gpu = torch.cuda.device_count()
        # print('device count', opt.num_gpu)
        if opt.num_gpu > 1:
            print('------ Use multi-GPU setting ------')
            print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
            # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
            opt.workers = opt.workers * opt.num_gpu
            opt.batch_size = opt.batch_size * opt.num_gpu

            """ previous version
            print('To equlize batch stats to 1-GPU setting, the batch_size is multiplied with num_gpu and multiplied batch_size is ', opt.batch_size)
            opt.batch_size = opt.batch_size * opt.num_gpu
            print('To equalize the number of epochs to 1-GPU setting, num_iter is divided with num_gpu by default.')
            If you dont care about it, just commnet out these line.)
            opt.num_iter = int(opt.num_iter / opt.num_gpu)
            """

        train(opt)



    ================================================
    FILE: backend/annotator/recognition/utils.py
    ================================================
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    class CTCLabelConverter(object):
        """ Convert between text-label and text-index """

        def __init__(self, character):
            # character (str): set of the possible characters.
            dict_character = list(character)

            self.dict = {}
            for i, char in enumerate(dict_character):
                # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
                self.dict[char] = i + 1

            self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

        def encode(self, text, batch_max_length=25):
            """convert text-label into text-index.
            input:
                text: text labels of each image. [batch_size]
                batch_max_length: max length of text label in the batch. 25 by default

            output:
                text: text index for CTCLoss. [batch_size, batch_max_length]
                length: length of each text. [batch_size]
            """
            length = [len(s) for s in text]

            # The index used for padding (=0) would not affect the CTC loss calculation.
            batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
            for i, t in enumerate(text):
                text = list(t)
                text = [self.dict[char] for char in text]
                batch_text[i][:len(text)] = torch.LongTensor(text)
            return (batch_text.to(device), torch.IntTensor(length).to(device))

        def decode(self, text_index, length):
            """ convert text-index into text-label. """
            texts = []
            for index, l in enumerate(length):
                t = text_index[index, :]
                char_list = []
                for i in range(l):
                    if t[i]!=0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
                text = ''.join(list(filter((" ").__ne__, char_list)))
                texts.append(text)
            return texts


    class CTCLabelConverterForBaiduWarpctc(object):
        """ Convert between text-label and text-index for baidu warpctc """

        def __init__(self, character):
            # character (str): set of the possible characters.
            dict_character = list(character)

            self.dict = {}
            for i, char in enumerate(dict_character):
                # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
                self.dict[char] = i + 1

            self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

        def encode(self, text, batch_max_length=25):
            """convert text-label into text-index.
            input:
                text: text labels of each image. [batch_size]
            output:
                text: concatenated text index for CTCLoss.
                        [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
                length: length of each text. [batch_size]
            """
            length = [len(s) for s in text]
            text = ''.join(text)
            text = [self.dict[char] for char in text]

            return (torch.IntTensor(text), torch.IntTensor(length))

        def decode(self, text_index, length):
            """ convert text-index into text-label. """
            texts = []
            index = 0
            for l in length:
                t = text_index[index:index + l]

                char_list = []
                for i in range(l):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                        char_list.append(self.character[t[i]])
                text = ''.join(char_list)

                texts.append(text)
                index += l
            return texts


    class AttnLabelConverter(object):
        """ Convert between text-label and text-index """

        def __init__(self, character):
            # character (str): set of the possible characters.
            # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
            list_token = ['[GO]', '[s]', '[UNK]']  # ['[s]','[UNK]','[PAD]','[GO]']
            list_character = list(character)
            self.character = list_token + list_character

            self.dict = {}
            for i, char in enumerate(self.character):
                # print(i, char)
                self.dict[char] = i

        def encode(self, text, batch_max_length=25):
            """ convert text-label into text-index.
            input:
                text: text labels of each image. [batch_size]
                batch_max_length: max length of text label in the batch. 25 by default

            output:
                text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                    text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
                length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
            """
            length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
            # batch_max_length = max(length) # this is not allowed for multi-gpu setting
            batch_max_length += 1
            # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
            batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
            for i, t in enumerate(text):
                text = list(t)
                text.append('[s]')
                text = [self.dict[char] for char in text]
                batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
            return (batch_text.to(device), torch.IntTensor(length).to(device))

        def decode(self, text_index, length):
            """ convert text-index into text-label. """
            texts = []
            for index, l in enumerate(length):
                text = ''.join([self.character[i] for i in text_index[index, :]])
                texts.append(text)
            return texts


    class Averager(object):
        """Compute average for torch.Tensor, used for loss average."""

        def __init__(self):
            self.reset()

        def add(self, v):
            count = v.data.numel()
            v = v.data.sum()
            self.n_count += count
            self.sum += v

        def reset(self):
            self.n_count = 0
            self.sum = 0

        def val(self):
            res = 0
            if self.n_count != 0:
                res = self.sum / float(self.n_count)
            return res



    ================================================
    FILE: backend/annotator/recognition/modules/feature_extraction.py
    ================================================
    import torch.nn as nn
    import torch.nn.functional as F


    class VGG_FeatureExtractor(nn.Module):
        """ FeatureExtractor of CRNN (https://arxiv.org/pdf/1507.05717.pdf) """

        def __init__(self, input_channel, output_channel=512):
            super(VGG_FeatureExtractor, self).__init__()
            self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                                int(output_channel / 2), output_channel]  # [64, 128, 256, 512]
            self.ConvNet = nn.Sequential(
                nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
                nn.MaxPool2d(2, 2),  # 64x16x50
                nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
                nn.MaxPool2d(2, 2),  # 128x8x25
                nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True),  # 256x8x25
                nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
                nn.MaxPool2d((2, 1), (2, 1)),  # 256x4x25
                nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),  # 512x4x25
                nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
                nn.MaxPool2d((2, 1), (2, 1)),  # 512x2x25
                nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0), nn.ReLU(True))  # 512x1x24

        def forward(self, input):
            return self.ConvNet(input)


    class RCNN_FeatureExtractor(nn.Module):
        """ FeatureExtractor of GRCNN (https://papers.nips.cc/paper/6637-gated-recurrent-convolution-neural-network-for-ocr.pdf) """

        def __init__(self, input_channel, output_channel=512):
            super(RCNN_FeatureExtractor, self).__init__()
            self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                                int(output_channel / 2), output_channel]  # [64, 128, 256, 512]
            self.ConvNet = nn.Sequential(
                nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
                nn.MaxPool2d(2, 2),  # 64 x 16 x 50
                GRCL(self.output_channel[0], self.output_channel[0], num_iteration=5, kernel_size=3, pad=1),
                nn.MaxPool2d(2, 2),  # 64 x 8 x 25
                GRCL(self.output_channel[0], self.output_channel[1], num_iteration=5, kernel_size=3, pad=1),
                nn.MaxPool2d(2, (2, 1), (0, 1)),  # 128 x 4 x 26
                GRCL(self.output_channel[1], self.output_channel[2], num_iteration=5, kernel_size=3, pad=1),
                nn.MaxPool2d(2, (2, 1), (0, 1)),  # 256 x 2 x 27
                nn.Conv2d(self.output_channel[2], self.output_channel[3], 2, 1, 0, bias=False),
                nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True))  # 512 x 1 x 26

        def forward(self, input):
            return self.ConvNet(input)


    class ResNet_FeatureExtractor(nn.Module):
        """ FeatureExtractor of FAN (http://openaccess.thecvf.com/content_ICCV_2017/papers/Cheng_Focusing_Attention_Towards_ICCV_2017_paper.pdf) """

        def __init__(self, input_channel, output_channel=512):
            super(ResNet_FeatureExtractor, self).__init__()
            self.ConvNet = ResNet(input_channel, output_channel, BasicBlock, [1, 2, 5, 3])

        def forward(self, input):
            return self.ConvNet(input)


    # For Gated RCNN
    class GRCL(nn.Module):

        def __init__(self, input_channel, output_channel, num_iteration, kernel_size, pad):
            super(GRCL, self).__init__()
            self.wgf_u = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False)
            self.wgr_x = nn.Conv2d(output_channel, output_channel, 1, 1, 0, bias=False)
            self.wf_u = nn.Conv2d(input_channel, output_channel, kernel_size, 1, pad, bias=False)
            self.wr_x = nn.Conv2d(output_channel, output_channel, kernel_size, 1, pad, bias=False)

            self.BN_x_init = nn.BatchNorm2d(output_channel)

            self.num_iteration = num_iteration
            self.GRCL = [GRCL_unit(output_channel) for _ in range(num_iteration)]
            self.GRCL = nn.Sequential(*self.GRCL)

        def forward(self, input):
            """ The input of GRCL is consistant over time t, which is denoted by u(0)
            thus wgf_u / wf_u is also consistant over time t.
            """
            wgf_u = self.wgf_u(input)
            wf_u = self.wf_u(input)
            x = F.relu(self.BN_x_init(wf_u))

            for i in range(self.num_iteration):
                x = self.GRCL[i](wgf_u, self.wgr_x(x), wf_u, self.wr_x(x))

            return x


    class GRCL_unit(nn.Module):

        def __init__(self, output_channel):
            super(GRCL_unit, self).__init__()
            self.BN_gfu = nn.BatchNorm2d(output_channel)
            self.BN_grx = nn.BatchNorm2d(output_channel)
            self.BN_fu = nn.BatchNorm2d(output_channel)
            self.BN_rx = nn.BatchNorm2d(output_channel)
            self.BN_Gx = nn.BatchNorm2d(output_channel)

        def forward(self, wgf_u, wgr_x, wf_u, wr_x):
            G_first_term = self.BN_gfu(wgf_u)
            G_second_term = self.BN_grx(wgr_x)
            G = F.sigmoid(G_first_term + G_second_term)

            x_first_term = self.BN_fu(wf_u)
            x_second_term = self.BN_Gx(self.BN_rx(wr_x) * G)
            x = F.relu(x_first_term + x_second_term)

            return x


    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, inplanes, planes, stride=1, downsample=None):
            super(BasicBlock, self).__init__()
            self.conv1 = self._conv3x3(inplanes, planes)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = self._conv3x3(planes, planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride

        def _conv3x3(self, in_planes, out_planes, stride=1):
            "3x3 convolution with padding"
            return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                            padding=1, bias=False)

        def forward(self, x):
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
            out = self.relu(out)

            return out


    class ResNet(nn.Module):

        def __init__(self, input_channel, output_channel, block, layers):
            super(ResNet, self).__init__()

            self.output_channel_block = [int(output_channel / 4), int(output_channel / 2), output_channel, output_channel]

            self.inplanes = int(output_channel / 8)
            self.conv0_1 = nn.Conv2d(input_channel, int(output_channel / 16),
                                    kernel_size=3, stride=1, padding=1, bias=False)
            self.bn0_1 = nn.BatchNorm2d(int(output_channel / 16))
            self.conv0_2 = nn.Conv2d(int(output_channel / 16), self.inplanes,
                                    kernel_size=3, stride=1, padding=1, bias=False)
            self.bn0_2 = nn.BatchNorm2d(self.inplanes)
            self.relu = nn.ReLU(inplace=True)

            self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
            self.conv1 = nn.Conv2d(self.output_channel_block[0], self.output_channel_block[
                                0], kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])

            self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.layer2 = self._make_layer(block, self.output_channel_block[1], layers[1], stride=1)
            self.conv2 = nn.Conv2d(self.output_channel_block[1], self.output_channel_block[
                                1], kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(self.output_channel_block[1])

            self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
            self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
            self.conv3 = nn.Conv2d(self.output_channel_block[2], self.output_channel_block[
                                2], kernel_size=3, stride=1, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])

            self.layer4 = self._make_layer(block, self.output_channel_block[3], layers[3], stride=1)
            self.conv4_1 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[
                                    3], kernel_size=2, stride=(2, 1), padding=(0, 1), bias=False)
            self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
            self.conv4_2 = nn.Conv2d(self.output_channel_block[3], self.output_channel_block[
                                    3], kernel_size=2, stride=1, padding=0, bias=False)
            self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])

        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv0_1(x)
            x = self.bn0_1(x)
            x = self.relu(x)
            x = self.conv0_2(x)
            x = self.bn0_2(x)
            x = self.relu(x)

            x = self.maxpool1(x)
            x = self.layer1(x)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.maxpool2(x)
            x = self.layer2(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)

            x = self.maxpool3(x)
            x = self.layer3(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)

            x = self.layer4(x)
            x = self.conv4_1(x)
            x = self.bn4_1(x)
            x = self.relu(x)
            x = self.conv4_2(x)
            x = self.bn4_2(x)
            x = self.relu(x)

            return x



    ================================================
    FILE: backend/annotator/recognition/modules/prediction.py
    ================================================
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    class Attention(nn.Module):

        def __init__(self, input_size, hidden_size, num_classes):
            super(Attention, self).__init__()
            self.attention_cell = AttentionCell(input_size, hidden_size, num_classes)
            self.hidden_size = hidden_size
            self.num_classes = num_classes
            self.generator = nn.Linear(hidden_size, num_classes)

        def _char_to_onehot(self, input_char, onehot_dim=38):
            input_char = input_char.unsqueeze(1)
            batch_size = input_char.size(0)
            one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(device)
            one_hot = one_hot.scatter_(1, input_char, 1)
            return one_hot

        def forward(self, batch_H, text, is_train=True, batch_max_length=25):
            """
            input:
                batch_H : contextual_feature H = hidden state of encoder. [batch_size x num_steps x contextual_feature_channels]
                text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [GO] token. text[:, 0] = [GO].
            output: probability distribution at each step [batch_size x num_steps x num_classes]
            """
            batch_size = batch_H.size(0)
            num_steps = batch_max_length + 1  # +1 for [s] at end of sentence.

            output_hiddens = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0).to(device)
            hidden = (torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device),
                    torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device))

            if is_train:
                for i in range(num_steps):
                    # one-hot vectors for a i-th char. in a batch
                    char_onehots = self._char_to_onehot(text[:, i], onehot_dim=self.num_classes)
                    # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                    hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                    output_hiddens[:, i, :] = hidden[0]  # LSTM hidden index (0: hidden, 1: Cell)
                probs = self.generator(output_hiddens)

            else:
                targets = torch.LongTensor(batch_size).fill_(0).to(device)  # [GO] token
                probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0).to(device)

                for i in range(num_steps):
                    char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                    hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                    probs_step = self.generator(hidden[0])
                    probs[:, i, :] = probs_step
                    _, next_input = probs_step.max(1)
                    targets = next_input

            return probs  # batch_size x num_steps x num_classes


    class AttentionCell(nn.Module):

        def __init__(self, input_size, hidden_size, num_embeddings):
            super(AttentionCell, self).__init__()
            self.i2h = nn.Linear(input_size, hidden_size, bias=False)
            self.h2h = nn.Linear(hidden_size, hidden_size)  # either i2i or h2h should have bias
            self.score = nn.Linear(hidden_size, 1, bias=False)
            self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
            self.hidden_size = hidden_size

        def forward(self, prev_hidden, batch_H, char_onehots):
            # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
            batch_H_proj = self.i2h(batch_H)
            prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
            e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj))  # batch_size x num_encoder_step * 1

            alpha = F.softmax(e, dim=1)
            context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)  # batch_size x num_channel
            concat_context = torch.cat([context, char_onehots], 1)  # batch_size x (num_channel + num_embedding)
            cur_hidden = self.rnn(concat_context, prev_hidden)
            return cur_hidden, alpha



    ================================================
    FILE: backend/annotator/recognition/modules/sequence_modeling.py
    ================================================
    import torch.nn as nn


    class BidirectionalLSTM(nn.Module):

        def __init__(self, input_size, hidden_size, output_size):
            super(BidirectionalLSTM, self).__init__()
            self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
            self.linear = nn.Linear(hidden_size * 2, output_size)

        def forward(self, input):
            """
            input : visual feature [batch_size x T x input_size]
            output : contextual feature [batch_size x T x output_size]
            """
            self.rnn.flatten_parameters()
            recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
            output = self.linear(recurrent)  # batch_size x T x output_size
            return output



    ================================================
    FILE: backend/annotator/recognition/modules/transformation.py
    ================================================
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    class TPS_SpatialTransformerNetwork(nn.Module):
        """ Rectification Network of RARE, namely TPS based STN """

        def __init__(self, F, I_size, I_r_size, I_channel_num=1):
            """ Based on RARE TPS
            input:
                batch_I: Batch Input Image [batch_size x I_channel_num x I_height x I_width]
                I_size : (height, width) of the input image I
                I_r_size : (height, width) of the rectified image I_r
                I_channel_num : the number of channels of the input image I
            output:
                batch_I_r: rectified image [batch_size x I_channel_num x I_r_height x I_r_width]
            """
            super(TPS_SpatialTransformerNetwork, self).__init__()
            self.F = F
            self.I_size = I_size
            self.I_r_size = I_r_size  # = (I_r_height, I_r_width)
            self.I_channel_num = I_channel_num
            self.LocalizationNetwork = LocalizationNetwork(self.F, self.I_channel_num)
            self.GridGenerator = GridGenerator(self.F, self.I_r_size)

        def forward(self, batch_I):
            batch_C_prime = self.LocalizationNetwork(batch_I)  # batch_size x K x 2
            build_P_prime = self.GridGenerator.build_P_prime(batch_C_prime)  # batch_size x n (= I_r_width x I_r_height) x 2
            build_P_prime_reshape = build_P_prime.reshape([build_P_prime.size(0), self.I_r_size[0], self.I_r_size[1], 2])
            
            if torch.__version__ > "1.2.0":
                batch_I_r = F.grid_sample(batch_I, build_P_prime_reshape, padding_mode='border', align_corners=True)
            else:
                batch_I_r = F.grid_sample(batch_I, build_P_prime_reshape, padding_mode='border')

            return batch_I_r


    class LocalizationNetwork(nn.Module):
        """ Localization Network of RARE, which predicts C' (K x 2) from I (I_width x I_height) """

        def __init__(self, F, I_channel_num):
            super(LocalizationNetwork, self).__init__()
            self.F = F
            self.I_channel_num = I_channel_num
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=self.I_channel_num, out_channels=64, kernel_size=3, stride=1, padding=1,
                        bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
                nn.MaxPool2d(2, 2),  # batch_size x 64 x I_height/2 x I_width/2
                nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
                nn.MaxPool2d(2, 2),  # batch_size x 128 x I_height/4 x I_width/4
                nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
                nn.MaxPool2d(2, 2),  # batch_size x 256 x I_height/8 x I_width/8
                nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),
                nn.AdaptiveAvgPool2d(1)  # batch_size x 512
            )

            self.localization_fc1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True))
            self.localization_fc2 = nn.Linear(256, self.F * 2)

            # Init fc2 in LocalizationNetwork
            self.localization_fc2.weight.data.fill_(0)
            """ see RARE paper Fig. 6 (a) """
            ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
            ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
            ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
            ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
            ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
            initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
            self.localization_fc2.bias.data = torch.from_numpy(initial_bias).float().view(-1)

        def forward(self, batch_I):
            """
            input:     batch_I : Batch Input Image [batch_size x I_channel_num x I_height x I_width]
            output:    batch_C_prime : Predicted coordinates of fiducial points for input batch [batch_size x F x 2]
            """
            batch_size = batch_I.size(0)
            features = self.conv(batch_I).view(batch_size, -1)
            batch_C_prime = self.localization_fc2(self.localization_fc1(features)).view(batch_size, self.F, 2)
            return batch_C_prime


    class GridGenerator(nn.Module):
        """ Grid Generator of RARE, which produces P_prime by multipling T with P """

        def __init__(self, F, I_r_size):
            """ Generate P_hat and inv_delta_C for later """
            super(GridGenerator, self).__init__()
            self.eps = 1e-6
            self.I_r_height, self.I_r_width = I_r_size
            self.F = F
            self.C = self._build_C(self.F)  # F x 2
            self.P = self._build_P(self.I_r_width, self.I_r_height)
            ## for multi-gpu, you need register buffer
            self.register_buffer("inv_delta_C", torch.tensor(self._build_inv_delta_C(self.F, self.C)).float())  # F+3 x F+3
            self.register_buffer("P_hat", torch.tensor(self._build_P_hat(self.F, self.C, self.P)).float())  # n x F+3
            ## for fine-tuning with different image width, you may use below instead of self.register_buffer
            #self.inv_delta_C = torch.tensor(self._build_inv_delta_C(self.F, self.C)).float().cuda()  # F+3 x F+3
            #self.P_hat = torch.tensor(self._build_P_hat(self.F, self.C, self.P)).float().cuda()  # n x F+3

        def _build_C(self, F):
            """ Return coordinates of fiducial points in I_r; C """
            ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
            ctrl_pts_y_top = -1 * np.ones(int(F / 2))
            ctrl_pts_y_bottom = np.ones(int(F / 2))
            ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
            ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
            C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
            return C  # F x 2

        def _build_inv_delta_C(self, F, C):
            """ Return inv_delta_C which is needed to calculate T """
            hat_C = np.zeros((F, F), dtype=float)  # F x F
            for i in range(0, F):
                for j in range(i, F):
                    r = np.linalg.norm(C[i] - C[j])
                    hat_C[i, j] = r
                    hat_C[j, i] = r
            np.fill_diagonal(hat_C, 1)
            hat_C = (hat_C ** 2) * np.log(hat_C)
            # print(C.shape, hat_C.shape)
            delta_C = np.concatenate(  # F+3 x F+3
                [
                    np.concatenate([np.ones((F, 1)), C, hat_C], axis=1),  # F x F+3
                    np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1),  # 2 x F+3
                    np.concatenate([np.zeros((1, 3)), np.ones((1, F))], axis=1)  # 1 x F+3
                ],
                axis=0
            )
            inv_delta_C = np.linalg.inv(delta_C)
            return inv_delta_C  # F+3 x F+3

        def _build_P(self, I_r_width, I_r_height):
            I_r_grid_x = (np.arange(-I_r_width, I_r_width, 2) + 1.0) / I_r_width  # self.I_r_width
            I_r_grid_y = (np.arange(-I_r_height, I_r_height, 2) + 1.0) / I_r_height  # self.I_r_height
            P = np.stack(  # self.I_r_width x self.I_r_height x 2
                np.meshgrid(I_r_grid_x, I_r_grid_y),
                axis=2
            )
            return P.reshape([-1, 2])  # n (= self.I_r_width x self.I_r_height) x 2

        def _build_P_hat(self, F, C, P):
            n = P.shape[0]  # n (= self.I_r_width x self.I_r_height)
            P_tile = np.tile(np.expand_dims(P, axis=1), (1, F, 1))  # n x 2 -> n x 1 x 2 -> n x F x 2
            C_tile = np.expand_dims(C, axis=0)  # 1 x F x 2
            P_diff = P_tile - C_tile  # n x F x 2
            rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)  # n x F
            rbf = np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))  # n x F
            P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
            return P_hat  # n x F+3

        def build_P_prime(self, batch_C_prime):
            """ Generate Grid from batch_C_prime [batch_size x F x 2] """
            batch_size = batch_C_prime.size(0)
            batch_inv_delta_C = self.inv_delta_C.repeat(batch_size, 1, 1)
            batch_P_hat = self.P_hat.repeat(batch_size, 1, 1)
            batch_C_prime_with_zeros = torch.cat((batch_C_prime, torch.zeros(
                batch_size, 3, 2).float().to(device)), dim=1)  # batch_size x F+3 x 2
            batch_T = torch.bmm(batch_inv_delta_C, batch_C_prime_with_zeros)  # batch_size x F+3 x 2
            batch_P_prime = torch.bmm(batch_P_hat, batch_T)  # batch_size x n x 2
            return batch_P_prime  # batch_size x n x 2



    ================================================
    FILE: backend/annotator/segmentation/craft.py
    ================================================
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torch.nn as nn
    import torch.nn.init as init
    import torchvision
    from torchvision import models
    # import matplotlib.pyplot as plt
    from collections import namedtuple
    from packaging import version
    from collections import OrderedDict


    # #GLOBAL VARIABLES
    # lineheight_baseline_percentile = None
    # binarize_threshold = None




    def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
        # should be RGB order
        img = in_img.copy().astype(np.float32)

        img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
        img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
        return img


    def init_weights(modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    class vgg16_bn(torch.nn.Module):
        def __init__(self, pretrained=True, freeze=True):
            super(vgg16_bn, self).__init__()
            if version.parse(torchvision.__version__) >= version.parse('0.13'):
                vgg_pretrained_features = models.vgg16_bn(
                    weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None
                ).features
            else: #torchvision.__version__ < 0.13
                models.vgg.model_urls['vgg16_bn'] = models.vgg.model_urls['vgg16_bn'].replace('https://', 'http://')
                vgg_pretrained_features = models.vgg16_bn(pretrained=pretrained).features

            self.slice1 = torch.nn.Sequential()
            self.slice2 = torch.nn.Sequential()
            self.slice3 = torch.nn.Sequential()
            self.slice4 = torch.nn.Sequential()
            self.slice5 = torch.nn.Sequential()
            for x in range(12):         # conv2_2
                self.slice1.add_module(str(x), vgg_pretrained_features[x])
            for x in range(12, 19):         # conv3_3
                self.slice2.add_module(str(x), vgg_pretrained_features[x])
            for x in range(19, 29):         # conv4_3
                self.slice3.add_module(str(x), vgg_pretrained_features[x])
            for x in range(29, 39):         # conv5_3
                self.slice4.add_module(str(x), vgg_pretrained_features[x])
                
            # fc6, fc7 without atrous conv
            self.slice5 = torch.nn.Sequential(
                    nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
                    nn.Conv2d(1024, 1024, kernel_size=1)
            )

            if not pretrained:
                init_weights(self.slice1.modules())
                init_weights(self.slice2.modules())
                init_weights(self.slice3.modules())
                init_weights(self.slice4.modules())

            init_weights(self.slice5.modules())        # no pretrained model for fc6 and fc7

            if freeze:
                for param in self.slice1.parameters():      # only first conv
                    param.requires_grad= False

        def forward(self, X):
            h = self.slice1(X)
            h_relu2_2 = h
            h = self.slice2(h)
            h_relu3_2 = h
            h = self.slice3(h)
            h_relu4_3 = h
            h = self.slice4(h)
            h_relu5_3 = h
            h = self.slice5(h)
            h_fc7 = h
            vgg_outputs = namedtuple("VggOutputs", ['fc7', 'relu5_3', 'relu4_3', 'relu3_2', 'relu2_2'])
            out = vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
            return out

    class BidirectionalLSTM(nn.Module):

        def __init__(self, input_size, hidden_size, output_size):
            super(BidirectionalLSTM, self).__init__()
            self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
            self.linear = nn.Linear(hidden_size * 2, output_size)

        def forward(self, input):
            """
            input : visual feature [batch_size x T x input_size]
            output : contextual feature [batch_size x T x output_size]
            """
            try: # multi gpu needs this
                self.rnn.flatten_parameters()
            except: # quantization doesn't work with this
                pass
            recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
            output = self.linear(recurrent)  # batch_size x T x output_size
            return output

    class VGG_FeatureExtractor(nn.Module):

        def __init__(self, input_channel, output_channel=256):
            super(VGG_FeatureExtractor, self).__init__()
            self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                                int(output_channel / 2), output_channel]
            self.ConvNet = nn.Sequential(
                nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
                nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
                nn.MaxPool2d((2, 1), (2, 1)),
                nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
                nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
                nn.MaxPool2d((2, 1), (2, 1)),
                nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0), nn.ReLU(True))

        def forward(self, input):
            return self.ConvNet(input)



    class Model(nn.Module):
        def __init__(self, input_channel, output_channel, hidden_size, num_class):
            super(Model, self).__init__()
            """ FeatureExtraction """
            self.FeatureExtraction = VGG_FeatureExtractor(input_channel, output_channel)
            self.FeatureExtraction_output = output_channel
            self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

            """ Sequence modeling"""
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
                BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
            self.SequenceModeling_output = hidden_size

            """ Prediction """
            self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)


        def forward(self, input, text):
            """ Feature extraction stage """
            visual_feature = self.FeatureExtraction(input)
            visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
            visual_feature = visual_feature.squeeze(3)

            """ Sequence modeling stage """
            contextual_feature = self.SequenceModeling(visual_feature)

            """ Prediction stage """
            prediction = self.Prediction(contextual_feature.contiguous())

            return prediction

    """### CRAFT Model"""

    #CRAFT

    class double_conv(nn.Module):
        def __init__(self, in_ch, mid_ch, out_ch):
            super(double_conv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            x = self.conv(x)
            return x


    class CRAFT(nn.Module):
        def __init__(self, pretrained=False, freeze=False):
            super(CRAFT, self).__init__()

            """ Base network """
            self.basenet = vgg16_bn(pretrained, freeze)

            """ U network """
            self.upconv1 = double_conv(1024, 512, 256)
            self.upconv2 = double_conv(512, 256, 128)
            self.upconv3 = double_conv(256, 128, 64)
            self.upconv4 = double_conv(128, 64, 32)

            num_class = 2
            self.conv_cls = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
                nn.Conv2d(16, num_class, kernel_size=1),
            )

            init_weights(self.upconv1.modules())
            init_weights(self.upconv2.modules())
            init_weights(self.upconv3.modules())
            init_weights(self.upconv4.modules())
            init_weights(self.conv_cls.modules())

        def forward(self, x):
            """ Base network """
            sources = self.basenet(x)

            """ U network """
            y = torch.cat([sources[0], sources[1]], dim=1)
            y = self.upconv1(y)

            y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
            y = torch.cat([y, sources[2]], dim=1)
            y = self.upconv2(y)

            y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
            y = torch.cat([y, sources[3]], dim=1)
            y = self.upconv3(y)

            y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
            y = torch.cat([y, sources[4]], dim=1)
            feature = self.upconv4(y)

            y = self.conv_cls(feature)

            return y.permute(0,2,3,1)

    def copyStateDict(state_dict):
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict

    def detect(img, detector, device):
        x = [np.transpose(normalizeMeanVariance(img), (2, 0, 1))]
        x = torch.from_numpy(np.array(x))
        x = x.to(device)
        with torch.no_grad():
            y = detector(x)
            
        region_score = y[0,:,:,0].cpu().data.numpy()
        affinity_score = y[0,:,:,1].cpu().data.numpy()

        # clear GPU memory
        del x
        del y
        torch.cuda.empty_cache()

        return region_score,affinity_score


    ================================================
    FILE: backend/annotator/segmentation/segment_from_point_clusters.py
    ================================================
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


    def load_node_features_and_labels(points_file, labels_file):
        # Load points
        points = np.loadtxt(points_file, dtype=int)

        # Load labels, handling 'None' entries
        with open(labels_file, "r") as f:
            labels = [line.strip() for line in f]

        # Convert labels to integers where possible, otherwise mark as None
        filtered_node_features = []
        filtered_labels = []

        for point, label in zip(points, labels):
            if label.lower() != "none":  # Exclude 'None' labels
                filtered_node_features.append(point)
                filtered_labels.append(int(label))  # Convert valid labels to int

        return np.array(filtered_node_features), np.array(filtered_labels)



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
                for (px, py, _), lab in zip(points, labels) #TODO ADD FEATURES
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
        for (px, py,_), label in zip(points, labels):  #TODO ADD FEATURES
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
        POINTS_FILEPATH = os.path.join(BASE_PATH, manuscript_name, "gnn-dataset", f"{page}_inputs_unnormalized.txt")
        LABELS_FILEPATH = os.path.join(BASE_PATH, manuscript_name, "gnn-dataset", f"{page}_labels_textline.txt")

        # Check if the manuscript lines directory exists
        if os.path.exists(os.path.join(BASE_PATH, manuscript_name, "lines", page)) == False:
            os.makedirs(os.path.join(BASE_PATH, manuscript_name, "lines", page))
            print("making the lines directory")
        LINES_DIR = os.path.join(BASE_PATH, manuscript_name, "lines", page)

        image = loadImage(IMAGE_FILEPATH)
        det = loadImage(HEATMAP_FILEPATH)
        filtered_node_features, filtered_labels = load_node_features_and_labels(POINTS_FILEPATH, LABELS_FILEPATH)

        det = det.squeeze()
        print(det.shape)
        if len(det.shape) == 3:  
            det = det[:, :, 0]  # Keep only one channel
        print(det.shape)

        #print(image.shape) this is x2 scale
        img2 = cv2.cvtColor(cv2.resize(image, det.shape[::-1]), cv2.COLOR_BGR2GRAY) 

        binarize_threshold = 100
        bounding_boxes = gen_bounding_boxes(det, binarize_threshold)
        labeled_bboxes = assign_labels_and_plot(bounding_boxes, filtered_node_features, filtered_labels, img2, output_path=os.path.join(BASE_PATH, manuscript_name, "frontend-graph-data", f"{page}.jpg"))

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





    ================================================
    FILE: backend/annotator/segmentation/segment_graph.py
    ================================================
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
        base_data_dir = f'instance/manuscripts/{m_name}/gnn-dataset'
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
                f.write(f"{width/2} {height/2}")


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








    ================================================
    FILE: backend/annotator/segmentation/segment_old_method.py
    ================================================
    import os
    import numpy as np
    import cv2
    from scipy.signal import find_peaks
    from skimage import io
    import glob # For easily finding all .png files in a directory

    from annotator.segmentation.utils import load_images_from_folder


    def gen_bounding_boxes(det,peaks, lineheight_baseline_percentile, binarize_threshold):
    img = np.uint8(det * 255)
    _, img1 = cv2.threshold(img, binarize_threshold, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    max_height = np.percentile(peaks[1:]-peaks[:-1],lineheight_baseline_percentile)
    # Extract bounding boxes from contours
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h<=max_height:
            bounding_boxes.append((x, y, w, h))
        else:
            n_b = np.int32(np.ceil(h/max_height))
            # Calculate the height of each box
            equal_height = h // n_b

            # Calculate the height adjustment needed for the last box to ensure total height is covered
            height_adjustment = h - (equal_height * n_b)

            for i in range(n_b):
                new_y = y + (i * equal_height)
                # Adjust the height of the last box if necessary
                box_height = equal_height + (height_adjustment if i == n_b - 1 else 0)
                bounding_boxes.append((x, new_y, w, box_height))

    return bounding_boxes

    def assign_lines(bounding_boxes,det):

    ys = det.sum(axis=1)
    thres = 0.5 * ys.max()
    peaks, _ = find_peaks(ys, height=thres,distance=det.shape[0]/100,width=5)

    lines = []
    xs = det.sum(axis = 0)
    thres = 0.5 * xs.max()
    xpeaks, _ = find_peaks(xs, height=thres)
    ys1 = det[:,xpeaks[0]:xpeaks[0]+100].sum(axis=1)
    thres = 0.5 * ys1.max()
    p1, _ = find_peaks(ys1, height=thres,distance=det.shape[0]/100,width=5)
    ys2 = det[:,xpeaks[-1]-100:xpeaks[-1]].sum(axis=1)
    thres = 0.5 * ys2.max()
    p2, _ = find_peaks(ys2, height=thres,distance=det.shape[0]/100,width=5)
    xmid = int((xpeaks[0]+xpeaks[-1])/2)
    ys3 = det[:,xmid-50:xmid+50].sum(axis=1)
    thres = 0.5 * ys3.max()
    p3, _ = find_peaks(ys3, height=thres,distance=det.shape[0]/100,width=5)
    if(peaks[0]-p1[0]>det.shape[0]/12):
        p1 = np.copy(p1[1:])
    p= min(p1, p2, p3, key=len)
    l = len(p)
    if(len(p1)>=l+1):
        k = len(p1) - len(p)
        ind = np.argmin(np.abs(p1[:k+1] - p[0]))
        peaks1 = p1[ind:l+ind]
    else:
        peaks1 = p1

    if(len(p2)>=l+1):
        k = len(p2) - len(p)
        ind = np.argmin(np.abs(p2[:k+1] - p[0]))
        peaks2 = p2[ind:l+ind]
    else:
        peaks2 = p2

    if(len(p3)>=l+1):
        k = len(p3) - len(p)
        ind = np.argmin(np.abs(p3[:k+1] - p[0]))
        peaks3 = p3[ind:l+ind]
    else:
        peaks3 = p3

    for box in bounding_boxes:
        x, y, _, h = box
        mid_y = y + h / 2  # Midpoint of the y-dimension
        wt1 = np.abs(x - xpeaks[0])
        wt2 = np.abs(x - xpeaks[-1])
        wt3 = np.abs(x - xmid)
        if x<=xmid:
            peaks = wt3*peaks1/(wt1+wt3)+wt1*peaks3/(wt1+wt3)
        else:
            peaks = wt3*peaks2/(wt2+wt3)+wt2*peaks3/(wt2+wt3)

        # Calculate the absolute difference between mid_y and each peak, then find the index of the minimum difference
        c_index = np.argmin(np.abs(peaks - mid_y))
        if(np.abs(mid_y-peaks[c_index])>20):
            c_index=-1
        lines.append(c_index)
    return lines,peaks1

    def crop_img(img):
        sum_rows = np.sum(img, axis=1)
        sum_cols = np.sum(img, axis=0)

        # Find indices where sum starts to vary for rows
        row_start = np.where(sum_rows != sum_rows[0])[0][0] if np.any(sum_rows != sum_rows[0]) else 0
        row_end = np.where(sum_rows != sum_rows[-1])[0][-1] if np.any(sum_rows != sum_rows[-1]) else len(sum_rows) - 1

        # Find indices where sum starts to vary for columns
        col_start = np.where(sum_cols != sum_cols[0])[0][0] if np.any(sum_cols != sum_cols[0]) else 0
        col_end = np.where(sum_cols != sum_cols[-1])[0][-1] if np.any(sum_cols != sum_cols[-1]) else len(sum_cols) - 1

        # Crop the image using the identified indices
        return np.copy(img[row_start:row_end+1, col_start:col_end+1])

    def gen_line_images(img2,peaks,bounding_boxes,lines, lineheight_baseline_percentile):
    # change here
    #   global lineheight_baseline_percentile
    line_images=[]
    max_height_line = np.percentile(peaks[1:]-peaks[:-1],lineheight_baseline_percentile)
    pad=int(max_height_line*0.2)
    for l in range(len(peaks)):
        # Filter bounding boxes for the current label
        filtered_boxes = [box for box, idx in zip(bounding_boxes, lines) if idx == l]

        if not filtered_boxes:
            continue

        # Calculate the total width and maximum height for the new image
        total_width = max(x for x, _,_, _ in filtered_boxes) + 500  # 10 pixels padding on each side
        max_height = max(h for _, _, _, h in filtered_boxes) + 250  # 5 pixels padding top and bottom
        miny = min(y for _, y,_, _ in filtered_boxes)
        # Create an empty image for this label
        new_img = np.ones((max_height, total_width), dtype=np.uint8)*np.int32(np.median(img2))

        for box in filtered_boxes:
            x, y, w, h = box
            blob = img2[y-pad:y+h+pad, x-10:x+w+10]
            new_img[y-miny:y-miny+h+2*pad,x-10:x+w+10]=blob
        line_images.append(crop_img(new_img))

    return line_images



    def load_saved_heatmaps_jpg(heatmap_folder_path):
        """
        Loads heatmaps that were saved as .jpg files by multiplying by 255
        and using cv2.imwrite.

        Args:
            heatmap_folder_path (str): The path to the folder where .jpg heatmaps are stored.
                                    e.g., "instance/manuscripts/your_m_name/heatmaps"

        Returns:
            tuple: (loaded_heatmaps, heatmap_filenames)
                - loaded_heatmaps (list): A list of NumPy arrays, each representing a heatmap
                                        with float values in the range [0, 1].
                - heatmap_filenames (list): A list of corresponding filenames (e.g., "image1.jpg").
        """
        if not os.path.isdir(heatmap_folder_path):
            print(f"Error: Heatmap folder not found at {heatmap_folder_path}")
            return [], []

        loaded_heatmaps = []
        heatmap_filenames = []

        # Find all .jpg files in the specified folder
        # Sort them to ensure a consistent order if that matters for your application
        jpg_files = sorted(glob.glob(os.path.join(heatmap_folder_path, "*.jpg"))) # Changed to *.jpg

        if not jpg_files:
            print(f"No .jpg files found in {heatmap_folder_path}") # Updated message
            return [], []

        print(f"Found {len(jpg_files)} heatmap .jpg files to load.")

        for file_path in jpg_files:
            # Load the image in grayscale mode
            # cv2.imread by default loads in BGR. For single channel (grayscale) like heatmaps:
            img_uint8 = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            if img_uint8 is None:
                print(f"Warning: Could not load image {file_path}. Skipping.")
                continue

            # Convert to float and scale back to [0, 1] range
            # The original data was likely float before being multiplied by 255 and saved as uint8
            heatmap_float = img_uint8.astype(np.float32) / 255.0

            loaded_heatmaps.append(heatmap_float)
            heatmap_filenames.append(os.path.basename(file_path))

        return loaded_heatmaps, heatmap_filenames


    def segment_lines(folder_path, lineheight_baseline_percentile=80, binarize_threshold=100):
        print(folder_path)
        #m_name = folder_path.split('/')[-2]
        m_name = os.path.basename(os.path.dirname(folder_path))



        # LOAD HEATMAP
        inp_images, file_names = load_images_from_folder(folder_path)
        out_images,_ = load_saved_heatmaps_jpg(f'instance/manuscripts/{m_name}/heatmaps')


        # ALGORITHM
        for det,image,file_name in zip(out_images,inp_images,file_names):
            print(file_name)
            ys = det.sum(axis=1)
            thres = 0.5 * ys.max()
            try:
                peaks, _ = find_peaks(ys, height=thres,distance=det.shape[0]/100,width=5)
                bounding_boxes = gen_bounding_boxes(det,peaks, lineheight_baseline_percentile, binarize_threshold)
                img2 = cv2.cvtColor(cv2.resize(image, det.shape[::-1]), cv2.COLOR_BGR2GRAY)
                
                lines,peaks1 = assign_lines(bounding_boxes,det)
                line_images = gen_line_images(img2,peaks1,bounding_boxes,lines, lineheight_baseline_percentile)

                if os.path.exists(f'instance/manuscripts/{m_name}/lines/{os.path.splitext(file_name)[0]}') == False:
                    os.makedirs(f'instance/manuscripts/{m_name}/lines/{os.path.splitext(file_name)[0]}')
                for i in range(len(line_images)):
                    cv2.imwrite(f'instance/manuscripts/{m_name}/lines/{os.path.splitext(file_name)[0]}/line{i+1:03d}.jpg',line_images[i])
            except:
                print("segmentation fails")
                with open(f'instance/manuscripts/{m_name}/gnn-dataset/failures.txt', 'a') as file:
                    file.write(f"{file_name}")

                if os.path.exists(f'instance/manuscripts/{m_name}/lines/{os.path.splitext(file_name)[0]}') == False:
                    os.makedirs(f'instance/manuscripts/{m_name}/lines/{os.path.splitext(file_name)[0]}')
                black_image = np.zeros((50, 900, 3), dtype=np.uint8)
                for i in range(5):
                    cv2.imwrite(f'instance/manuscripts/{m_name}/lines/{os.path.splitext(file_name)[0]}/line{i+1:03d}.jpg',black_image)
        



    ================================================
    FILE: backend/annotator/segmentation/utils.py
    ================================================
    import os
    import numpy as np
    import cv2
    from skimage import io

    # Function Definitions
    def loadImage(img_file):
        img = io.imread(img_file)           # RGB order
        if img.shape[0] == 2: img = img[0]
        if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.shape[2] == 4:   img = img[:,:,:3]
        img = np.array(img)

        return img

    def load_images_from_folder(folder_path):
        inp_images = []
        file_names = []
        
        # Get all files in the directory
        files = sorted(os.listdir(folder_path))
        
        for file in files:
            # Check if the file is an image (PNG or JPG)
            if file.lower().endswith(('.png', '.jpg', '.jpeg','.tif')):
                try:
                    # Construct the full file path
                    file_path = os.path.join(folder_path, file)
                    
                    # Open the image file
                    image = loadImage(file_path)
                    
                    # Append the image and filename to our lists
                    inp_images.append(image)
                    file_names.append(file)
                except Exception as e:
                    print(f"Error loading {file}: {str(e)}")
        
        return inp_images, file_names


    ================================================
    FILE: backend/instance/manuscripts/manuscript data stored here.md
    ================================================
    Data for each manuscript will be stored here


    ================================================
    FILE: backend/instance/models/recognition/put devanagari pth here.md
    ================================================
    Download devanagari.pth from [here](https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/devanagari.zip)


    ================================================
    FILE: backend/instance/models/segmentation/put craft_mlt_25k pth here.md
    ================================================
    Download craft_mlt_pth from [here](https://huggingface.co/amitesh863/craft/resolve/main/craft_mlt_25k.pth?download=true)


    ================================================
    FILE: frontend/README.md
    ================================================
    # frontend

    This template should help get you started developing with Vue 3 in Vite.

    ## Recommended IDE Setup

    [VSCode](https://code.visualstudio.com/) + [Volar](https://marketplace.visualstudio.com/items?itemName=Vue.volar) (and disable Vetur).

    ## Customize configuration

    See [Vite Configuration Reference](https://vite.dev/config/).

    ## Project Setup

    ```sh
    npm install
    ```

    ### Compile and Hot-Reload for Development

    ```sh
    npm run dev
    ```

    ### Compile and Minify for Production

    ```sh
    npm run build
    ```

    ### Lint with [ESLint](https://eslint.org/)

    ```sh
    npm run lint
    ```



    ================================================
    FILE: frontend/eslint.config.js
    ================================================
    import js from '@eslint/js'
    import pluginVue from 'eslint-plugin-vue'
    import skipFormatting from '@vue/eslint-config-prettier/skip-formatting'

    export default [
    {
        name: 'app/files-to-lint',
        files: ['**/*.{js,mjs,jsx,vue}'],
    },

    {
        name: 'app/files-to-ignore',
        ignores: ['**/dist/**', '**/dist-ssr/**', '**/coverage/**'],
    },

    js.configs.recommended,
    ...pluginVue.configs['flat/essential'],
    skipFormatting,
    ]



    ================================================
    FILE: frontend/index.html
    ================================================
    <!DOCTYPE html>
    <html lang="">
    <head>
        <meta charset="UTF-8">
        <link rel="icon" href="/favicon.ico">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Manuscript Annotation Tool</title>
    </head>
    <body>
        <div id="app"></div>
        <script type="module" src="/src/main.js"></script>
    </body>
    </html>



    ================================================
    FILE: frontend/jsconfig.json
    ================================================
    {
    "compilerOptions": {
        "paths": {
        "@/*": ["./src/*"]
        }
    },
    "exclude": ["node_modules", "dist"]
    }



    ================================================
    FILE: frontend/package.json
    ================================================
    {
    "name": "frontend",
    "version": "0.0.0",
    "private": true,
    "type": "module",
    "scripts": {
        "dev": "vite",
        "build": "vite build",
        "preview": "vite preview",
        "lint": "eslint . --fix",
        "format": "prettier --write src/"
    },
    "dependencies": {
        "@indic-transliteration/sanscript": "^1.3.1",
        "@popperjs/core": "^2.11.8",
        "@zip.js/zip.js": "^2.7.57",
        "axios": "^1.8.4",
        "bootstrap": "^5.3.3",
        "dropzone": "^6.0.0-beta.2",
        "pinia": "^2.2.6",
        "vue": "^3.5.13",
        "vue-router": "^4.4.5"
    },
    "devDependencies": {
        "@eslint/js": "^9.14.0",
        "@vitejs/plugin-vue": "^5.2.1",
        "@vue/eslint-config-prettier": "^10.1.0",
        "eslint": "^9.14.0",
        "eslint-plugin-vue": "^9.30.0",
        "prettier": "^3.3.3",
        "sass": "1.77.6",
        "vite": "^6.0.1",
        "vite-plugin-vue-devtools": "^7.6.5"
    }
    }



    ================================================
    FILE: frontend/vite.config.js
    ================================================
    import { fileURLToPath, URL } from 'node:url'

    import { defineConfig } from 'vite'
    import vue from '@vitejs/plugin-vue'
    import vueDevTools from 'vite-plugin-vue-devtools'

    // https://vite.dev/config/
    export default defineConfig({
    plugins: [
        vue(),
        vueDevTools(),
    ],
    resolve: {
        alias: {
        '@': fileURLToPath(new URL('./src', import.meta.url))
        },
    },
    })



    ================================================
    FILE: frontend/.editorconfig
    ================================================
    [*.{js,jsx,mjs,cjs,ts,tsx,mts,cts,vue}]
    charset = utf-8
    indent_size = 2
    indent_style = space
    insert_final_newline = true
    trim_trailing_whitespace = true



    ================================================
    FILE: frontend/.prettierrc.json
    ================================================

    {
    "$schema": "https://json.schemastore.org/prettierrc",
    "semi": false,
    "singleQuote": true,
    "printWidth": 100
    }



    ================================================
    FILE: frontend/src/App.vue
    ================================================
    <script setup>
    import { RouterView } from 'vue-router'

    function setThemeBasedOnPreference() {
    const prefersDarkScheme = window.matchMedia('(prefers-color-scheme: dark)').matches
    document.documentElement.setAttribute('data-bs-theme', prefersDarkScheme ? 'dark' : 'light')
    }

    setThemeBasedOnPreference()

    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    setThemeBasedOnPreference()
    })
    </script>

    <template>
    <RouterView />
    </template>



    ================================================
    FILE: frontend/src/main.js
    ================================================
    import './assets/main.scss'
    import "dropzone/dist/dropzone.css";

    import { createApp } from 'vue'

    import App from './App.vue'
    import { createPinia } from 'pinia';
    import router from './router';

    const pinia = createPinia();
    const app = createApp(App)

    app.use(router)
    app.use(pinia);
    app.mount('#app');



    ================================================
    FILE: frontend/src/assets/base.scss
    ================================================
    /* color palette from <https://github.com/vuejs/theme> */

    @import "bootstrap/scss/bootstrap";

    :root {
    --vt-c-white: #ffffff;
    --vt-c-white-soft: #f8f8f8;
    --vt-c-white-mute: #f2f2f2;

    --vt-c-black: #181818;
    --vt-c-black-soft: #222222;
    --vt-c-black-mute: #282828;

    --vt-c-indigo: #2c3e50;

    --vt-c-divider-light-1: rgba(60, 60, 60, 0.29);
    --vt-c-divider-light-2: rgba(60, 60, 60, 0.12);
    --vt-c-divider-dark-1: rgba(84, 84, 84, 0.65);
    --vt-c-divider-dark-2: rgba(84, 84, 84, 0.48);

    --vt-c-text-light-1: var(--vt-c-indigo);
    --vt-c-text-light-2: rgba(60, 60, 60, 0.66);
    --vt-c-text-dark-1: var(--vt-c-white);
    --vt-c-text-dark-2: rgba(235, 235, 235, 0.64);
    }

    /* semantic color variables for this project */
    :root {
    --color-background: var(--vt-c-white);
    --color-background-soft: var(--vt-c-white-soft);
    --color-background-mute: var(--vt-c-white-mute);

    --color-border: var(--vt-c-divider-light-2);
    --color-border-hover: var(--vt-c-divider-light-1);

    --color-heading: var(--vt-c-text-light-1);
    --color-text: var(--vt-c-text-light-1);

    --section-gap: 160px;
    }

    @media (prefers-color-scheme: dark) {
    :root {
        --color-background: var(--vt-c-black);
        --color-background-soft: var(--vt-c-black-soft);
        --color-background-mute: var(--vt-c-black-mute);

        --color-border: var(--vt-c-divider-dark-2);
        --color-border-hover: var(--vt-c-divider-dark-1);

        --color-heading: var(--vt-c-text-dark-1);
        --color-text: var(--vt-c-text-dark-2);
    }
    }

    *,
    *::before,
    *::after {
    box-sizing: border-box;
    margin: 0;
    font-weight: normal;
    }

    body {
    min-height: 100vh;
    color: var(--color-text);
    background: var(--color-background);
    transition:
        color 0.5s,
        background-color 0.5s;
    line-height: 1.6;
    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        'Segoe UI',
        Roboto,
        Oxygen,
        Ubuntu,
        Cantarell,
        'Fira Sans',
        'Droid Sans',
        'Helvetica Neue',
        sans-serif;
    text-rendering: optimizeLegibility;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    }



    ================================================
    FILE: frontend/src/assets/main.scss
    ================================================
    @import './base.scss';

    #app {
    max-width: 100%;
    height: 100vh;
    margin: 0 auto;
    font-weight: normal;
    }



    ================================================
    FILE: frontend/src/components/new-AnnotationBlock.vue
    ================================================
    <script setup>
    import { reactive, ref, watch, onMounted, computed } from 'vue' // Added computed
    import Sanscript from '@indic-transliteration/sanscript'
    import { useAnnotationStore } from '@/stores/annotationStore'
    import { handleInput } from './typing-utils/devanagariInputUtils'

    const BASE_PATH = `${import.meta.env.VITE_BACKEND_URL}/line-images`

    const props = defineProps(['line_name', 'line_data', 'page_name', 'manuscript_name'])
    const annotationStore = useAnnotationStore()

    const isHK = ref(false)

    const textboxClassObject = reactive({
    'form-control': true,
    'mb-2': true,
    'me-2': true,
    'devanagari-textbox': true,
    'is-valid': false,
    })


    const devanagari = ref(props.line_data.predicted_label)
    const hk = ref(Sanscript.t(props.line_data.predicted_label, 'devanagari', 'hk'))


    const devanagariInput = ref(null)

    // Watcher for devanagari changes to update user annotations in real-time (or on blur)
    // This is optional, the save button already does this explicitly.
    // watch(devanagari, (newValue) => {
    //   // Debounce this if you want real-time saving to avoid too many store updates
    //   if (annotationStore.userAnnotations.length > 0 &&
    //       annotationStore.userAnnotations[0]['annotations'][props.page_name] &&
    //       annotationStore.userAnnotations[0]['annotations'][props.page_name][props.line_name]) {
    //     annotationStore.userAnnotations[0]['annotations'][props.page_name][props.line_name]['ground_truth'] = newValue;
    //   } else {
    //     // Structure might not be ready, defer to save button or ensure structure
    //     // is built when line_data is first available.
    //   }
    // });

    watch(hk, function () {
    if (!isHK.value) return
    devanagari.value = Sanscript.t(hk.value, 'hk', 'devanagari')
    })

    function toggleHK() {
    hk.value = Sanscript.t(devanagari.value, 'devanagari', 'hk')
    isHK.value = !isHK.value
    }

    function save() {
    // Ensure the path in userAnnotations exists before trying to assign to it.
    // AnnotationPage.vue should have initialized userAnnotations[0]['annotations'][props.page_name] = {}
    if (annotationStore.userAnnotations.length > 0 &&
        annotationStore.userAnnotations[0]['annotations'][props.page_name]) {
        // Initialize the specific line's annotation object if it doesn't exist
        if (!annotationStore.userAnnotations[0]['annotations'][props.page_name][props.line_name]) {
        annotationStore.userAnnotations[0]['annotations'][props.page_name][props.line_name] = {};
        }
        annotationStore.userAnnotations[0]['annotations'][props.page_name][props.line_name]['ground_truth'] = devanagari.value;
        textboxClassObject['is-valid'] = true;
    } else {
        console.error("Cannot save annotation, userAnnotations structure not properly initialized for page:", props.page_name);
        // Optionally provide user feedback
    }
    }


    const boundHandleInput = (event) => handleInput(event, devanagari)

    onMounted(() => {
    if (devanagariInput.value) {
        devanagariInput.value.addEventListener('keydown', boundHandleInput);
    }
    // When the component mounts, check if there's already a ground_truth for this line
    // and pre-fill the devanagari input if so. This allows edits to persist across page navigations.
    if (annotationStore.userAnnotations.length > 0 &&
        annotationStore.userAnnotations[0]['annotations'][props.page_name] &&
        annotationStore.userAnnotations[0]['annotations'][props.page_name][props.line_name] &&
        annotationStore.userAnnotations[0]['annotations'][props.page_name][props.line_name]['ground_truth'] !== undefined) {
        devanagari.value = annotationStore.userAnnotations[0]['annotations'][props.page_name][props.line_name]['ground_truth'];
        // Also update hk if devanagari was loaded from store
        hk.value = Sanscript.t(devanagari.value, 'devanagari', 'hk');
        textboxClassObject['is-valid'] = true; // Mark as valid if previously saved
    } else {
        // If no existing ground_truth, devanagari.value remains the predicted_label
        // And textboxClassObject['is-valid'] remains false until saved.
    }
    })
    </script>

    <template>
    <!-- The image_path prop for the line is passed as props.line_data.image_path -->
    <!-- In your new backend, props.line_data.image_path IS props.line_name (without extension) -->
    <!-- So, this looks correct. -->
    <img
        :src="`${BASE_PATH}/${props.manuscript_name}/${props.page_name}/${props.line_data.image_path}`"
        class="mb-2 manuscript-segment-img"
        :alt="`Line image for ${props.line_name}`"
    />
    <div class="annotation-input">
        <input 
        ref="devanagariInput"
        v-model="devanagari" 
        type="text" 
        :class="textboxClassObject" 
        />
        <button class="btn mb-2 me-2 small-grey-btn" @click="toggleHK">Roman</button>
        <button class="btn mb-2 me-2 small-grey-btn" @click="save">Save</button>
    </div>
    <input v-model="hk" type="text" class="form-control mb-2" v-if="isHK" />
    </template>

    <style>
    .manuscript-segment-img {
    display: block;
    }

    .annotation-input {
    width: 100%;
    display: flex;
    }

    .devanagari-textbox {
    flex-grow: 1;
    display: inline-block;
    }

    .small-grey-btn {
    font-size: 0.8rem;
    padding: 0.25rem 0.5rem;
    background-color: #6c757d; /* Bootstrap's secondary/grey color */
    border: none;
    color: #fff;
    }
    </style>


    ================================================
    FILE: frontend/src/components/new-AnnotationPage.vue
    ================================================
    <script setup>
    import { useAnnotationStore } from '@/stores/annotationStore'
    import AnnotationBlock from './new-AnnotationBlock.vue'

    const props = defineProps(['data', 'page_name', 'manuscript_name'])
    const annotationStore = useAnnotationStore()

    // PROBLEM AREA: This line runs when AnnotationPage is initialized.
    // If we navigate from Page A -> C -> B, and in C we haven't "saved" (triggered recognition) yet,
    // then props.data might be an empty object initially.
    // More importantly, this line re-initializes the user annotations for the page EVERY time
    // AnnotationPage is rendered, potentially wiping out existing user annotations if they switch pages.
    // annotationStore.userAnnotations[0]['annotations'][props.page_name] = {}
    //
    // SUGGESTED CHANGE: Initialize only if not already present.
    // And ensure there's a userAnnotation entry.

    if (annotationStore.userAnnotations.length > 0) {
    if (!annotationStore.userAnnotations[0]['annotations'][props.page_name]) {
        annotationStore.userAnnotations[0]['annotations'][props.page_name] = {};
    }
    } else {
    // This case should ideally be handled earlier, e.g., when a manuscript is first processed
    // or when the user logs in/starts a session.
    // For now, we can log a warning or create the initial structure if absolutely necessary,
    // but it's better if userAnnotations[0] is guaranteed to exist by this point.
    console.warn('userAnnotations array is empty. Cannot initialize page annotations.');
    // If you MUST initialize it here (less ideal):
    // annotationStore.userAnnotations.push({
    // manuscript_name: props.manuscript_name, // Or get from store if available
    // selected_model: annotationStore.modelName, // Or get from store
    // annotations: {
    // [props.page_name]: {}
    // }
    // });
    }

    </script>

    <template>
    <div>
        <!-- This v-for loop is fine. It will correctly iterate over props.data
            which comes from annotationStore.recognitions[manuscript_name][page_name].
            If props.data is empty (e.g., before recognition on Page C is done),
            nothing will be rendered here, which is correct. -->
        <div v-for="(line_data, line_name) in props.data" :key="line_name">
        <AnnotationBlock
            :line_name="line_name"
            :line_data="line_data"
            :page_name="props.page_name"
            :manuscript_name="props.manuscript_name"
        />
        </div>
    </div>
    </template>


    ================================================
    FILE: frontend/src/components/new-IMG2TXT.vue
    ================================================
    <script setup>
    import { useRouter } from 'vue-router'
    import AnnotationPage from '@/components/new-AnnotationPage.vue'
    import { useAnnotationStore } from '@/stores/annotationStore'
    import CharacterPalette from './typing-utils/characterPalette.vue'

    const router = useRouter()
    const annotationStore = useAnnotationStore();

    const manuscript_name = Object.keys(annotationStore.recognitions)[0];
    const manuscriptPages = Object.keys(annotationStore.recognitions[manuscript_name] || {}); // Ensure manuscript_name exists
    console.log('ff',manuscript_name)
    console.log('ff',manuscriptPages)

    if (manuscript_name && annotationStore.currentPage && manuscriptPages.includes(annotationStore.currentPage)) {
    // If currentPage in store is valid for the current manuscript, use it.
    // No change needed to annotationStore.currentPage here, it's already correct.
    } else if (manuscript_name && manuscriptPages.length > 0) {
    // Otherwise, default to the first page of the current manuscript
    annotationStore.currentPage = manuscriptPages[0];
    } else {
    // Handle case where there are no pages or no manuscript
    annotationStore.currentPage = null;
    console.warn('No pages found for manuscript or manuscript_name is missing in new-IMG2TXT.vue');
    }

    function uploadGroundTruth() {
    annotationStore.calculateLevenshteinDistances()
    annotationStore.userAnnotations.forEach((elem) => {
        elem['model_name'] = annotationStore.modelName
        console.log('added Model name', annotationStore.modelName)
    })
    fetch(import.meta.env.VITE_BACKEND_URL + '/fine-tune', {
        method: 'POST',
        headers: {
        'Content-Type': 'application/json',
        },
        body: JSON.stringify(annotationStore.userAnnotations),
    }).then(() => {
        annotationStore.reset()
        router.push({ name: 'upload-manuscript' })
    })
    }


    function switchToSemiAutoSegmentation() {
    router.push({ name: 'new-semi-segment' }) // 
    }

    </script>

    <template>
    <div class="mb-3">
        <label for="model-name" class="form-label">Model name</label>
        <input
        class="form-control"
        placeholder="Name your model..."
        v-model="annotationStore.modelName"
        />
    </div>
    <div class="mb-3">
        <button class="btn btn-primary me-2" @click="uploadGroundTruth">Fine-tune</button>
        <!-- <button class="btn btn-warning me-2" @click="switchToSegmentation">Correct Image Segments</button> -->
        <button class="btn btn-warning me-2" @click="switchToSemiAutoSegmentation">Semi Segmentation</button>
        <button class="btn btn-success me-2" @click="annotationStore.exportToTxt">Export</button>
        <CharacterPalette />
    </div>
    <div class="mb-3">
        <label for="page" class="form-label">Page</label>
        <select
        class="form-select"
        id="page"
        v-model="annotationStore.currentPage"
        placeholder="Select a model"
        >
        <option
            v-for="(page_data, page_name) in annotationStore.recognitions[manuscript_name]"
            :key="page_name"
            :value="page_name"
        >
            {{ page_name }}
        </option>
        </select>
    </div>
    <AnnotationPage
        v-for="(page_data, page_name) in annotationStore.recognitions[manuscript_name]"
        :key="page_data"
        :data="page_data"
        :page_name="page_name"
        :manuscript_name="manuscript_name"
        v-show="annotationStore.currentPage === page_name"
    />
    </template>



    ================================================
    FILE: frontend/src/components/new-SemiSegmentationSection.vue
    ================================================
    <template>
    <div class="manuscript-viewer">
        <!-- Top Toolbar: Collapsible -->
        <div class="toolbar">
        <h10>{{ manuscriptName }} - Page {{ currentPage }}</h10>
        <div v-show="!isToolbarCollapsed" class="toolbar-controls">
            <button @click="previousPage" :disabled="loading || isProcessingSave">Previous</button>
            <button @click="nextPage" :disabled="loading || isProcessingSave">Next</button>
            <button @click="saveAndGoNext" :disabled="loading || isProcessingSave">Save & Next (S)</button>
            <button @click="goToIMG2TXTPage" :disabled="loading || isProcessingSave">Annotate Text</button>
            <div class="toggle-container">
            <label>
                <input type="checkbox" v-model="editMode" :disabled="isProcessingSave" />
                Edit Mode (W)
            </label>
            </div>
        </div>
        <button class="panel-toggle-btn" @click="isToolbarCollapsed = !isToolbarCollapsed">
            {{ isToolbarCollapsed ? 'Show Toolbar' : 'Hide' }}
        </button>
        </div>

        <!-- Main Content: Visualization Area -->
        <div class="visualization-container" ref="container">
        <div v-if="isProcessingSave" class="processing-save-notice">
            Saving graph and processing... Please wait.
        </div>
        <div v-if="error" class="error-message">
            {{ error }}
        </div>
        <div v-if="loading" class="loading">
            Loading Page Data...
        </div>
        <div v-else class="image-container" :style="{ width: `${scaledWidth}px`, height: `${scaledHeight}px` }">
            <img
            v-if="imageData"
            :src="`data:image/jpeg;base64,${imageData}`"
            :width="scaledWidth"
            :height="scaledHeight"
            class="manuscript-image"
            @load="imageLoaded = true"
            />
            <div v-else class="placeholder-image" :style="{ width: `${scaledWidth}px`, height: `${scaledHeight}px` }">
            No image available
            </div>

            <svg
            v-if="graphIsLoaded"
            class="graph-overlay"
            :class="{ 'is-visible': editMode }"
            :width="scaledWidth"
            :height="scaledHeight"
            :style="{ cursor: svgCursor }"
            @click="editMode && onBackgroundClick($event)"
            @mousemove="handleSvgMouseMove"
            @mouseleave="handleSvgMouseLeave"
            ref="svgOverlayRef"
            >
            <line
                v-for="(edge, index) in workingGraph.edges"
                :key="`edge-${index}`"
                :x1="scaleX(workingGraph.nodes[edge.source].x)"
                :y1="scaleY(workingGraph.nodes[edge.source].y)"
                :x2="scaleX(workingGraph.nodes[edge.target].x)"
                :y2="scaleY(workingGraph.nodes[edge.target].y)"
                :stroke="getEdgeColor(edge)"
                :stroke-width="isEdgeSelected(edge) ? 3 : 2.5"
                @click.stop="editMode && onEdgeClick(edge, $event)"
            />

            <circle
                v-for="(node, nodeIndex) in workingGraph.nodes"
                :key="`node-${nodeIndex}`"
                :cx="scaleX(node.x)"
                :cy="scaleY(node.y)"
                :r="getNodeRadius(nodeIndex)"
                :fill="getNodeColor(nodeIndex)"
                @click.stop="editMode && onNodeClick(nodeIndex, $event)"
            />

            <line
                v-if="editMode && selectedNodes.length === 1 && tempEndPoint && !isAKeyPressed && !isDKeyPressed"
                :x1="scaleX(workingGraph.nodes[selectedNodes[0]].x)"
                :y1="scaleY(workingGraph.nodes[selectedNodes[0]].y)"
                :x2="tempEndPoint.x"
                :y2="tempEndPoint.y"
                stroke="#ff9500"
                stroke-width="2.5"
                stroke-dasharray="5,5"
            />
            </svg>
        </div>
        </div>

        <!-- Bottom Panel: Collapsible -->
        <div class="bottom-panel">
        <div class="panel-toggle-bar" @click="isControlsCollapsed = !isControlsCollapsed">
            <div class="edit-instructions">
                <p v-if="isControlsCollapsed && editMode">Hold 'a' to connect, 'd' to delete. Press 's' to save & next. Toggle edit with 'w'.</p>
                <p v-else-if="isControlsCollapsed && !editMode">Press 'w' to enter edit mode.</p>
                <p v-else-if="!isAKeyPressed && !isDKeyPressed">Select nodes to manage edges, or use hotkeys.</p>
                <p v-else-if="isAKeyPressed">Release 'A' to connect nodes.</p>
                <p v-else-if="isDKeyPressed">Release 'D' to stop deleting.</p>
            </div>
            <button class="panel-toggle-btn">
                {{ isControlsCollapsed ? 'Show Controls' : 'Hide Controls' }}
            </button>
        </div>

        <div v-show="!isControlsCollapsed" class="bottom-panel-content">
            <div v-if="editMode && !isAKeyPressed && !isDKeyPressed" class="edit-controls">
                <div class="edit-actions">
                    <button @click="resetSelection">Cancel Selection</button>
                    <button @click="addEdge" :disabled="selectedNodes.length !== 2 || edgeExists(selectedNodes[0], selectedNodes[1])">Add Edge</button>
                    <button @click="deleteEdge" :disabled="selectedNodes.length !== 2 || !edgeExists(selectedNodes[0], selectedNodes[1])">Delete Edge</button>
                </div>
            </div>

            <div v-if="editMode && graphIsLoaded" class="modifications-log-container">
                <button @click="saveCurrentGraph" :disabled="loading || isProcessingSave">Save Graph</button>
                <div v-if="modifications.length > 0" class="modifications-details">
                    <h3>Modifications ({{ modifications.length }})</h3>
                    <button @click="resetModifications" :disabled="loading">Reset All Changes</button>
                    <ul>
                    <li v-for="(mod, index) in modifications" :key="index" class="modification-item">
                        {{ mod.type === 'add' ? 'Added' : 'Removed' }} edge: {{ mod.source }} ↔ {{ mod.target }}
                        <button @click="undoModification(index)" class="undo-button">Undo</button>
                    </li>
                    </ul>
                </div>
                <p v-else-if="!loading">No modifications in this session.</p>
            </div>
        </div>
        </div>
    </div>
    </template>

    <script setup>
    import { ref, onMounted, onBeforeUnmount, computed, watch, reactive } from 'vue';
    import { useAnnotationStore } from '@/stores/annotationStore';
    import { generateLayoutGraph } from './layout-analysis-utils/LayoutGraphGenerator.js';
    import { useRouter } from 'vue-router';

    const router = useRouter();
    const annotationStore = useAnnotationStore();

    // --- Core State ---
    const manuscriptName = computed(() => Object.keys(annotationStore.recognitions)[0] || '');
    const currentPage = computed(() => annotationStore.currentPage);
    const loading = ref(true);
    const isProcessingSave = ref(false);
    const error = ref(null);
    const imageData = ref('');
    const imageLoaded = ref(false);

    // --- UI State ---
    const isToolbarCollapsed = ref(true);
    const isControlsCollapsed = ref(true);
    const editMode = ref(true);
    const svgCursor = computed(() => {
    if (!editMode.value) return 'default';
    if (isAKeyPressed.value) return 'crosshair';
    if (isDKeyPressed.value) return 'not-allowed';
    return 'default';
    });

    // --- Graph & Geometry State ---
    const dimensions = ref([0, 0]);
    const points = ref([]);
    const graph = ref({ nodes: [], edges: [] });
    const workingGraph = reactive({ nodes: [], edges: [] });
    const modifications = ref([]);
    const nodeEdgeCounts = ref({});
    const selectedNodes = ref([]);
    const tempEndPoint = ref(null);
    const isDKeyPressed = ref(false);
    const isAKeyPressed = ref(false);
    const hoveredNodesForMST = reactive(new Set());
    const container = ref(null);
    const svgOverlayRef = ref(null);

    // --- Constants ---
    const scaleFactor = 1.0;
    const NODE_HOVER_RADIUS = 7;
    const EDGE_HOVER_THRESHOLD = 5;

    // --- Computed Properties ---
    const scaledWidth = computed(() => Math.floor(dimensions.value[0] * scaleFactor));
    const scaledHeight = computed(() => Math.floor(dimensions.value[1] * scaleFactor));
    const scaleX = (x) => x * scaleFactor;
    const scaleY = (y) => y * scaleFactor;
    const graphIsLoaded = computed(() => workingGraph.nodes && workingGraph.nodes.length > 0);

    // --- Data Fetching and Initialization ---
    const fetchPageData = async () => {
    if (!manuscriptName.value || !currentPage.value) return;
    loading.value = true;
    error.value = null;
    modifications.value = [];

    try {
        const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/semi-segment/${manuscriptName.value}/${currentPage.value}`);
        if (!response.ok) throw new Error((await response.json()).error || 'Failed to fetch page data');
        const data = await response.json();

        dimensions.value = data.dimensions;
        imageData.value = data.image || '';
        points.value = data.points.map(p => ({ coordinates: [p[0], p[1]], segment: null }));

        if (data.graph) {
        graph.value = data.graph;
        } else if (data.points?.length > 0) {
        graph.value = generateLayoutGraph(data.points);
        await saveGeneratedGraph(manuscriptName.value, currentPage.value, graph.value);
        }
        resetWorkingGraph();
    } catch (err) {
        console.error('Error fetching page data:', err);
        error.value = err.message;
    } finally {
        loading.value = false;
    }
    };

    const updateUniqueNodeEdgeCounts = () => {
        const counts = {};
        if (!workingGraph.nodes) return;
        workingGraph.nodes.forEach((_, index) => { counts[index] = 0; });

        if (!workingGraph.edges) {
            nodeEdgeCounts.value = counts;
            return;
        }

        const uniqueEdges = new Set();
        for (const edge of workingGraph.edges) {
            const key = `${Math.min(edge.source, edge.target)}-${Math.max(edge.source, edge.target)}`;
            uniqueEdges.add(key);
        }

        for (const key of uniqueEdges) {
            const [source, target] = key.split('-').map(Number);
            if (counts[source] !== undefined) counts[source]++;
            if (counts[target] !== undefined) counts[target]++;
        }

        nodeEdgeCounts.value = counts;
    };

    watch(() => workingGraph.edges, updateUniqueNodeEdgeCounts, { deep: true, immediate: true });

    const resetWorkingGraph = () => {
    workingGraph.nodes = JSON.parse(JSON.stringify(graph.value.nodes || []));
    workingGraph.edges = JSON.parse(JSON.stringify(graph.value.edges || []));
    resetSelection();
    };

    // --- Graph Styling ---
    const getNodeColor = (nodeIndex) => {
    if (isAKeyPressed.value && hoveredNodesForMST.has(nodeIndex)) return '#00bcd4';
    if (isNodeSelected(nodeIndex)) return '#ff9500';

    const edgeCount = nodeEdgeCounts.value[nodeIndex];
    if (edgeCount < 2) return '#f44336';
    if (edgeCount === 2) return '#4CAF50';
    if (edgeCount > 2) return '#2196F3';
    return '#cccccc';
    };
    const getNodeRadius = (nodeIndex) => {
    const edgeCount = nodeEdgeCounts.value[nodeIndex];
    if (isAKeyPressed.value && hoveredNodesForMST.has(nodeIndex)) return 7;
    if (isNodeSelected(nodeIndex)) return 6;
    return (edgeCount < 2) ? 5 : 3;
    };
    const getEdgeColor = (edge) => edge.modified ? '#f44336' : '#ffffff';
    const isNodeSelected = (nodeIndex) => selectedNodes.value.includes(nodeIndex);
    const isEdgeSelected = (edge) => {
    return selectedNodes.value.length === 2 &&
        ((selectedNodes.value[0] === edge.source && selectedNodes.value[1] === edge.target) ||
        (selectedNodes.value[0] === edge.target && selectedNodes.value[1] === edge.source));
    };

    // --- User Interactions & Event Handlers ---
    const resetSelection = () => { selectedNodes.value = []; tempEndPoint.value = null; };
    const onNodeClick = (nodeIndex, event) => {
    if (isAKeyPressed.value || isDKeyPressed.value) return;
    event.stopPropagation();
    const existingIndex = selectedNodes.value.indexOf(nodeIndex);
    if (existingIndex !== -1) selectedNodes.value.splice(existingIndex, 1);
    else selectedNodes.value.length < 2 ? selectedNodes.value.push(nodeIndex) : selectedNodes.value = [nodeIndex];
    };
    const onEdgeClick = (edge, event) => {
    if (isAKeyPressed.value || isDKeyPressed.value) return;
    event.stopPropagation();
    selectedNodes.value = [edge.source, edge.target];
    };
    const onBackgroundClick = () => { if (!isAKeyPressed.value && !isDKeyPressed.value) resetSelection(); };
    const handleSvgMouseMove = (event) => {
    if (!editMode.value || !svgOverlayRef.value) return;
    const { left, top } = svgOverlayRef.value.getBoundingClientRect();
    const mouseX = event.clientX - left;
    const mouseY = event.clientY - top;

    if (isDKeyPressed.value) handleEdgeHoverDelete(mouseX, mouseY);
    else if (isAKeyPressed.value) handleNodeHoverCollect(mouseX, mouseY);
    else if (selectedNodes.value.length === 1) tempEndPoint.value = { x: mouseX, y: mouseY };
    else tempEndPoint.value = null;
    };
    const handleSvgMouseLeave = () => { if (selectedNodes.value.length === 1) tempEndPoint.value = null; };
    const handleGlobalKeyDown = (e) => {
    // Allow toggling edit mode regardless of focus
    if (e.key.toLowerCase() === 'w' && !e.repeat) {
        e.preventDefault();
        editMode.value = !editMode.value;
        return;
    }

    // Hotkeys that should only work in edit mode
    if (!editMode.value || e.repeat) return;

    const key = e.key.toLowerCase();
    if (key === 's') {
        e.preventDefault();
        if (!loading.value && !isProcessingSave.value) saveAndGoNext();
    }
    if (key === 'd') {
        e.preventDefault();
        isDKeyPressed.value = true;
        resetSelection();
    }
    if (key === 'a') {
        e.preventDefault();
        isAKeyPressed.value = true;
        hoveredNodesForMST.clear();
        resetSelection();
    }
    };
    const handleGlobalKeyUp = (e) => {
    // Key up events should also be guarded by edit mode
    if (!editMode.value) return;

    const key = e.key.toLowerCase();
    if (key === 'd') isDKeyPressed.value = false;
    if (key === 'a') {
        isAKeyPressed.value = false;
        if (hoveredNodesForMST.size >= 2) addMSTEdges();
        hoveredNodesForMST.clear();
    }
    };


    // --- Edge Manipulation Logic (FIXED FOR 'label') ---
    const edgeExists = (nodeA, nodeB) => workingGraph.edges.some(e => (e.source === nodeA && e.target === nodeB) || (e.source === nodeB && e.target === nodeA));
    const addEdge = () => {
    if (selectedNodes.value.length !== 2 || edgeExists(...selectedNodes.value)) return;
    const [source, target] = selectedNodes.value;
    const newEdge = { source, target, label: 0, modified: true };
    workingGraph.edges.push(newEdge);
    modifications.value.push({ type: 'add', source, target, label: 0 });
    resetSelection();
    };
    const deleteEdge = () => {
    if (selectedNodes.value.length !== 2) return;
    const [source, target] = selectedNodes.value;
    const edgeIndex = workingGraph.edges.findIndex(e => (e.source === source && e.target === target) || (e.source === target && e.target === source));
    if (edgeIndex === -1) return;
    const removedEdge = workingGraph.edges.splice(edgeIndex, 1)[0];
    modifications.value.push({ type: 'delete', source: removedEdge.source, target: removedEdge.target, label: removedEdge.label });
    resetSelection();
    };
    const undoModification = (index) => {
    const mod = modifications.value.splice(index, 1)[0];
    if (mod.type === 'add') {
        const edgeIndex = workingGraph.edges.findIndex(e => e.source === mod.source && e.target === mod.target);
        if (edgeIndex !== -1) workingGraph.edges.splice(edgeIndex, 1);
    } else if (mod.type === 'delete') {
        workingGraph.edges.push({ source: mod.source, target: mod.target, label: mod.label, modified: true });
    }
    };
    const resetModifications = () => { resetWorkingGraph(); modifications.value = []; };

    // --- Hover-based Edge Manipulation (FIXED FOR 'label') ---
    const distanceToLineSegment = (px, py, x1, y1, x2, y2) => Math.hypot(px - (x1 + Math.max(0, Math.min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2) || 1))) * (x2 - x1)), py - (y1 + Math.max(0, Math.min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2) || 1))) * (y2 - y1)));
    const handleEdgeHoverDelete = (mouseX, mouseY) => {
    for (let i = workingGraph.edges.length - 1; i >= 0; i--) {
        const edge = workingGraph.edges[i];
        const n1 = workingGraph.nodes[edge.source], n2 = workingGraph.nodes[edge.target];
        if (n1 && n2 && distanceToLineSegment(mouseX, mouseY, scaleX(n1.x), scaleY(n1.y), scaleX(n2.x), scaleY(n2.y)) < EDGE_HOVER_THRESHOLD) {
        const removed = workingGraph.edges.splice(i, 1)[0];
        modifications.value.push({ type: 'delete', source: removed.source, target: removed.target, label: removed.label });
        }
    }
    };
    const handleNodeHoverCollect = (mouseX, mouseY) => {
    workingGraph.nodes.forEach((node, index) => {
        if (Math.hypot(mouseX - scaleX(node.x), mouseY - scaleY(node.y)) < NODE_HOVER_RADIUS) hoveredNodesForMST.add(index);
    });
    };
    const calculateMST = (indices, nodes) => {
    const points = indices.map(i => ({ ...nodes[i], originalIndex: i }));
    const edges = [];
    for (let i = 0; i < points.length; i++) for (let j = i + 1; j < points.length; j++) {
        edges.push({ source: points[i].originalIndex, target: points[j].originalIndex, weight: Math.hypot(points[i].x - points[j].x, points[i].y - points[j].y) });
    }
    edges.sort((a, b) => a.weight - b.weight);
    const parent = {};
    indices.forEach(i => parent[i] = i);
    const find = i => (parent[i] === i ? i : (parent[i] = find(parent[i])));
    const union = (i, j) => { const rootI = find(i), rootJ = find(j); if (rootI !== rootJ) { parent[rootJ] = rootI; return true; } return false; };
    return edges.filter(e => union(e.source, e.target));
    };
    const addMSTEdges = () => {
    calculateMST(Array.from(hoveredNodesForMST), workingGraph.nodes).forEach(edge => {
        if (!edgeExists(edge.source, edge.target)) {
        const newEdge = { source: edge.source, target: edge.target, label: 0, modified: true };
        workingGraph.edges.push(newEdge);
        modifications.value.push({ type: 'add', ...newEdge });
        }
    });
    };

    // --- Saving and Navigation ---
    const saveGeneratedGraph = async (name, page, g) => {
    try { await fetch(`${import.meta.env.VITE_BACKEND_URL}/save-graph/${name}/${page}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ graph: g }) }); }
    catch (e) { console.error('Error saving generated graph:', e); }
    };
    const saveModifications = async () => {
    try {
        const res = await fetch(`${import.meta.env.VITE_BACKEND_URL}/semi-segment/${manuscriptName.value}/${currentPage.value}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ graph: workingGraph, modifications: modifications.value }) });
        if (!res.ok) throw new Error((await res.json()).error || 'Save failed');
        const data = await res.json();
        graph.value = JSON.parse(JSON.stringify(workingGraph));
        modifications.value = [];
        if (data.lines) annotationStore.recognitions[manuscriptName.value][currentPage.value] = data.lines;
        error.value = null;
    } catch (err) {
        error.value = err.message;
        throw err;
    }
    };
    const saveCurrentGraph = async () => {
    if (isProcessingSave.value) return;
    isProcessingSave.value = true;
    try { await saveModifications(); alert("Graph saved!"); }
    catch (err) { alert(`Save failed: ${err.message}`); }
    finally { isProcessingSave.value = false; }
    };
    const confirmAndNavigate = async (navAction) => {
    if (isProcessingSave.value) return;
    if (modifications.value.length > 0) {
        isProcessingSave.value = true;
        try { await saveModifications(); navAction(); }
        catch (err) { alert("Save failed, navigation cancelled."); }
        finally { isProcessingSave.value = false; }
    } else {
        navAction();
    }
    };
    const nextPage = () => confirmAndNavigate(() => annotationStore.nextPage());
    const previousPage = () => confirmAndNavigate(() => annotationStore.previousPage());
    const goToIMG2TXTPage = () => confirmAndNavigate(() => router.push({ name: 'img-2-txt' }));
    const saveAndGoNext = async () => {
    if (loading.value || isProcessingSave.value) return;
    isProcessingSave.value = true;
    try { await saveModifications(); annotationStore.nextPage(); }
    catch (err) { alert(`Save failed: ${err.message}`); }
    finally { isProcessingSave.value = false; }
    };

    // --- Lifecycle and Watchers ---
    watch(() => annotationStore.currentPage, (newPage) => { if (newPage) fetchPageData(); }, { immediate: true });
    watch(editMode, (isEditing) => { if (!isEditing) { resetSelection(); isAKeyPressed.value = false; isDKeyPressed.value = false; hoveredNodesForMST.clear(); }});
    onMounted(() => { window.addEventListener('keydown', handleGlobalKeyDown); window.addEventListener('keyup', handleGlobalKeyUp); });
    onBeforeUnmount(() => { window.removeEventListener('keydown', handleGlobalKeyDown); window.removeEventListener('keyup', handleGlobalKeyUp); });

    </script>

    <style scoped>
    .manuscript-viewer {
    display: flex; flex-direction: column; height: 100vh;
    width: 100%; overflow: hidden; background-color: #333; color: #fff;
    }

    /* --- Toolbar --- */
    .toolbar {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 16px; background-color: #424242; border-bottom: 1px solid #555;
    flex-shrink: 0; gap: 16px;
    }
    .toolbar-controls {
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
    }

    /* --- Main Visualization Area --- */
    .visualization-container {
    position: relative; overflow: auto; flex-grow: 1;
    display: flex; justify-content: center; align-items: flex-start; padding: 1rem;
    }
    .image-container { position: relative; box-shadow: 0 0 15px rgba(0,0,0,0.5); }
    .manuscript-image { display: block; user-select: none; opacity: 0.7; }
    .graph-overlay {
    position: absolute; top: 0; left: 0;
    /* Improvement: Hide by default and control interaction */
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.2s ease-in-out;
    }
    .graph-overlay.is-visible {
    /* Improvement: Show only when edit mode is active */
    opacity: 1;
    pointer-events: auto;
    }

    /* --- Bottom Panel --- */
    .bottom-panel {
    background-color: #4f4f4f; border-top: 1px solid #555;
    flex-shrink: 0; transition: all 0.3s ease;
    }
    .panel-toggle-bar {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 16px; cursor: pointer;
    }
    .edit-instructions p { margin: 0; font-size: 0.9em; color: #ccc; font-style: italic; }
    .bottom-panel-content {
    padding: 10px 16px; display: flex; flex-direction: column; gap: 16px;
    }
    .edit-controls, .modifications-log-container {
    display: flex; align-items: flex-start; gap: 20px;
    }
    .edit-actions { display: flex; gap: 8px; }

    /* --- UI Elements & States --- */
    .panel-toggle-btn {
    padding: 4px 10px; font-size: 0.8em;
    background-color: #616161; border: 1px solid #757575;
    }
    .processing-save-notice, .loading, .error-message {
    position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
    padding: 20px 30px; border-radius: 8px; z-index: 10000; text-align: center;
    }
    .processing-save-notice { background-color: rgba(0, 0, 0, 0.8); }
    .error-message { background-color: #c62828; }
    .loading { font-size: 1.2rem; color: #aaa; background: none; }
    button {
    padding: 6px 14px; border-radius: 4px; border: 1px solid #666;
    background-color: #555; color: #fff; cursor: pointer; transition: background-color 0.2s ease;
    }
    button:hover:not(:disabled) { background-color: #6a6a6a; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }

    /* --- Modifications Log --- */
    .modifications-details { flex-grow: 1; }
    .modifications-details h3 { margin: 0 0 8px 0; font-size: 1.1em; color: #eee; }
    .modifications-details ul {
    list-style-type: none; padding: 0; max-height: 120px; overflow-y: auto;
    border: 1px solid #666; background-color: #3e3e3e; border-radius: 3px;
    }
    .modification-item {
    display: flex; justify-content: space-between; align-items: center;
    padding: 6px 10px; border-bottom: 1px solid #555; font-size: 0.9em;
    }
    .modification-item:last-child { border-bottom: none; }
    .undo-button { background-color: #6d6d3d; border-color: #888855; }
    .undo-button:hover:not(:disabled) { background-color: #7a7a4a; }
    </style>


    ================================================
    FILE: frontend/src/components/new-UploadForm.vue
    ================================================
    <script setup>
    import Dropzone from 'dropzone'
    import { ref, onMounted } from 'vue'
    import { useRouter } from 'vue-router'
    import { useAnnotationStore } from '@/stores/annotationStore'

    const annotationStore = useAnnotationStore()
    const uploadForm = ref()
    const manuscriptName = ref('') // Initialize to prevent undefined issues, bind with v-model
    const models = ref([])
    const modelSelected = ref('') // Initialize, bind with v-model

    const router = useRouter()

    fetch(import.meta.env.VITE_BACKEND_URL + '/models')
    .then((response) => response.json())
    .then((object) => {
        models.value = object
    })

    onMounted(() => {
    uploadForm.value = new Dropzone('#upload-form', {
        url: import.meta.env.VITE_BACKEND_URL + '/new-process-manuscript', // Set URL here for Dropzone
        uploadMultiple: true,
        autoProcessQueue: false, // We'll call processQueue manually
        parallelUploads: Infinity,
        // It's good practice to define acceptedFiles if you expect specific image types
        // acceptedFiles: 'image/jpeg,image/png,image/gif', // Example
    })

    uploadForm.value.on('completemultiple', function (files) {
        // This event fires after all files in a batch are processed (uploaded or failed).
        // We should ideally check file.status === Dropzone.SUCCESS for each file,
        // but for simplicity, we'll assume 'files' here are the ones intended for processing if the overall batch didn't error out.
        // If Dropzone's 'successmultiple' event is available and preferred, that could be used too.
        

        const currentManuscriptNameFromForm = manuscriptName.value
        const currentModelSelectedFromForm = modelSelected.value
        console.log('Page A: currentModelSelectedFromForm is:', currentModelSelectedFromForm); // DEBUG
        console.log('Page A: manuscriptName.value is:', manuscriptName.value); // DEBUG

        // 1. Basic validation: Ensure manuscript name and model are selected
        if (!currentManuscriptNameFromForm) {
        alert('Please enter a manuscript name.')
        console.error('Manuscript name not provided.')
        // Potentially clear the queue or re-enable the form
        // uploadForm.value.removeAllFiles(); // if you want to clear on error
        return
        }
        if (!currentModelSelectedFromForm) {
        alert('Please select a model.')
        console.error('Model not selected.')
        // uploadForm.value.removeAllFiles();
        return
        }
        if (files.length === 0) {
        alert('Please add files to upload.')
        console.error('No files in the upload queue for processing.')
        return
        }

        console.log('Upload complete. Backend responded. Now updating store from frontend data.')

        // 2. Update annotationStore
        annotationStore.reset(); // Optional: Reset store if starting a new manuscript processing session

        // Set the model name
        annotationStore.modelName = currentModelSelectedFromForm

        // Initialize recognitions for this manuscript
        annotationStore.recognitions[currentManuscriptNameFromForm] = {}

        // 3. Populate annotationStore.recognitions using uploaded file names as page IDs
        // Extract successfully uploaded file names. Dropzone's `files` in `completemultiple`
        // should be the list of files processed in that batch.
        const uploadedPageIds = files
        .filter(file => file.status === Dropzone.SUCCESS && file.name) // Ensure successful & has a name
        .map(file => file.name.split('.')[0]) // Use the original filename as the page ID
        .filter(pageId => pageId && pageId.trim() !== '')

        if (uploadedPageIds.length === 0 && files.length > 0) {
            console.warn("No files were successfully uploaded, or they lack names. Cannot set up pages.");
            // Potentially inform user more directly
            return;
        }
        if (uploadedPageIds.length === 0 && files.length === 0) {
            console.log("No files were in the queue to begin with.");
            return;
        }


        // Sort page IDs (filenames) to ensure consistent order
        // The store's `sortedPageIds` computed property will also sort,
        // but sorting here before insertion might be slightly cleaner if order of keys matters for non-display logic.
        // However, relying on `sortedPageIds` from the store is the canonical way.
        // For now, we just add them, and the store's computed property will handle sorting for navigation.
        uploadedPageIds.forEach(pageId => {
        // For each uploaded file, create an entry in recognitions.
        // Initially, there's no line data (predicted_label, etc.) because
        // the backend response is just "Hello world".
        // This structure prepares the store for these pages.
        // Line data would need to be fetched later or entered manually.
        annotationStore.recognitions[currentManuscriptNameFromForm][pageId] = {}
        })

        // 4. Update userAnnotations array
        annotationStore.userAnnotations.push({
        manuscript_name: currentManuscriptNameFromForm,
        selected_model: currentModelSelectedFromForm,
        annotations: {}, // User's ground truth annotations will go here
        })

        // 5. Set the initial page in the store
        // This will use the `sortedPageIds` computed property, which now knows about the pages
        // we just added from the uploaded filenames.
        annotationStore.setInitialPage()

        if (!annotationStore.currentPage) {
            console.warn("Current page could not be set. This might happen if no files were successfully processed as pages.");
            // Handle this case, maybe route to a different page or show an error.
        }

        // 6. Navigate to the annotation view
        console.log(`Navigating. Current page set to: ${annotationStore.currentPage}`)
        router.push({ name: 'new-semi-segment' })
    })

    // It's important that Dropzone knows where to send the files.
    // The <form :action="UPLOAD_URL"> is for non-JS submissions.
    // Dropzone needs its own `url` option, or it will use the form's action.
    // Ensure your Dropzone instance is configured with the correct upload URL.
    // If '#upload-form' already has an action attribute, Dropzone might pick it up.
    // Explicitly setting it in Dropzone options is safer:
    // new Dropzone('#upload-form', { url: UPLOAD_URL, ... })
    // The code above already includes it.
    })

    // UPLOAD_URL is used by the button click handler with processQueue, Dropzone uses its 'url' option.
    // const UPLOAD_URL = import.meta.env.VITE_BACKEND_URL + '/new-process-manuscript'; // Already defined
    </script>

    <template>
    <div class="mb-3">
        <label for="manuscriptName" class="form-label">Manuscript Name</label>
        <input type="text" class="form-control" id="manuscriptName" v-model="manuscriptName" />
    </div>
    <div class="mb-3">
        <label for="model" class="form-label">Model</label>
        <select class="form-select" id="model" v-model="modelSelected"> <!-- Removed placeholder for v-model -->
        <option disabled value="">Select a model</option> <!-- Default disabled option -->
        <option v-for="model in models" :key="model" :value="model">{{ model }}</option>
        </select>
    </div>
    <!--
        The action attribute on the form is a fallback or can be used by Dropzone if `url` option isn't set.
        For Dropzone, the `url` option specified during initialization is primary.
        The hidden inputs for manuscript_name and model are still useful as Dropzone will include them as form data.
    -->
    <form :action="UPLOAD_URL" class="dropzone" id="upload-form">
        <div class="dz-message" data-dz-message><span>Drop files here or click to upload.</span></div>
        <div class="previews"></div> <!-- Dropzone might use this for previews if configured -->
        <input type="hidden" name="manuscript_name" :value="manuscriptName" />
        <input type="hidden" name="model" :value="modelSelected" />
    </form>
    <button @click="uploadForm.processQueue()" class="btn btn-primary mt-3"
            :disabled="!manuscriptName || !modelSelected"> <!-- Disable button if critical info missing -->
        Submit
    </button>
    </template>

    <style>
    form.dropzone {
    background-color: var(--bs-body-bg);
    border: var(--bs-border-width) solid var(--bs-border-color);
    color: var(--bs-body-color);
    font-family: inherit;
    min-height: 150px; /* Give dropzone some default height */
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    }

    /* Basic styling for Dropzone message if you use the default one */
    .dz-message {
    text-align: center;
    margin: 2em 0;
    }
    </style>


    ================================================
    FILE: frontend/src/components/PROMPT-new-SemiSegmentationSection.txt
    ================================================
    The follow is the code of a .vue annotation tool which does the following:
    This Annotation tool is used for annotating the layout of Historical Manuscripts, which handwritten, curvy line, messy footnotes etc. 
    To do this, an upstream application already marks each character present on the manusscript page as a node, and connects each node (character) to it's previous and next characters.
    Hence for each page the code loads a graph, with nodes representing a character, and edges connecting a character to it's previous and next characters.
    Ideally all nodes of a text-line have to be connected together with edges. 
    But sometimes the upstream algorith makes mistakes, requiring us to annotate - by adding or deleting edges.
    Once annotation is done, we can save the updated_graph.
    The tool also has other downstream applications such as Annnotate text.

    Primarily right now, I want your help in improving the User Experience (UX) of addition and deletion of edges. PLease make the following changes
    - when I press and hold 'd' all edges I HOVER over get deleted
    - When I press and hold 'a' , all the nodes I HOVER over, get connected with edges using a Minimum Spanning Tree
    - Have a threshold, such that I won't need to HOVER exactly over the nodes to join, it's okay if I just hover near them (but still close)
    This make require making extensive changes to the code. Please make them, but make sure all other functions apart from the UX remain the same.
    I should be able to use the new code a drop in replacement.
    - minor change: also all me to save the graph, even if I have not made any modifications. 
    - replace buttons:  Show Points, Show Graph, Edit Mode! with only Edit Mode (we can show graph and points when Edit Mode is on). Edit Mode will be ON by default.
    


    ================================================
    FILE: frontend/src/components/archive/AnnotationBlock.vue
    ================================================
    <script setup>
    import { reactive, ref, watch, onMounted } from 'vue'
    import Sanscript from '@indic-transliteration/sanscript'
    import { useAnnotationStore } from '@/stores/annotationStore'
    import { handleInput } from '../typing-utils/devanagariInputUtils'  // Import the new utility function

    const BASE_PATH = `${import.meta.env.VITE_BACKEND_URL}/line-images`

    const props = defineProps(['line_name', 'line_data', 'page_name', 'manuscript_name'])
    const annotationStore = useAnnotationStore()

    const isHK = ref(false)

    const textboxClassObject = reactive({
    'form-control': true,
    'mb-2': true,
    'me-2': true,
    'devanagari-textbox': true,
    'is-valid': false,
    })

    const devanagari = ref(props.line_data.predicted_label)
    const hk = ref(Sanscript.t(props.line_data.predicted_label, 'devanagari', 'hk'))

    const devanagariInput = ref(null)

    watch(hk, function () {
    if (!isHK.value) return
    devanagari.value = Sanscript.t(hk.value, 'hk', 'devanagari')
    })

    function toggleHK() {
    hk.value = Sanscript.t(devanagari.value, 'devanagari', 'hk')
    isHK.value = !isHK.value
    }

    function save() {
    annotationStore.userAnnotations[0]['annotations'][props.page_name][props.line_name] = {}
    annotationStore.userAnnotations[0]['annotations'][props.page_name][props.line_name][
        'ground_truth'
    ] = devanagari.value
    textboxClassObject['is-valid'] = true
    }


    const boundHandleInput = (event) => handleInput(event, devanagari)

    onMounted(() => {
    if (devanagariInput.value) {
        devanagariInput.value.addEventListener('keydown', boundHandleInput)
    }
    })
    </script>

    <template>
    <img
        :src="`${BASE_PATH}/${props.manuscript_name}/${props.page_name}/${props.line_name}`"
        class="mb-2 manuscript-segment-img"
    />
    <div class="annotation-input">
        <input 
        ref="devanagariInput"
        v-model="devanagari" 
        type="text" 
        :class="textboxClassObject" 
        />
        <button class="btn btn-primary mb-2 me-2" @click="toggleHK">Roman</button>
        <button class="btn btn-success mb-2 me-2" @click="save">Save</button>
    </div>
    <input v-model="hk" type="text" class="form-control mb-2" v-if="isHK" />
    </template>

    <style>
    .manuscript-segment-img {
    display: block;
    }

    .annotation-input {
    width: 100%;
    display: flex;
    }

    .devanagari-textbox {
    flex-grow: 1;
    display: inline-block;
    }
    </style>


    ================================================
    FILE: frontend/src/components/archive/AnnotationPage.vue
    ================================================
    <script setup>
    import { useAnnotationStore } from '@/stores/annotationStore'
    import AnnotationBlock from './AnnotationBlock.vue'

    const props = defineProps(['data', 'page_name', 'manuscript_name'])
    const annotationStore = useAnnotationStore()

    annotationStore.userAnnotations[0]['annotations'][props.page_name] = {}
    </script>

    <template>
    <div>
        <div v-for="(line_data, line_name) in props.data" :key="line_name">
        <AnnotationBlock
            :line_name="line_name"
            :line_data="line_data"
            :page_name="props.page_name"
            :manuscript_name="props.manuscript_name"
        />
        </div>
    </div>
    </template>



    ================================================
    FILE: frontend/src/components/archive/AnnotationSection.vue
    ================================================
    <script setup>
    import { useRouter } from 'vue-router'
    import AnnotationPage from '@/components/archive/AnnotationPage.vue'
    import { useAnnotationStore } from '@/stores/annotationStore'
    import CharacterPalette from '../typing-utils/characterPalette.vue'

    const router = useRouter()
    const annotationStore = useAnnotationStore()
    const manuscript_name = Object.keys(annotationStore.recognitions)[0]
    annotationStore.currentPage = Object.keys(annotationStore.recognitions[manuscript_name])[0]

    function uploadGroundTruth() {
    annotationStore.calculateLevenshteinDistances()
    annotationStore.userAnnotations.forEach((elem) => {
        elem['model_name'] = annotationStore.modelName
        console.log('added Model name', annotationStore.modelName)
    })
    fetch(import.meta.env.VITE_BACKEND_URL + '/fine-tune', {
        method: 'POST',
        headers: {
        'Content-Type': 'application/json',
        },
        body: JSON.stringify(annotationStore.userAnnotations),
    }).then(() => {
        annotationStore.reset()
        router.push({ name: 'upload-manuscript' })
    })
    }

    function switchToSemiAutoSegmentation() {
    router.push({ name: 'semi-segment' })
    }

    </script>

    <template>
    <div class="mb-3">
        <label for="model-name" class="form-label">Model name</label>
        <input
        class="form-control"
        placeholder="Name your model..."
        v-model="annotationStore.modelName"
        />
    </div>
    <div class="mb-3">
        <button class="btn btn-primary me-2" @click="uploadGroundTruth">Fine-tune</button>
        <!-- <button class="btn btn-warning me-2" @click="switchToSegmentation">Correct Image Segments</button> -->
        <button class="btn btn-warning me-2" @click="switchToSemiAutoSegmentation">Semi Automatic Segmentation</button>
        <button class="btn btn-success me-2" @click="annotationStore.exportToTxt">Export</button>
        <CharacterPalette />
    </div>
    <div class="mb-3">
        <label for="page" class="form-label">Page</label>
        <select
        class="form-select"
        id="page"
        v-model="annotationStore.currentPage"
        placeholder="Select a model"
        >
        <option
            v-for="(page_data, page_name) in annotationStore.recognitions[manuscript_name]"
            :key="page_name"
            :value="page_name"
        >
            {{ page_name }}
        </option>
        </select>
    </div>
    <AnnotationPage
        v-for="(page_data, page_name) in annotationStore.recognitions[manuscript_name]"
        :key="page_data"
        :data="page_data"
        :page_name="page_name"
        :manuscript_name="manuscript_name"
        v-show="annotationStore.currentPage === page_name"
    />
    </template>



    ================================================
    FILE: frontend/src/components/archive/LayoutGraphGenerator_backup.js
    ================================================
    // layoutGraphGenerator.js
    /**
    * Build a KD-Tree for fast neighbor lookup
    */
    class KDTree {
    constructor(points) {
        this.points = points;
        this.tree = this.buildTree(points.map((p, i) => ({ point: p, index: i })), 0);
    }

    buildTree(points, depth) {
        if (points.length === 0) return null;
        if (points.length === 1) return points[0];

        const k = 2; // 2D points
        const axis = depth % k;
        
        points.sort((a, b) => a.point[axis] - b.point[axis]);
        const median = Math.floor(points.length / 2);
        
        return {
        point: points[median].point,
        index: points[median].index,
        left: this.buildTree(points.slice(0, median), depth + 1),
        right: this.buildTree(points.slice(median + 1), depth + 1),
        axis: axis
        };
    }

    query(queryPoint, k) {
        const best = [];
        
        const search = (node, depth) => {
        if (!node) return;
        
        const distance = this.euclideanDistance(queryPoint, node.point);
        
        if (best.length < k) {
            best.push({ distance, index: node.index });
            best.sort((a, b) => a.distance - b.distance);
        } else if (distance < best[best.length - 1].distance) {
            best[best.length - 1] = { distance, index: node.index };
            best.sort((a, b) => a.distance - b.distance);
        }
        
        const axis = depth % 2;
        const diff = queryPoint[axis] - node.point[axis];
        
        const closer = diff < 0 ? node.left : node.right;
        const farther = diff < 0 ? node.right : node.left;
        
        search(closer, depth + 1);
        
        if (best.length < k || Math.abs(diff) < best[best.length - 1].distance) {
            search(farther, depth + 1);
        }
        };
        
        search(this.tree, 0);
        return best.map(b => b.index);
    }

    euclideanDistance(p1, p2) {
        return Math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2);
    }
    }

    /**
    * DBSCAN clustering implementation to identify majority cluster and outliers
    */
    function clusterWithSingleMajority(toCluster, eps = 10, minSamples = 2) {
    if (toCluster.length === 0) return [];
    
    // DBSCAN implementation
    const labels = dbscan(toCluster, eps, minSamples);
    
    // Count the occurrences of each label
    const labelCounts = {};
    labels.forEach(label => {
        labelCounts[label] = (labelCounts[label] || 0) + 1;
    });
    
    // Find the majority cluster label (excluding -1 outliers)
    let majorityLabel = null;
    let maxCount = 0;
    
    for (const [label, count] of Object.entries(labelCounts)) {
        const labelNum = parseInt(label);
        if (labelNum !== -1 && count > maxCount) {
        majorityLabel = labelNum;
        maxCount = count;
        }
    }
    
    // Create a new label array where the majority cluster is 0 and all others are -1
    const newLabels = new Array(labels.length).fill(-1); // Initialize all as outliers
    
    if (majorityLabel !== null) {
        for (let i = 0; i < labels.length; i++) {
        if (labels[i] === majorityLabel) {
            newLabels[i] = 0; // Assign 0 to the majority cluster
        }
        }
    }
    
    return newLabels;
    }

    /**
    * DBSCAN clustering algorithm implementation
    */
    function dbscan(points, eps, minSamples) {
    const labels = new Array(points.length).fill(-1); // -1 means unclassified
    let clusterId = 0;
    
    for (let i = 0; i < points.length; i++) {
        if (labels[i] !== -1) continue; // Already processed
        
        const neighbors = getNeighbors(points, i, eps);
        
        if (neighbors.length < minSamples) {
        labels[i] = -1; // Mark as noise/outlier
        } else {
        // Start a new cluster
        expandCluster(points, labels, i, neighbors, clusterId, eps, minSamples);
        clusterId++;
        }
    }
    
    return labels;
    }

    /**
    * Get neighbors within eps distance
    */
    function getNeighbors(points, pointIndex, eps) {
    const neighbors = [];
    const point = points[pointIndex];
    
    for (let i = 0; i < points.length; i++) {
        if (euclideanDistance(point, points[i]) <= eps) {
        neighbors.push(i);
        }
    }
    
    return neighbors;
    }

    /**
    * Expand cluster by adding density-reachable points
    */
    function expandCluster(points, labels, pointIndex, neighbors, clusterId, eps, minSamples) {
    labels[pointIndex] = clusterId;
    
    let i = 0;
    while (i < neighbors.length) {
        const neighborIndex = neighbors[i];
        
        if (labels[neighborIndex] === -1) {
        labels[neighborIndex] = clusterId;
        
        const neighborNeighbors = getNeighbors(points, neighborIndex, eps);
        if (neighborNeighbors.length >= minSamples) {
            // Add new neighbors to the list (union operation)
            for (const newNeighbor of neighborNeighbors) {
            if (!neighbors.includes(newNeighbor)) {
                neighbors.push(newNeighbor);
            }
            }
        }
        }
        
        i++;
    }
    }

    function euclideanDistance(p1, p2) {
    return Math.sqrt(p1.reduce((sum, val, i) => sum + (val - p2[i]) ** 2, 0));
    }

    /**
    * Generate a graph representation of text layout based on points.
    * This function implements the core layout analysis logic.
    */
    export function generateLayoutGraph(points) { // TODO ADD FEATURES
    const NUM_NEIGHBOURS = 6;
    const cos_similarity_less_than = -0.8;
    
    // Build a KD-tree for fast neighbor lookup
    const tree = new KDTree(points);
    const indices = points.map((point, i) => tree.query(point, NUM_NEIGHBOURS));
    
    // Store graph edges and their properties
    const edges = [];
    const edgeProperties = [];
    
    // Process nearest neighbors
    for (let currentPointIndex = 0; currentPointIndex < indices.length; currentPointIndex++) {
        const nbrIndices = indices[currentPointIndex];
        const currentPoint = points[currentPointIndex];
        
        const normalizedPoints = nbrIndices.map(idx => [
        points[idx][0] - currentPoint[0],
        points[idx][1] - currentPoint[1]
        ]);
        
        const scalingFactor = Math.max(...normalizedPoints.flat().map(Math.abs)) || 1;
        const scaledPoints = normalizedPoints.map(np => [np[0] / scalingFactor, np[1] / scalingFactor]);
        
        // Create a list of relative neighbors with their global indices
        const relativeNeighbours = nbrIndices.map((globalIdx, i) => ({
        globalIdx,
        scaledPoint: scaledPoints[i],
        normalizedPoint: normalizedPoints[i]
        }));
        
        const filteredNeighbours = [];
        
        for (let i = 0; i < relativeNeighbours.length; i++) {
        for (let j = i + 1; j < relativeNeighbours.length; j++) {
            const neighbor1 = relativeNeighbours[i];
            const neighbor2 = relativeNeighbours[j];
            
            const norm1 = Math.sqrt(neighbor1.scaledPoint[0] ** 2 + neighbor1.scaledPoint[1] ** 2);
            const norm2 = Math.sqrt(neighbor2.scaledPoint[0] ** 2 + neighbor2.scaledPoint[1] ** 2);
            
            let cosSimilarity = 0.0;
            if (norm1 * norm2 !== 0) {
            const dotProduct = neighbor1.scaledPoint[0] * neighbor2.scaledPoint[0] + 
                            neighbor1.scaledPoint[1] * neighbor2.scaledPoint[1];
            cosSimilarity = dotProduct / (norm1 * norm2);
            }
            
            // Calculate non-normalized distances
            const norm1Real = Math.sqrt(neighbor1.normalizedPoint[0] ** 2 + neighbor1.normalizedPoint[1] ** 2);
            const norm2Real = Math.sqrt(neighbor2.normalizedPoint[0] ** 2 + neighbor2.normalizedPoint[1] ** 2);
            const totalLength = norm1Real + norm2Real;
            
            // Select pairs with angles close to 180 degrees (opposite directions)
            if (cosSimilarity < cos_similarity_less_than) {
            filteredNeighbours.push({
                neighbor1,
                neighbor2,
                totalLength,
                cosSimilarity
            });
            }
        }
        }
        
        if (filteredNeighbours.length > 0) {
        // Find the shortest total length pair
        const shortestPair = filteredNeighbours.reduce((min, curr) => 
            curr.totalLength < min.totalLength ? curr : min
        );
        
        const { neighbor1: connection1, neighbor2: connection2, totalLength, cosSimilarity } = shortestPair;
        
        // Calculate angles with x-axis
        const thetaA = Math.atan2(connection1.normalizedPoint[1], connection1.normalizedPoint[0]) * 180 / Math.PI;
        const thetaB = Math.atan2(connection2.normalizedPoint[1], connection2.normalizedPoint[0]) * 180 / Math.PI;
        
        // Add edges to the graph
        edges.push([currentPointIndex, connection1.globalIdx]);
        edges.push([currentPointIndex, connection2.globalIdx]);
        
        // Calculate feature values for clustering
        const yDiff1 = Math.abs(connection1.normalizedPoint[1]);
        const yDiff2 = Math.abs(connection2.normalizedPoint[1]);
        const avgYDiff = (yDiff1 + yDiff2) / 2;
        
        const xDiff1 = Math.abs(connection1.normalizedPoint[0]);
        const xDiff2 = Math.abs(connection2.normalizedPoint[0]);
        const avgXDiff = (xDiff1 + xDiff2) / 2;
        
        // Calculate aspect ratio (height/width)
        const aspectRatio = avgYDiff / Math.max(avgXDiff, 0.001);
        
        // Calculate vertical alignment consistency
        const vertConsistency = Math.abs(yDiff1 - yDiff2);
        
        // Store edge properties for clustering
        edgeProperties.push([
            totalLength,
            Math.abs(thetaA + thetaB),
            // aspectRatio,
            // vertConsistency,
            // avgYDiff
        ]);
        }
    }
    
    // Cluster the edges based on their properties
    const edgeLabels = clusterWithSingleMajority(edgeProperties);
    
    // Create a mask for edges that are not outliers (label != -1)
    const nonOutlierMask = edgeLabels.map(label => label !== -1);
    
    // Prepare the final graph structure
    const graphData = {
        nodes: points.map((point, i) => ({
        id: i,
        x: parseFloat(point[0]),
        y: parseFloat(point[1]),
        s: parseFloat(point[2]),
        })),
        edges: []
    };
    
    // Add edges with their labels, filtering out outliers
    for (let i = 0; i < edges.length; i++) {
        const edge = edges[i];
        // Determine the corresponding edge label using division by 2 (each edge appears twice)
        const labelIndex = Math.floor(i / 2);
        const edgeLabel = edgeLabels[labelIndex];
        
        // Only add the edge if it is not an outlier
        if (nonOutlierMask[labelIndex]) {
        graphData.edges.push({
            source: parseInt(edge[0]),
            target: parseInt(edge[1]),
            label: parseInt(edgeLabel)
        });
        }
    }
    
    return graphData;
    }


    ================================================
    FILE: frontend/src/components/archive/SemiSegmentationSection.vue
    ================================================
    <template>
    <div class="manuscript-viewer">
        <div class="toolbar">
        <h2>{{ manuscriptName }} - Page {{ currentPage }}</h2>
        <div class="controls">
            <button @click="previousPage" :disabled="loading">Previous</button>
            <button @click="nextPage" :disabled="loading">Next</button>
            <div class="toggle-container">
            <label>
                <input type="checkbox" v-model="showPoints" />
                Show Points
            </label>
            <label>
                <input type="checkbox" v-model="showGraph" />
                Show Graph
            </label>
            <label>
                <input type="checkbox" v-model="editMode" />
                Edit Mode
            </label>
            </div>
        </div>
        </div>

        <div v-if="error" class="error-message">
        {{ error }}
        </div>

        <div v-if="loading" class="loading">
        Loading page data...
        </div>

        <div v-else class="visualization-container" ref="container">
        <div class="image-container" :style="{ width: `${scaledWidth}px`, height: `${scaledHeight}px` }">
            <!-- Background image -->
            <img 
            v-if="imageData" 
            :src="`data:image/jpeg;base64,${imageData}`" 
            :width="scaledWidth" 
            :height="scaledHeight" 
            class="manuscript-image"
            @load="imageLoaded = true"
            />
            <div v-else class="placeholder-image" :style="{ width: `${scaledWidth}px`, height: `${scaledHeight}px` }">
            No image available
            </div>
            
            <!-- Points overlay -->
            <div 
            v-if="showPoints && points.length > 0" 
            class="points-overlay"
            >
            <div 
                v-for="(point, index) in points" 
                :key="`point-${index}`"
                class="point"
                :style="{
                left: `${scaleX(point.coordinates[0])}px`,
                top: `${scaleY(point.coordinates[1])}px`
                }"
                :title="`Point ${index}: (${point.coordinates[0]}, ${point.coordinates[1]})`"
            ></div>
            </div>
            
            <!-- Graph overlay -->
            <svg 
            v-if="showGraph && workingGraph.nodes && workingGraph.edges" 
            class="graph-overlay"
            :width="scaledWidth"
            :height="scaledHeight"
            @click="editMode && onBackgroundClick"
            >
            <!-- Edges -->
            <line
                v-for="(edge, index) in workingGraph.edges"
                :key="`edge-${index}`"
                :x1="scaleX(workingGraph.nodes[edge.source].x)"
                :y1="scaleY(workingGraph.nodes[edge.source].y)"
                :x2="scaleX(workingGraph.nodes[edge.target].x)"
                :y2="scaleY(workingGraph.nodes[edge.target].y)"
                :stroke="getEdgeColor(edge)"
                :stroke-width="isEdgeSelected(edge) ? 3 : 1.5"
                :stroke-opacity="0.7"
                @click="editMode && onEdgeClick(edge, $event)"
            />
            
            <!-- Nodes -->
            <circle
                v-for="(node, index) in workingGraph.nodes"
                :key="`node-${index}`"
                :cx="scaleX(node.x)"
                :cy="scaleY(node.y)"
                :r="isNodeSelected(index) ? 6 : 3"
                :fill="isNodeSelected(index) ? '#ff9500' : '#f44336'"
                :fill-opacity="0.7"
                @click="editMode && onNodeClick(index, $event)"
            />
            
            <!-- Selection line (when one node is selected) -->
            <line
                v-if="editMode && selectedNodes.length === 1 && tempEndPoint"
                :x1="scaleX(workingGraph.nodes[selectedNodes[0]].x)"
                :y1="scaleY(workingGraph.nodes[selectedNodes[0]].y)"
                :x2="tempEndPoint.x"
                :y2="tempEndPoint.y"
                stroke="#ff9500"
                stroke-width="1.5"
                stroke-dasharray="5,5"
                stroke-opacity="0.7"
            />
            </svg>
        </div>
        </div>

        <div v-if="editMode" class="edit-controls">
        <div class="edit-instructions">
            <p v-if="selectedNodes.length === 0">Select first node to create/delete edge</p>
            <p v-else-if="selectedNodes.length === 1">Select second node to create/delete edge</p>
            <p v-else>Click "Add Edge" or "Delete Edge" below</p>
        </div>
        <div class="edit-actions">
            <button @click="resetSelection">Cancel</button>
            <button 
            @click="addEdge" 
            :disabled="selectedNodes.length !== 2 || edgeExists(selectedNodes[0], selectedNodes[1])"
            >Add Edge</button>
            <button 
            @click="deleteEdge" 
            :disabled="selectedNodes.length !== 2 || !edgeExists(selectedNodes[0], selectedNodes[1])"
            >Delete Edge</button>
        </div>
        </div>

        <div v-if="modifications.length > 0" class="modifications-log">
            <h3>Modifications ({{ modifications.length }})</h3>
            <button @click="saveModifications">Save Changes</button>
            <button @click="resetModifications">Reset All</button>
            <ul>
            <li v-for="(mod, index) in modifications" :key="index" class="modification-item">
                {{ mod.type === 'add' ? 'Added' : 'Removed' }} edge between Node {{ mod.source }} and Node {{ mod.target }}
                <button @click="undoModification(index)" class="undo-button">Undo</button>
            </li>
            </ul>
        </div>
    </div>
    </template>

    <script setup>
    import { ref, onMounted, onBeforeUnmount, onUnmounted, computed, watch, reactive } from 'vue';
    import { useAnnotationStore } from '@/stores/annotationStore';
    import { generateLayoutGraph } from '../layout-analysis-utils/LayoutGraphGenerator.js';  // Import the new utility function

    const handleKeydown = (e) => {
    if (!editMode.value) return;
    if (e.key === 'a') addEdge();
    if (e.key === 'd') deleteEdge();
    };

    onMounted(() => window.addEventListener('keydown', handleKeydown));
    onBeforeUnmount(() => window.removeEventListener('keydown', handleKeydown));

    const annotationStore = useAnnotationStore();

    const manuscriptName = computed(() => Object.keys(annotationStore.recognitions)[0] || '');
    const currentPage = computed(() => annotationStore.currentPage);

    const loading = ref(true);
    const error = ref(null);
    const dimensions = ref([0, 0]);
    const points = ref([]);
    const graph = ref({ nodes: [], edges: [] });
    const imageData = ref('');
    const imageLoaded = ref(false);
    const showPoints = ref(false);
    const showGraph = ref(true);

    // Editing state
    const editMode = ref(true);
    const selectedNodes = ref([]);
    const tempEndPoint = ref(null);
    const modifications = ref([]);
    const workingGraph = reactive({ nodes: [], edges: [] });

    // Scale factor (similar to the Python code's resize)
    const scaleFactor = 0.7; // This is equivalent to dividing by 2 as in your Python code

    // Calculate scaled dimensions
    const scaledWidth = computed(() => Math.floor(dimensions.value[0] * scaleFactor));
    const scaledHeight = computed(() => Math.floor(dimensions.value[1] * scaleFactor));

    // Scale functions to map original coordinates to scaled view
    const scaleX = (x) => x * scaleFactor;
    const scaleY = (y) => y * scaleFactor;

    // Container ref for potential scrolling/zooming features
    const container = ref(null);

    const updateCanvasSize = (width, height) => {
    dimensions.value = [width, height];
    };

    // Function to save generated graph back to backend
    const saveGeneratedGraph = async (manuscriptName, page, graphData) => {
    try {
        console.log(`Saving graph for ${manuscriptName}, page ${page}`);
        const response = await fetch(
        import.meta.env.VITE_BACKEND_URL + `/save-graph/${manuscriptName}/${page}`,
        {
            method: 'POST',
            headers: {
            'Content-Type': 'application/json',
            },
            body: JSON.stringify({ graph: graphData })
        }
        );
        
        if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to save graph');
        }
        
        const result = await response.json();
        console.log('Graph saved to backend successfully:', result);
        return result;
    } catch (error) {
        console.error('Error saving graph to backend:', error);
        // Non-critical error, don't throw to avoid breaking the main flow
        return null;
    }
    };

    const fetchPageData = async () => {
    if (!manuscriptName.value || !currentPage.value) return;
    
    loading.value = true;
    error.value = null;
    points.value = [];
    graph.value = { nodes: [], edges: [] };
    imageData.value = '';
    imageLoaded.value = false;
    
    try {
        console.log(`Fetching data for manuscript: ${manuscriptName.value}, page: ${currentPage.value}`);
        const response = await fetch(
        import.meta.env.VITE_BACKEND_URL + `/semi-segment/${manuscriptName.value}/${currentPage.value}`
        );
        
        if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to fetch page data');
        }
        
        const data = await response.json();
        console.log("Received data:", Object.keys(data));
        
        // Update canvas size
        updateCanvasSize(data.dimensions[0], data.dimensions[1]);
        
        // Process points
        points.value = data.points.map(point => ({
        coordinates: [point[0], point[1]],
        segment: null,
        }));
        
        // Process graph
        if (data.graph) {
        // Graph was loaded from existing file on backend
        graph.value = data.graph;
        console.log("Using existing graph from backend");
        } else if (data.points && data.points.length > 0) {
        // No existing graph found, generate new one in frontend
        console.log("Generating new graph in frontend");
        try {
            const generatedGraph = generateLayoutGraph(data.points);
            graph.value = generatedGraph;
            console.log("Successfully generated graph:", generatedGraph);
            
            // Save the generated graph back to the backend
            console.log("Attempting to save generated graph...");
            const saveResult = await saveGeneratedGraph(manuscriptName.value, currentPage.value, generatedGraph);
            
            if (saveResult) {
            console.log("Graph saved successfully");
            } else {
            console.log("Graph generation successful but saving failed (non-critical)");
            }
            
        } catch (graphError) {
            console.error('Error generating graph:', graphError);
            // Fallback to empty graph if generation fails
            graph.value = { nodes: [], edges: [] };
        }
        }
        
        // Clone to working graph
        resetWorkingGraph();
        
        // Process image
        if (data.image) {
        console.log(`Loading image data, length: ${data.image.length}`);
        imageData.value = data.image;
        } else {
        console.log("No image data found in response");
        }
    } catch (err) {
        console.error('Error fetching page data:', err);
        error.value = err.message || 'Failed to load page data';
    } finally {
        loading.value = false;
    }
    };

    const resetWorkingGraph = () => {
    // Deep clone the original graph to working graph
    workingGraph.nodes = JSON.parse(JSON.stringify(graph.value.nodes || []));
    workingGraph.edges = JSON.parse(JSON.stringify(graph.value.edges || []));
    resetSelection();
    modifications.value = [];
    };

    const resetSelection = () => {
    selectedNodes.value = [];
    tempEndPoint.value = null;
    };

    const onNodeClick = (nodeIndex, event) => {
    event.stopPropagation();
    
    // If node is already selected, deselect it
    const existingIndex = selectedNodes.value.indexOf(nodeIndex);
    if (existingIndex !== -1) {
        selectedNodes.value.splice(existingIndex, 1);
        return;
    }
    
    // Add to selection (but limit to 2 nodes)
    if (selectedNodes.value.length < 2) {
        selectedNodes.value.push(nodeIndex);
    } else {
        // Replace selection if already have 2 nodes
        selectedNodes.value = [nodeIndex];
    }
    
    tempEndPoint.value = null;
    };

    const onEdgeClick = (edge, event) => {
    event.stopPropagation();
    
    // Select the nodes that form this edge
    selectedNodes.value = [edge.source, edge.target];
    };

    const onBackgroundClick = () => {
    resetSelection();
    };

    const edgeExists = (nodeA, nodeB) => {
    return workingGraph.edges.some(e => 
        (e.source === nodeA && e.target === nodeB) || 
        (e.source === nodeB && e.target === nodeA)
    );
    };

    const addEdge = () => {
    if (selectedNodes.value.length !== 2) return;
    
    const [source, target] = selectedNodes.value;
    
    // Check if edge already exists
    if (edgeExists(source, target)) {
        console.log('Edge already exists');
        return;
    }
    
    // Add edge to working graph
    workingGraph.edges.push({
        source,
        target,
        label: 0, // Default label for same-line connection
        modified: true
    });
    
    // Track modification
    modifications.value.push({
        type: 'add',
        source,
        target,
        label: 0
    });
    
    resetSelection();
    };

    const deleteEdge = () => {
    if (selectedNodes.value.length !== 2) return;
    
    const [source, target] = selectedNodes.value;
    
    // Find the edge index
    const edgeIndex = workingGraph.edges.findIndex(e => 
        (e.source === source && e.target === target) ||
        (e.source === target && e.target === source)
    );
    
    if (edgeIndex === -1) {
        console.log('Edge not found');
        return;
    }
    
    // Track modification before removing
    const removedEdge = workingGraph.edges[edgeIndex];
    modifications.value.push({
        type: 'delete',
        source: removedEdge.source,
        target: removedEdge.target,
        label: removedEdge.label
    });
    
    // Remove edge
    workingGraph.edges.splice(edgeIndex, 1);
    
    resetSelection();
    };

    const undoModification = (index) => {
    const mod = modifications.value[index];
    
    if (mod.type === 'add') {
        // Find and remove the added edge
        const edgeIndex = workingGraph.edges.findIndex(e => 
        (e.source === mod.source && e.target === mod.target) ||
        (e.source === mod.target && e.target === mod.source)
        );
        
        if (edgeIndex !== -1) {
        workingGraph.edges.splice(edgeIndex, 1);
        }
    } else if (mod.type === 'delete') {
        // Re-add the deleted edge
        workingGraph.edges.push({
        source: mod.source,
        target: mod.target,
        label: mod.label
        });
    }
    
    // Remove this modification from the list
    modifications.value.splice(index, 1);
    };

    const resetModifications = () => {
    resetWorkingGraph();
    };

    // #TODO add code to save the updated graph
    const isNodeSelected = (nodeIndex) => {
    return selectedNodes.value.includes(nodeIndex);
    };

    const isEdgeSelected = (edge) => {
    return selectedNodes.value.length === 2 &&
        ((selectedNodes.value[0] === edge.source && selectedNodes.value[1] === edge.target) ||
        (selectedNodes.value[0] === edge.target && selectedNodes.value[1] === edge.source));
    };

    const getEdgeColor = (edge) => {
    // Modified edges get a different color
    if (edge.modified) return '#f44336';
    // Original edge coloring logic
    return edge.label === 0 ? '#ffffff' : '#e74c3c';
    };

    const nextPage = async () => { // Make it async if saveModifications is async
    if (modifications.value.length > 0) {
        if (confirm('You have unsaved changes. Do you want to save them before moving to the next page?')) {
        try {
            await saveModifications(); // Assuming saveModifications is async and returns a Promise
            annotationStore.nextPage();
        } catch (err) {
            console.error("Failed to save, not navigating to next page:", err);
            // Optionally, inform the user about the save failure
        }
        } else {
        modifications.value = []; // Discard changes
        annotationStore.nextPage();
        }
    } else {
        annotationStore.nextPage();
    }
    };

    const previousPage = async () => { // Make it async
    if (modifications.value.length > 0) {
        if (confirm('You have unsaved changes. Do you want to save them before moving to the previous page?')) {
        try {
            await saveModifications();
            annotationStore.previousPage();
        } catch (err) {
            console.error("Failed to save, not navigating to previous page:", err);
        }
        } else {
        modifications.value = []; // Discard changes
        annotationStore.previousPage();
        }
    } else {
        annotationStore.previousPage();
    }
    };

    // Add mouse move handler for visualization when selecting nodes
    const handleMouseMove = (event) => {
    if (!editMode.value || selectedNodes.value.length !== 1) return;
    
    const rect = container.value.getBoundingClientRect();
    tempEndPoint.value = {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top
    };
    };

    // Watch for page changes
    watch(
    () => annotationStore.currentPage,
    (newPage, oldPage) => {
        if (newPage) {
        // Fetch data if the page ID actually changed, or if it's an initial load for this page
        // (e.g. newPage is set, but imageLoaded.value is false)
        console.log(`Component Watcher: currentPage changed from ${oldPage} to ${newPage}.`);
        fetchPageData(); // Your existing function to fetch page-specific details
        } else if (oldPage && !newPage) {
        // currentPage was cleared (e.g., after reset or no pages available)
        console.log("Component Watcher: currentPage became undefined. Clearing local data.");
        // Reset component's page-specific data
        points.value = [];
        graph.value = { nodes: [], edges: [] };
        imageData.value = '';
        imageLoaded.value = false;
        modifications.value = [];
        resetWorkingGraph(); // Your function to reset working graph
        loading.value = false; // Or true if you want to show a loading state for "no page"
        error.value = null;
        }
    },
    { immediate: true } // Crucial: runs the watcher handler immediately on component mount
    );

    // Watch for edit mode toggle
    watch(editMode, (newValue) => {
    if (newValue) {
        // Add mouse move listener when entering edit mode
        document.addEventListener('mousemove', handleMouseMove);
    } else {
        // Remove listener when leaving edit mode
        // It's safe to call removeEventListener even if the listener wasn't added for some reason
        document.removeEventListener('mousemove', handleMouseMove);
        resetSelection();
    }
    }, { immediate: true }); // <--- ADD THIS





    // Clean up
    onUnmounted(() => {
    document.removeEventListener('mousemove', handleMouseMove);
    });

    const saveModifications = async () => {
    try {
        console.log('Saving modifications and generating line labels...');
        
        // Prepare the request with the modified graph data
        const request = {
        graph: workingGraph,
        modifications: modifications.value,
        points: points.value.map(point => point.segment),
        modelName: annotationStore.modelName
        };
        
        const response = await fetch(
        import.meta.env.VITE_BACKEND_URL + 
        `/semi-segment/${manuscriptName.value}/${currentPage.value}`,
        {
            method: 'POST',
            headers: {
            'Content-Type': 'application/json',
            },
            body: JSON.stringify(request),
        }
        );
        
        if (!response.ok) {
        throw new Error('Failed to save modifications and generate labels');
        }
        
        // Update the original graph with the working graph
        graph.value = JSON.parse(JSON.stringify(workingGraph));
        modifications.value = [];
        
        console.log('Graph modifications saved and labels generated successfully');
    } catch (err) {
        console.error('Error saving modifications:', err);
        error.value = err.message || 'Failed to save modifications';
    }
    };



    </script>

    <style scoped>
    .manuscript-viewer {
    display: flex;
    flex-direction: column;
    height: 100%;
    width: 100%;
    }

    .toolbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px;
    background-color: #f5f5f5;
    border-bottom: 1px solid #ddd;
    }

    .controls {
    display: flex;
    align-items: center;
    gap: 12px;
    }

    .toggle-container {
    display: flex;
    gap: 8px;
    }

    .visualization-container {
    position: relative;
    overflow: auto;
    flex: 1;
    background-color: #eee;
    }

    .image-container {
    position: relative;
    margin: 0 auto;
    }

    .manuscript-image {
    display: block;
    }

    .placeholder-image {
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: #ddd;
    color: #666;
    }

    .points-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    }

    .point {
    position: absolute;
    width: 4px;
    height: 4px;
    background-color: rgba(255, 0, 0, 0.5);
    border-radius: 50%;
    transform: translate(-50%, -50%);
    }

    .graph-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    }

    .loading {
    padding: 20px;
    text-align: center;
    font-style: italic;
    color: #666;
    }

    .error-message {
    padding: 20px;
    background-color: #fee;
    color: #c00;
    border: 1px solid #faa;
    margin: 10px;
    border-radius: 4px;
    }

    /* Edit mode styling */
    .edit-controls {
    padding: 10px;
    background-color: #f9f9f9;
    border-bottom: 1px solid #ddd;
    }

    .edit-instructions {
    margin-bottom: 10px;
    font-size: 14px;
    color: #555;
    }

    .edit-actions {
    display: flex;
    gap: 8px;
    margin-bottom: 15px;
    }

    button {
    padding: 6px 12px;
    border-radius: 4px;
    border: 1px solid #ccc;
    background-color: #fff;
    cursor: pointer;
    }

    button:hover {
    background-color: #f0f0f0;
    }

    button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    }

    .modifications-log {
    border-top: 1px solid #ddd;
    padding-top: 10px;
    margin-top: 10px;
    }

    .modifications-log h3 {
    font-size: 16px;
    margin-bottom: 10px;
    }

    .modification-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid #eee;
    }

    .undo-button {
    font-size: 12px;
    padding: 2px 6px;
    }
    </style>


    ================================================
    FILE: frontend/src/components/archive/UploadForm.vue
    ================================================
    <script setup>
    import Dropzone from 'dropzone'

    import { ref, onMounted } from 'vue'
    import { useRouter } from 'vue-router'
    import { useAnnotationStore } from '@/stores/annotationStore'

    const annotationStore = useAnnotationStore()
    const uploadForm = ref()
    const manuscriptName = ref()
    const models = ref([])
    const modelSelected = ref('')

    const router = useRouter()

    fetch(import.meta.env.VITE_BACKEND_URL + '/models')
    .then((response) => response.json())
    .then((object) => {
        models.value = object
    })

    onMounted(() => {
    uploadForm.value = new Dropzone('#upload-form', {
        uploadMultiple: true,
        autoProcessQueue: false,
        parallelUploads: Infinity,
    })
    uploadForm.value.on('completemultiple', function (files) {
        const response = JSON.parse(files[0].xhr.response)
        const manuscript_name = Object.values(response)[0][0].manuscript_name
        const selected_model = Object.values(response)[0][0].selected_model
        annotationStore.recognitions[manuscript_name] = {}

        for (const page of Object.keys(response)) {
        annotationStore.recognitions[manuscript_name][page] = {}
        for (const line in response[page]) {
            const line_name = response[page][line]['line']
            annotationStore.recognitions[manuscript_name][page][line_name] = {}
            annotationStore.recognitions[manuscript_name][page][line_name]['predicted_label'] =
            response[page][line]['predicted_label']
            annotationStore.recognitions[manuscript_name][page][line_name]['image_path'] =
            response[page][line]['image_path']
            annotationStore.recognitions[manuscript_name][page][line_name]['confidence_score'] =
            response[page][line]['confidence_score']
        }
        }
        annotationStore.userAnnotations.push({
        manuscript_name: manuscript_name,
        selected_model: selected_model,
        annotations: {},
        })

        router.push({ name: 'annotation-section' })
    })
    })

    const UPLOAD_URL = import.meta.env.VITE_BACKEND_URL + '/upload-manuscript'

    </script>

    <template>
    <div class="mb-3">
        <label for="manuscriptName" class="form-label">Manuscript Name</label>
        <input type="text" class="form-control" id="manuscriptName" v-model="manuscriptName" />
    </div>
    <div class="mb-3">
        <label for="model" class="form-label">Model</label>
        <select class="form-select" id="model" v-model="modelSelected" placeholder="Select a model">
        <option disabled hidden value="">Select a model</option>
        <option v-for="model in models" :key="model" :value="model">{{ model }}</option>
        </select>
    </div>
    <form :action="UPLOAD_URL" class="dropzone" id="upload-form">
        <div class="previews"></div>
        <input type="hidden" name="manuscript_name" :value="manuscriptName" />
        <input type="hidden" name="model" :value="modelSelected" />
    </form>
    <button @click="uploadForm.processQueue()" class="btn btn-primary mt-3">Submit</button>
    </template>

    <style>

    form.dropzone {
    background-color: var(--bs-body-bg);
    border: var(--bs-border-width) solid var(--bs-border-color);
    color: var(--bs-body-color);
    font-family: inherit;
    }

    </style>


    ================================================
    FILE: frontend/src/components/layout-analysis-utils/LayoutGraphGenerator.js
    ================================================
    // layoutGraphGenerator.js
    /**
    * Build a KD-Tree for fast neighbor lookup
    */
    class KDTree {
    constructor(points) {
        this.points = points;
        this.tree = this.buildTree(points.map((p, i) => ({ point: p, index: i })), 0);
    }

    buildTree(points, depth) {
        if (points.length === 0) return null;
        if (points.length === 1) return points[0];

        const k = 2; // 2D points
        const axis = depth % k;
        
        points.sort((a, b) => a.point[axis] - b.point[axis]);
        const median = Math.floor(points.length / 2);
        
        return {
        point: points[median].point,
        index: points[median].index,
        left: this.buildTree(points.slice(0, median), depth + 1),
        right: this.buildTree(points.slice(median + 1), depth + 1),
        axis: axis
        };
    }

    query(queryPoint, k) {
        const best = [];
        
        const search = (node, depth) => {
        if (!node) return;
        
        const distance = this.euclideanDistance(queryPoint, node.point);
        
        if (best.length < k) {
            best.push({ distance, index: node.index });
            best.sort((a, b) => a.distance - b.distance);
        } else if (distance < best[best.length - 1].distance) {
            best[best.length - 1] = { distance, index: node.index };
            best.sort((a, b) => a.distance - b.distance);
        }
        
        const axis = depth % 2;
        const diff = queryPoint[axis] - node.point[axis];
        
        const closer = diff < 0 ? node.left : node.right;
        const farther = diff < 0 ? node.right : node.left;
        
        search(closer, depth + 1);
        
        if (best.length < k || Math.abs(diff) < best[best.length - 1].distance) {
            search(farther, depth + 1);
        }
        };
        
        search(this.tree, 0);
        return best.map(b => b.index);
    }

    euclideanDistance(p1, p2) {
        return Math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2);
    }
    }

    /**
    * DBSCAN clustering implementation to identify majority cluster and outliers
    */
    function clusterWithSingleMajority(toCluster, eps = 10, minSamples = 2) {
    if (toCluster.length === 0) return [];
    
    // DBSCAN implementation
    const labels = dbscan(toCluster, eps, minSamples);
    
    // Count the occurrences of each label
    const labelCounts = {};
    labels.forEach(label => {
        labelCounts[label] = (labelCounts[label] || 0) + 1;
    });
    
    // Find the majority cluster label (excluding -1 outliers)
    let majorityLabel = null;
    let maxCount = 0;
    
    for (const [label, count] of Object.entries(labelCounts)) {
        const labelNum = parseInt(label);
        if (labelNum !== -1 && count > maxCount) {
        majorityLabel = labelNum;
        maxCount = count;
        }
    }
    
    // Create a new label array where the majority cluster is 0 and all others are -1
    const newLabels = new Array(labels.length).fill(-1); // Initialize all as outliers
    
    if (majorityLabel !== null) {
        for (let i = 0; i < labels.length; i++) {
        if (labels[i] === majorityLabel) {
            newLabels[i] = 0; // Assign 0 to the majority cluster
        }
        }
    }
    
    return newLabels;
    }

    /**
    * DBSCAN clustering algorithm implementation
    */
    function dbscan(points, eps, minSamples) {
    const labels = new Array(points.length).fill(-1); // -1 means unclassified
    let clusterId = 0;
    
    for (let i = 0; i < points.length; i++) {
        if (labels[i] !== -1) continue; // Already processed
        
        const neighbors = getNeighbors(points, i, eps);
        
        if (neighbors.length < minSamples) {
        labels[i] = -1; // Mark as noise/outlier
        } else {
        // Start a new cluster
        expandCluster(points, labels, i, neighbors, clusterId, eps, minSamples);
        clusterId++;
        }
    }
    
    return labels;
    }

    /**
    * Get neighbors within eps distance
    */
    function getNeighbors(points, pointIndex, eps) {
    const neighbors = [];
    const point = points[pointIndex];
    
    for (let i = 0; i < points.length; i++) {
        if (euclideanDistance(point, points[i]) <= eps) {
        neighbors.push(i);
        }
    }
    
    return neighbors;
    }

    /**
    * Expand cluster by adding density-reachable points
    */
    function expandCluster(points, labels, pointIndex, neighbors, clusterId, eps, minSamples) {
    labels[pointIndex] = clusterId;
    
    let i = 0;
    while (i < neighbors.length) {
        const neighborIndex = neighbors[i];
        
        if (labels[neighborIndex] === -1) {
        labels[neighborIndex] = clusterId;
        
        const neighborNeighbors = getNeighbors(points, neighborIndex, eps);
        if (neighborNeighbors.length >= minSamples) {
            // Add new neighbors to the list (union operation)
            for (const newNeighbor of neighborNeighbors) {
            if (!neighbors.includes(newNeighbor)) {
                neighbors.push(newNeighbor);
            }
            }
        }
        }
        
        i++;
    }
    }

    function euclideanDistance(p1, p2) {
    return Math.sqrt(p1.reduce((sum, val, i) => sum + (val - p2[i]) ** 2, 0));
    }

    /**
    * Generate a graph representation of text layout based on points.
    * This function implements the core layout analysis logic.
    */
    export function generateLayoutGraph(points) { // TODO ADD FEATURES
    const NUM_NEIGHBOURS = 6;
    const cos_similarity_less_than = -0.8;
    
    // Build a KD-tree for fast neighbor lookup
    const tree = new KDTree(points);
    const indices = points.map((point, i) => tree.query(point, NUM_NEIGHBOURS));
    
    // Store graph edges and their properties
    const edges = [];
    const edgeProperties = [];
    
    // Process nearest neighbors
    for (let currentPointIndex = 0; currentPointIndex < indices.length; currentPointIndex++) {
        const nbrIndices = indices[currentPointIndex];
        const currentPoint = points[currentPointIndex];
        
        const normalizedPoints = nbrIndices.map(idx => [
        points[idx][0] - currentPoint[0],
        points[idx][1] - currentPoint[1]
        ]);
        
        const scalingFactor = Math.max(...normalizedPoints.flat().map(Math.abs)) || 1;
        const scaledPoints = normalizedPoints.map(np => [np[0] / scalingFactor, np[1] / scalingFactor]);
        
        // Create a list of relative neighbors with their global indices
        const relativeNeighbours = nbrIndices.map((globalIdx, i) => ({
        globalIdx,
        scaledPoint: scaledPoints[i],
        normalizedPoint: normalizedPoints[i]
        }));
        
        const filteredNeighbours = [];
        
        for (let i = 0; i < relativeNeighbours.length; i++) {
        for (let j = i + 1; j < relativeNeighbours.length; j++) {
            const neighbor1 = relativeNeighbours[i];
            const neighbor2 = relativeNeighbours[j];
            
            const norm1 = Math.sqrt(neighbor1.scaledPoint[0] ** 2 + neighbor1.scaledPoint[1] ** 2);
            const norm2 = Math.sqrt(neighbor2.scaledPoint[0] ** 2 + neighbor2.scaledPoint[1] ** 2);
            
            let cosSimilarity = 0.0;
            if (norm1 * norm2 !== 0) {
            const dotProduct = neighbor1.scaledPoint[0] * neighbor2.scaledPoint[0] + 
                            neighbor1.scaledPoint[1] * neighbor2.scaledPoint[1];
            cosSimilarity = dotProduct / (norm1 * norm2);
            }
            
            // Calculate non-normalized distances
            const norm1Real = Math.sqrt(neighbor1.normalizedPoint[0] ** 2 + neighbor1.normalizedPoint[1] ** 2);
            const norm2Real = Math.sqrt(neighbor2.normalizedPoint[0] ** 2 + neighbor2.normalizedPoint[1] ** 2);
            const totalLength = norm1Real + norm2Real;
            
            // Select pairs with angles close to 180 degrees (opposite directions)
            if (cosSimilarity < cos_similarity_less_than) {
            filteredNeighbours.push({
                neighbor1,
                neighbor2,
                totalLength,
                cosSimilarity
            });
            }
        }
        }
        
        if (filteredNeighbours.length > 0) {
        // Find the shortest total length pair
        const shortestPair = filteredNeighbours.reduce((min, curr) => 
            curr.totalLength < min.totalLength ? curr : min
        );
        
        const { neighbor1: connection1, neighbor2: connection2, totalLength, cosSimilarity } = shortestPair;
        
        // Calculate angles with x-axis
        const thetaA = Math.atan2(connection1.normalizedPoint[1], connection1.normalizedPoint[0]) * 180 / Math.PI;
        const thetaB = Math.atan2(connection2.normalizedPoint[1], connection2.normalizedPoint[0]) * 180 / Math.PI;
        
        // Add edges to the graph
        edges.push([currentPointIndex, connection1.globalIdx]);
        edges.push([currentPointIndex, connection2.globalIdx]);
        
        // Calculate feature values for clustering
        const yDiff1 = Math.abs(connection1.normalizedPoint[1]);
        const yDiff2 = Math.abs(connection2.normalizedPoint[1]);
        const avgYDiff = (yDiff1 + yDiff2) / 2;
        
        const xDiff1 = Math.abs(connection1.normalizedPoint[0]);
        const xDiff2 = Math.abs(connection2.normalizedPoint[0]);
        const avgXDiff = (xDiff1 + xDiff2) / 2;
        
        // Calculate aspect ratio (height/width)
        const aspectRatio = avgYDiff / Math.max(avgXDiff, 0.001);
        
        // Calculate vertical alignment consistency
        const vertConsistency = Math.abs(yDiff1 - yDiff2);
        
        // Store edge properties for clustering
        edgeProperties.push([
            totalLength,
            Math.abs(thetaA + thetaB),
            // aspectRatio,
            // vertConsistency,
            // avgYDiff
        ]);
        }
    }
    
    // Cluster the edges based on their properties
    const edgeLabels = clusterWithSingleMajority(edgeProperties);
    
    // Create a mask for edges that are not outliers (label != -1)
    const nonOutlierMask = edgeLabels.map(label => label !== -1);
    
    // Prepare the final graph structure
    const graphData = {
        nodes: points.map((point, i) => ({
        id: i,
        x: parseFloat(point[0]),
        y: parseFloat(point[1]),
        s: parseFloat(point[2]),
        })),
        edges: []
    };
    
    // Add edges with their labels, filtering out outliers
    for (let i = 0; i < edges.length; i++) {
        const edge = edges[i];
        // Determine the corresponding edge label using division by 2 (each edge appears twice)
        const labelIndex = Math.floor(i / 2);
        const edgeLabel = edgeLabels[labelIndex];
        
        // Only add the edge if it is not an outlier
        if (nonOutlierMask[labelIndex]) {
        graphData.edges.push({
            source: parseInt(edge[0]),
            target: parseInt(edge[1]),
            label: parseInt(edgeLabel)
        });
        }
    }
    
    return graphData;
    }


    ================================================
    FILE: frontend/src/components/typing-utils/characterPalette.vue
    ================================================
    <script setup>
    import { ref } from 'vue'

    const togglePalette = ref(false)
    const copied = ref()

    function copyToClipboard(char) {
    navigator.clipboard
        .writeText(char)
        .then(() => {
        copied.value = char; 
        })
        .catch((err) => {
        console.error('Failed to copy:', err)
        })
    }

    const characters = [
    'ऀ',
    'ऄ',
    'ऎ',
    'ऒ',
    'ऴ',
    'ऺ',
    'ऻ',
    'ऽ',
    'ॆ',
    'ॊ',
    'ॎ',
    'ॏ',
    '॑',
    '॒',
    'ॕ',
    'ॖ',
    'ॗ',
    'ऌ',
    'ॡ',
    'ॢ',
    'ॢ',
    '॥',
    '॰',
    'ॲ',
    'ॳ',
    'ॴ',
    'ॵ',
    'ॶ',
    'ॷ',
    'ॸ',
    'ॹ',
    'ॺ',
    'ॻ',
    'ॼ',
    'ॽ',
    'ॾ',
    'ॿ',
    '꣠',
    '꣡',
    '꣢',
    '꣣',
    '꣤',
    '꣥',
    '꣦',
    '꣧',
    '꣨',
    '꣩',
    '꣪',
    '꣫',
    '꣬',
    '꣭',
    '꣮',
    '꣯',
    '꣰',
    '꣱',
    'ꣲ',
    'ꣳ',
    'ꣴ',
    'ꣵ',
    'ꣶ',
    'ꣷ',
    '꣸',
    '꣹',
    '꣺',
    'ꣻ',
    '꣼',
    'ꣽ',
    'ꣾ',
    'ꣿ',
    ]
    </script>

    <template>
    <div class="characterPalette-container">
        <button @click="togglePalette = !togglePalette" class="btn btn-outline-warning me-2">
        ॳ Rare Characters
        </button>
        <span v-if="copied">Copied &zwnj;{{ copied }} !</span>
        <div v-if="togglePalette" class="characterPalette mt-2">
        <button
            class="btn btn-outline-secondary character-button"
            v-for="character in characters"
            :key="character"
            @click="copyToClipboard(character)"
        >
            {{ character }}
        </button>
        </div>
    </div>
    </template>

    <style>
    .characterPalette-container {
    display: inline-block;
    /* max-width: 20rem; */
    }

    .characterPalette {
    /* display: flex; */
    position: absolute;
    max-width: 20rem;
    /* flex-shrink: 1; */
    padding: 0.5em;
    background-color: var(--color-background);
    border: var(--bs-border-width) solid var(--bs-border-color);
    border-radius: var(--bs-border-radius);
    justify-content: space-around;
    }

    .character-button {
    width: 2rem;
    margin-left: 0.5em;
    margin-bottom: 0.5em;
    padding: 6px 0px 6px 0px;
    text-align: center;
    background-color: var(--bs-body-bg);
    }
    </style>



    ================================================
    FILE: frontend/src/components/typing-utils/devanagariInputUtils.js
    ================================================
    import {
        singleConsonantMap, doubleCharMap, tripleCharMap,
        dependentVowelMap, independentVowelMap, combinedVowelMap,
        potentialVowelKeys, vowelReplacementMap, sequencePrefixes,
        miscMap, simpleInsertMap, // Import new maps
        handleSingleConsonant, insertCharacter, replacePreviousChars,
        applyDependentVowel, insertConsonantSequence, replaceConsonantSequence,
        applyNukta, // Import new helper
        logCharactersBeforeCursor,
        HALANT, ZWNJ, ZWJ, NUKTA, ANUSVARA, VISARGA, CANDRABINDU, DANDA, DOUBLE_DANDA, OM // Import constants
    } from './InputClusterCode'
    
    let lastEffectiveKey = null;
    
    export function handleInput(event, devanagariRef) {
        const key = event.key;
        const input = event.target;
        const cursorPosition = input.selectionStart;
        const currentValue = input.value;


        // Helper to check if a character is a "bare" Devanagari consonant
        // (not a matra, not halant, not modifier, etc., just the consonant character itself)
        const isBareDevanagariConsonant = (char) => {
            if (!char || char.length !== 1) return false; // Must be a single character
            const cp = char.charCodeAt(0);
            // Devanagari Unicode block range for consonants:
            // Main consonants: U+0915 (क) to U+0939 (ह)
            // Additional consonants (e.g., ळ, or nukta forms like क़ if they are single codepoints): U+0958 to U+095F
            if ((cp >= 0x0915 && cp <= 0x0939) || (cp >= 0x0958 && cp <= 0x095F)) {
                return true;
            }
            return false;
        };
    
        // --- Basic Filtering ---
        if (event.metaKey || event.ctrlKey || event.altKey) {
            console.log("Ignoring Ctrl/Meta/Alt key press");
            return;
        }
    
        let effectiveKey = key;
        if (event.shiftKey && key.length === 1 && !key.match(/[a-zA-Z]/)) {
            // Allow Shift + '.' for Nukta trigger? Or other specific combos?
            // For now, treat Shift + non-letter as potentially ignorable or map explicitly
            if (key === '.') { // Allow Shift + '.' if needed later for something else?
                // effectiveKey = '>'; // Example if Shift+. has meaning
                console.log("Shift + . detected, treating as '.' for now");
                effectiveKey = '.'; // Treat as period for now, can change if needed
            } else {
                console.log("Ignoring Shift + Symbol key press:", key);
                lastEffectiveKey = null;
                return;
            }
        } else if (key.length > 1 && key !== 'Backspace') {
            console.log(`Ignoring functional key: ${key}`);
            lastEffectiveKey = null;
            return;
        } else if (event.shiftKey && key.length === 1 && key.match(/[A-Z]/)) {
            effectiveKey = key; // Uppercase letter
        } else if (key.length === 1 && key.match(/[a-z]/)) {
            effectiveKey = key; // Lowercase letter
        } else if (key === 'Backspace') {
            effectiveKey = 'Backspace';
        } else if (simpleInsertMap[key] !== undefined) { // Check if it's a simple insert key (digit, space, ., etc.)
            effectiveKey = key;
        } else if (key === '`') { // Keep explicit halant trigger
            effectiveKey = '`';
        } else if (key === '.') { // Allow period
            effectiveKey = '.';
        } else {
            console.log(`Key "${key}" might pass to fallback or be ignored`);
            // Decide whether to ignore unmapped symbols or let them pass
            // Let's ignore unmapped symbols for now to avoid unexpected chars
            // You can remove this 'return' to allow them.
            // lastEffectiveKey = null; // Reset if ignoring
            // return;
            effectiveKey = key; // Allow pass-through for now
        }
    
    
        console.log("-------------------------");
        console.log(`Effective Key: ${effectiveKey} (at pos ${cursorPosition}) | Last Key: ${lastEffectiveKey}`);
        console.log("State BEFORE processing:");
        logCharactersBeforeCursor(input);
    
        const charM1 = currentValue[cursorPosition - 1];
        const charM2 = currentValue[cursorPosition - 2];
        const charM3 = currentValue[cursorPosition - 3];
        const charM4 = currentValue[cursorPosition - 4];
        const charM5 = currentValue[cursorPosition - 5];
    
        // --- Explicit Halant + ZWNJ ('`' key) ---
        // Insert HALANT + ZWNJ (useful for controlling conjuncts explicitly)
        if (effectiveKey === '`') {
            event.preventDefault();
            const sequence = HALANT + ZWNJ;
            insertCharacter(input, devanagariRef, sequence, cursorPosition);
            console.log('Inserted explicit halant + ZWNJ');
            lastEffectiveKey = effectiveKey;
            return;
        }
    
        // --- Backspace Handling (Keep existing logic) ---
        if (effectiveKey === 'Backspace') {
            lastEffectiveKey = null; // Reset sequence tracking
            if (charM1 === ZWNJ && charM2 === HALANT && cursorPosition >=3 ) {
                event.preventDefault();
                console.log('Backspace: removing Base/Modifier + Halant + ZWNJ'); // Nukta case C+Nukta+H+ZWNJ needs different handling? No, 3 chars works.
                const newValue = currentValue.slice(0, cursorPosition - 3) + currentValue.slice(cursorPosition);
                devanagariRef.value = newValue; input.value = newValue;
                input.setSelectionRange(cursorPosition - 3, cursorPosition - 3);
                logCharactersBeforeCursor(input); return;
            }
            else if (charM2 === HALANT && cursorPosition >= 2) {
                event.preventDefault();
                const newValue = currentValue.slice(0, cursorPosition - 1) + ZWNJ + currentValue.slice(cursorPosition);
                console.log('Backspace: Removed last char, Inserted ZWNJ after halant (original logic)');
                devanagariRef.value = newValue; input.value = newValue;
                input.setSelectionRange(cursorPosition, cursorPosition);
                logCharactersBeforeCursor(input); return;
            }
            else {
                console.log('Backspace: Default behavior');
                queueMicrotask(() => { devanagariRef.value = input.value; logCharactersBeforeCursor(input); });
                return;
            }
        }
    
        // --- Simple Insertions (Space, Digits, ZWJ, ZWNJ, Period, Avagraha etc.) ---
        if (simpleInsertMap[effectiveKey] !== undefined) {
            event.preventDefault();
            const charToInsert = simpleInsertMap[effectiveKey];
            insertCharacter(input, devanagariRef, charToInsert, cursorPosition);
            // Reset last key for space and punctuation, but maybe not for ZWJ/ZWNJ?
            if (charToInsert === ' ' || charToInsert === '.' || charToInsert === AVAGRAHA ) {
                lastEffectiveKey = null;
            } else {
                lastEffectiveKey = effectiveKey; // Keep sequence potential for digits? or ZWJ/ZWNJ? Let's update.
            }
            return;
        }
    
        // --- Miscellaneous Sequence Handling (MM, ff, .N, om) ---
        let potentialMiscSequence = '';
        let miscSequenceHandled = false;
        if (lastEffectiveKey && sequencePrefixes[lastEffectiveKey]?.includes(effectiveKey)) {
            potentialMiscSequence = lastEffectiveKey + effectiveKey;
            console.log("Potential Misc sequence:", potentialMiscSequence);
    
            // Check for MM (Chandrabindu)
            if (potentialMiscSequence === 'MM' && charM1 === ANUSVARA) {
                event.preventDefault();
                replacePreviousChars(input, devanagariRef, 1, CANDRABINDU, cursorPosition);
                miscSequenceHandled = true;
            }
            // Check for ff (Double Danda) - Note conflict with consonant 'f'
            // Prioritize 'ff' if previous was DANDA.
            else if (potentialMiscSequence === 'ff' && charM1 === DANDA) {
                event.preventDefault();
                replacePreviousChars(input, devanagariRef, 1, DOUBLE_DANDA, cursorPosition);
                miscSequenceHandled = true;
            }
            // Check for .N (Nukta)
            else if (potentialMiscSequence === '.N') {
                // Requires C+H+ZWNJ context
                if (cursorPosition >= 3 && charM1 === ZWNJ && charM2 === HALANT && !potentialVowelKeys.has(charM3) /* Ensure it's a consonant base */ ) {
                    event.preventDefault();
                    applyNukta(input, devanagariRef, cursorPosition); // Use helper
                    miscSequenceHandled = true;
                } else {
                    console.log("Nukta (.N) sequence detected but invalid context.");
                    // Prevent default insertion of 'N'? Or allow 'N'? Let's prevent.
                    event.preventDefault();
                    // Don't set miscSequenceHandled = true, let 'N' be potentially handled later if needed
                    lastEffectiveKey = effectiveKey; // Update last key to N
                    return; // Exit early, nukta cannot be applied here
                }
            }
            // Check for om
            else if (potentialMiscSequence === 'om' && miscMap[potentialMiscSequence]) {
                event.preventDefault();
                insertCharacter(input, devanagariRef, OM, cursorPosition);
                miscSequenceHandled = true;
                lastEffectiveKey = null; // Reset sequence after om
                return; // Handled 'om'
            }
        }
    
        if (miscSequenceHandled) {
            lastEffectiveKey = effectiveKey; // Update last key
            return; // Exit if a misc sequence was handled
        }
    
        // --- Explicit HALANT Insertion ('q' key) ---
        // Only inserts HALANT, potentially removing ZWNJ if present
        if (effectiveKey === 'q') {
            event.preventDefault();
            if (charM1 === ZWNJ && charM2 === HALANT) {
                // We are after C + H + ZWNJ. Replace ZWNJ with H. Net effect: remove ZWNJ.
                replacePreviousChars(input, devanagariRef, 1, '', cursorPosition); // Remove ZWNJ
                console.log("Applied explicit halant (q): Removed ZWNJ after existing Halant.");
            } else if (charM1 === ZWNJ) {
                // After explicit HALANT+ZWNJ (` key). Replace ZWNJ with just HALANT.
                replacePreviousChars(input, devanagariRef, 1, HALANT, cursorPosition);
                console.log("Applied explicit halant (q): Replaced ZWNJ with Halant.");
            }
            else {
                // Insert HALANT after a vowel or a consonant+matra
                insertCharacter(input, devanagariRef, HALANT, cursorPosition);
                console.log("Applied explicit halant (q): Inserted Halant.");
            }
            lastEffectiveKey = effectiveKey;
            return;
        }
    
        // --- Custom 'a' Vowel Handling (Schwa Deletion & Matra Application) ---
        // Define which keys trigger schwa deletion (C+H+ZWNJ -> C)
        const isSchwaDeletionKey = (effectiveKey === 'a' || effectiveKey === 'A');

        // Define which keys trigger 'aa' matra (C -> C+ ा) and what that matra is.
        let aaMatra = null;
        if (effectiveKey === 'a' || effectiveKey === 'A') {
            aaMatra = dependentVowelMap['a']; // Should be 'ा'
        } else if (effectiveKey === 'aa' || effectiveKey === 'AA') { // If you have 'aa'/'AA' mapping
            aaMatra = dependentVowelMap['aa']; // Should also be 'ा'
        }
        // Add more else if for other 'a'-like keys if necessary

        // 1. Handle Schwa Deletion: C + Halant + ZWNJ + 'a'/'A'  --->  C
        if (isSchwaDeletionKey && cursorPosition >= 2 && charM1 === ZWNJ && charM2 === HALANT) {
            // Context: charM3 (Base Consonant) + charM2 (Halant) + charM1 (ZWNJ)
            // Action: Pressing 'a' or 'A' removes Halant + ZWNJ, leaving just charM3.
            event.preventDefault();
            replacePreviousChars(input, devanagariRef, 2, '', cursorPosition); // Removes the last 2 chars (Halant + ZWNJ)
            console.log(`Schwa Deletion by '${effectiveKey}': Removed H+ZWNJ after '${charM3}' to form full consonant.`);
            lastEffectiveKey = effectiveKey; // Update last key
            return; // Crucial: exit after handling
        }

        // 2. Handle 'aa' Matra Application: C + 'a'/'A'/'aa'/'AA'  --->  C + ा
        // This executes if the schwa deletion didn't happen (e.g., cursor is after a full consonant).
        if (aaMatra === 'ा' && charM1 && isBareDevanagariConsonant(charM1)) {
            // Context: charM1 is a bare consonant (e.g., 'क', 'ख')
            // Action: Pressing an 'a'-like key appends the 'ा' matra.
            event.preventDefault();
            replacePreviousChars(input, devanagariRef, 1, charM1 + aaMatra, cursorPosition); // Replaces charM1 with charM1 + 'ा'
            console.log(`'${effectiveKey}' applied Matra '${aaMatra}' to bare consonant '${charM1}'.`);
            lastEffectiveKey = effectiveKey; // Update last key
            return; // Crucial: exit after handling
        }
        // --- END Custom 'a' Vowel Handling ---


        // --- Single Anusvara / Visarga ('M', 'H') Application ---
        if (effectiveKey === 'M' || effectiveKey === 'H') {
            event.preventDefault();
            const modifier = (effectiveKey === 'M') ? ANUSVARA : VISARGA;
    
            if (cursorPosition >= 3 && charM1 === ZWNJ && charM2 === HALANT) {
                // Preceded by C + H + ZWNJ. Replace H+ZWNJ with modifier.
                // Base char is charM3
                const baseChar = charM3;
                replacePreviousChars(input, devanagariRef, 2, modifier, cursorPosition); // Remove H+ZWNJ, add modifier
                console.log(`Applied modifier ${modifier} after ${baseChar} (replacing H+ZWNJ)`);
            } else if (cursorPosition > 0 && charM1 !== HALANT) {
                // Preceded by a full character (Vowel or C+Matra). Append modifier.
                insertCharacter(input, devanagariRef, modifier, cursorPosition);
                console.log(`Appended modifier ${modifier} after ${charM1}`);
            } else {
                // Context not suitable (e.g., start of input, after halant without ZWNJ)
                console.log(`Cannot apply modifier ${modifier} in current context.`);
                // Optionally insert with dotted circle: insertCharacter(input, devanagariRef, '\u25CC' + modifier, cursorPosition);
            }
            lastEffectiveKey = effectiveKey;
            return;
        }
    
        // --- Single Danda Insertion ('f' key) ---
        // Needs careful handling due to 'f' also mapping to consonant 'फ'
        // Rule: If 'f' is pressed AND it wasn't part of 'ff', treat as DANDA *unless*
        // the context implies the consonant 'फ'.
        // Let's prioritize DANDA if not after C+H+ZWNJ.
        if (effectiveKey === 'f') {
            const isConsonantContext = cursorPosition >= 3 && charM1 === ZWNJ && charM2 === HALANT;
            const isConsonantReplacementContext = doubleCharMap['h']?.[charM3] !== undefined && charM1 === ZWNJ && charM2 === HALANT; // e.g., p+h -> ph
    
            // If not likely forming 'फ' or 'ph', insert Danda.
            if (!isConsonantContext && !isConsonantReplacementContext) {
                event.preventDefault();
                insertCharacter(input, devanagariRef, DANDA, cursorPosition);
                console.log("Inserted Danda (|)");
                lastEffectiveKey = effectiveKey; // Treat as sequence starter for 'ff'
                return;
            }
            // Otherwise, let it fall through to consonant handling below.
            console.log("'f' key pressed in consonant context, will be handled as 'फ'");
        }
    
    
        // --- Consonant Sequence Completion (Triples, Doubles) ---
        // Keep this logic exactly as it was
        const tripleMappings = tripleCharMap[effectiveKey];
        if (tripleMappings && cursorPosition >= 5) { /* ... triple check logic ... */
            if (charM1 === ZWNJ && charM2 === HALANT && charM4 === HALANT) {
                const precedingSequence = charM5 + charM3;
                if (tripleMappings[precedingSequence]) {
                    const mapping = tripleMappings[precedingSequence];
                    event.preventDefault();
                    replaceConsonantSequence(input, devanagariRef, mapping.resultChar, cursorPosition, mapping.remove);
                    lastEffectiveKey = effectiveKey; return;
                }
            }
            if (effectiveKey === 'r' && charM1 === ZWNJ && charM2 === HALANT && charM3 === 'श' && tripleMappings['श']) {
                const mapping = tripleMappings['श'];
                event.preventDefault();
                replaceConsonantSequence(input, devanagariRef, mapping.resultChar, cursorPosition, mapping.remove);
                lastEffectiveKey = effectiveKey; return;
            }
        }
        const doubleMappings = doubleCharMap[effectiveKey];
        if (doubleMappings && cursorPosition >= 3) { /* ... double check logic ... */
            if (charM1 === ZWNJ && charM2 === HALANT) {
                const precedingBase = charM3;
                if (doubleMappings[precedingBase]) {
                    const mapping = doubleMappings[precedingBase];
                    event.preventDefault();
                    replaceConsonantSequence(input, devanagariRef, mapping.resultChar, cursorPosition, mapping.remove);
                    lastEffectiveKey = effectiveKey; return;
                }
            }
        }
    
        // --- Vowel Handling Logic (Keep existing logic) ---
        let potentialVowelSequence = '';
        if (lastEffectiveKey && sequencePrefixes[lastEffectiveKey]?.includes(effectiveKey) && potentialVowelKeys.has(effectiveKey[0])) {
            potentialVowelSequence = lastEffectiveKey + effectiveKey;
            console.log("Potential Vowel sequence:", potentialVowelSequence);
            if (combinedVowelMap[potentialVowelSequence]) {
                const isDependentContext = charM1 === ZWNJ && charM2 === HALANT && cursorPosition >= 3;
                const isVowelReplacementContext = vowelReplacementMap[charM1]?.[effectiveKey];
    
                if (isVowelReplacementContext) {
                    event.preventDefault();
                    const replacementChar = vowelReplacementMap[charM1][effectiveKey];
                    replacePreviousChars(input, devanagariRef, 1, replacementChar, cursorPosition);
                    console.log(`Vowel Replacement: ${charM1} + ${effectiveKey} -> ${replacementChar}`);
                    lastEffectiveKey = effectiveKey; return;
                } else if (isDependentContext && dependentVowelMap[potentialVowelSequence]) {
                    event.preventDefault();
                    applyDependentVowel(input, devanagariRef, dependentVowelMap[potentialVowelSequence], cursorPosition);
                    console.log(`Applied complex matra: ${dependentVowelMap[potentialVowelSequence]}`);
                    lastEffectiveKey = effectiveKey; return;
                } else if (!isDependentContext && independentVowelMap[potentialVowelSequence]) {
                    event.preventDefault();
                    insertCharacter(input, devanagariRef, independentVowelMap[potentialVowelSequence], cursorPosition);
                    console.log(`Inserted complex independent vowel: ${independentVowelMap[potentialVowelSequence]}`);
                    lastEffectiveKey = effectiveKey; return;
                } else {
                    console.log(`Sequence ${potentialVowelSequence} valid but context mismatch?`);
                }
            }
        }
        // Vowel Replacement Check (single key)
        if (potentialVowelKeys.has(effectiveKey) && charM1 && vowelReplacementMap[charM1]?.[effectiveKey]) {
            event.preventDefault();
            const replacementChar = vowelReplacementMap[charM1][effectiveKey];
            replacePreviousChars(input, devanagariRef, 1, replacementChar, cursorPosition);
            console.log(`Vowel Replacement (single key): ${charM1} + ${effectiveKey} -> ${replacementChar}`);
            lastEffectiveKey = effectiveKey; return;
        }
        // Single Vowel / Single Consonant Handling
        const isDepContext = charM1 === ZWNJ && charM2 === HALANT && cursorPosition >= 3;
        const devDep = dependentVowelMap[effectiveKey];
        const devIndep = independentVowelMap[effectiveKey];
        const devCons = singleConsonantMap[effectiveKey];
    
        if (isDepContext) {
            if (devDep) {
                event.preventDefault(); applyDependentVowel(input, devanagariRef, devDep, cursorPosition);
                lastEffectiveKey = effectiveKey; return;
            } else if (devCons) {
                event.preventDefault();
                replacePreviousChars(input, devanagariRef, 1, devCons + HALANT + ZWNJ, cursorPosition);
                console.log(`Forming conjunct: Removed ZWNJ, added ${devCons}+H+ZWNJ`);
                lastEffectiveKey = effectiveKey; return;
            } else if (devIndep) {
                event.preventDefault();
                replacePreviousChars(input, devanagariRef, 2, devIndep, cursorPosition);
                console.log(`WARN: Independent vowel after C+H+ZWNJ. Replaced H+ZWNJ with ${devIndep}`);
                lastEffectiveKey = effectiveKey; return;
            }
        } else {
            if (devIndep) {
                event.preventDefault(); insertCharacter(input, devanagariRef, devIndep, cursorPosition);
                lastEffectiveKey = effectiveKey; return;
            } else if (devCons) {
                // Check if it's 'f' which should have been handled as Danda already if appropriate
                if (effectiveKey === 'f') {
                    // If we reached here, 'f' should be treated as consonant 'फ'
                    event.preventDefault();
                    handleSingleConsonant(event, devanagariRef, devCons);
                    lastEffectiveKey = effectiveKey; return;
                } else {
                // Handle other consonants normally
                event.preventDefault(); handleSingleConsonant(event, devanagariRef, devCons);
                lastEffectiveKey = effectiveKey; return;
                }
            } else if (devDep) {
                event.preventDefault();
                const standaloneMatra = '\u25CC' + devDep;
                insertCharacter(input, devanagariRef, standaloneMatra, cursorPosition);
                console.log(`WARN: Dependent vowel in independent context. Inserted ${standaloneMatra}`);
                lastEffectiveKey = effectiveKey; return;
            }
        }
    
        // --- Handle 'h' as a single consonant if it didn't form a double/triple ---
        if (effectiveKey === 'h' && !doubleMappings?.[charM3] && !tripleMappings?.[charM5+charM3]) {
            event.preventDefault();
            handleSingleConsonant(event, devanagariRef, 'ह');
            lastEffectiveKey = effectiveKey;
            return;
        }
    
    
        // --- Fallback ---
        console.log(`Key "${effectiveKey}" not handled by custom logic. Default behavior might occur.`);
        lastEffectiveKey = effectiveKey; // Update last key even if default occurs
        queueMicrotask(() => {
            devanagariRef.value = input.value;
            logCharactersBeforeCursor(input);
        });
    }


    ================================================
    FILE: frontend/src/components/typing-utils/InputClusterCode.js
    ================================================
    // InputClusterCode.js

    export function logCharactersBeforeCursor(input) {
    const cursorPosition = input.selectionStart;
    const currentValue = input.value;
    // Log more characters for debugging multi-char sequences
    console.log({
        '-5': currentValue[cursorPosition - 5],
        '-4': currentValue[cursorPosition - 4],
        '-3': currentValue[cursorPosition - 3],
        '-2': currentValue[cursorPosition - 2],
        '-1': currentValue[cursorPosition - 1]
    });
    return;
    }

    // --- Character Constants ---
    export const HALANT = '\u094D';
    export const ZWNJ = '\u200C'; // Zero-Width Non-Joiner
    export const ZWJ = '\u200D';  // Zero-Width Joiner
    export const NUKTA = '\u093C'; // Combining Dot Below (Nukta)
    export const ANUSVARA = '\u0902'; // ं
    export const VISARGA = '\u0903'; // ः
    export const CANDRABINDU = '\u0901'; // ँ
    export const AVAGRAHA = '\u093D'; // ऽ
    export const DANDA = '\u0964'; // ।
    export const DOUBLE_DANDA = '\u0965'; // ॥
    export const OM = '\u0950'; // ॐ
    // Add constants for other special characters if keys are assigned
    // export const DEVANAGARI_ABBREVIATION_SIGN = '\u0970';
    // export const DEVANAGARI_SIGN_HIGH_SPACING_DOT = '\u0971';
    // export const DEVANAGARI_SIGN_INVERTED_CANDRABINDU = '\u0900';
    // export const DEVANAGARI_STRESS_SIGN_UDATTA = '\u0951';
    // export const DEVANAGARI_STRESS_SIGN_ANUDATTA = '\u0952';

    // --- Consonant Mappings ---
    // Maps single Roman keys directly to Devanagari base consonants
    export const singleConsonantMap = {
    'k': 'क', 'g': 'ग', 'c': 'च', 'j': 'ज', 'T': 'ट', 't': 'त', 'D': 'ड',
    'd': 'द', 'N': 'ण', 'n': 'न', 'p': 'प', 'b': 'ब', 'm': 'म', 'y': 'य',
    'r': 'र', 'l': 'ल', 'v': 'व', 'V': 'ङ', 'S': 'ष', 's': 'स', 'h': 'ह',
    'L': 'ळ', 'Y': 'ञ',
    'f': 'फ', // Note: 'f' also used for DANDA in miscMap
    'z': 'ज', // Note: 'z' also used for vowel prefixes
    'q': 'क', // Note: 'q' also used for HALANT in miscMap
    };

    // Structure: triggerKey: { precedingDevanagariBase: { resultChar: devanagariBase, remove: count } }
    export const doubleCharMap = {
    'h': { // Aspirates + sh
        'क': { resultChar: 'ख', remove: 3 }, 'ग': { resultChar: 'घ', remove: 3 },
        'च': { resultChar: 'छ', remove: 3 }, 'ज': { resultChar: 'झ', remove: 3 },
        'ट': { resultChar: 'ठ', remove: 3 }, 'ड': { resultChar: 'ढ', remove: 3 },
        'त': { resultChar: 'थ', remove: 3 }, 'द': { resultChar: 'ध', remove: 3 },
        'प': { resultChar: 'फ', remove: 3 }, 'ब': { resultChar: 'भ', remove: 3 },
        'स': { resultChar: 'श', remove: 3 },
    },
    's': {
        'क': { resultChar: 'क्ष', remove: 3 }, // k + s -> ks (maps to kS = क्ष)
    },
    'S': {
        'क': { resultChar: 'क्ष', remove: 3 }  // k + S -> kS
    },
    };

    // Structure: triggerKey: { precedingDevSequence: { resultChar: devanagariBase, remove: count } }
    export const tripleCharMap = {
    'y': {
        'दन': { resultChar: 'ज्ञ', remove: 5 }, // d + n + y -> dny (ज्ञ)
        'गञ': { resultChar: 'ज्ञ', remove: 5 }, // g + Y + y -> gny (ज्ञ)
        'गन': { resultChar: 'ज्ञ', remove: 5 }, // g + n + y -> gny (ज्ञ)
    },
    'r': {
        'श': { resultChar: 'श्र', remove: 3 }, // sh + r -> shr
    },
    };


    // --- Vowel Mappings ---
    // Dependent Vowels (Matras)
    export const dependentVowelMap = {
        'a':'ा', 'e':'े', 'i':'ि', 'o':'ो', 'u':'ु',
        'aa': 'ा', 'ee': 'ी', 'ii': 'ी', 'uu': 'ू', 'oo': 'ू',
        'ai':'ै', 'au':'ौ', 'ou':'ौ',
        'Rri':'ृ', 'RrI':'ॄ', 'Lli':'ॢ', 'LlI':'ॣ',
        'ze':'ॆ', 'zo':'ॊ', 'aE':'ॅ', 'aO':'ॉ',
        'zau':'\u094F', // Kashmiri/Bihari Au Matra
    };

    // Independent Vowels
    export const independentVowelMap = {
        'a':'अ', 'A':'अ', 'i':'इ', 'I':'इ', 'u':'उ', 'U':'उ',
        'e':'ए', 'E':'ए', 'o':'ओ', 'O':'ओ',
        'aa':'आ', 'AA':'आ', 'ii':'ई', 'II':'ई', 'ee':'ई',
        'uu':'ऊ', 'UU':'ऊ', 'oo':'ऊ',
        'ai':'ऐ', 'AI':'ऐ', 'au':'औ', 'AU':'औ', 'ou':'औ',
        'RRi':'ऋ', 'RRI':'ॠ', 'LLi':'ऌ', 'LLI':'ॡ',
        'AE':'ॲ', // Marathi AE
        'AO':'ऑ', // Marathi/Borrowed AO
        // 'aE':'ऍ', // Alternate AE - choose one or handle contextually
        // 'aO':'ऑ', // Alternate AO - choose one or handle contextually
        'zEE':'ऎ', // South Indian Short E
        'zO':'ऒ',  // South Indian Short O
        'zA':'ऄ', // Historic/Regional A
        'zAU':'ॵ', // Historic/Regional Au
    };

    // Combined lookup for potential vowel starting keys/sequences
    export const potentialVowelKeys = new Set([
        'a', 'A', 'e', 'E', 'i', 'I', 'o', 'O', 'u', 'U',
        'R', 'L', 'z' // Covers Rri, Lli, ze, zo, zau etc.
    ]);

    // Combined map for resolving full vowel sequences
    export const combinedVowelMap = { ...dependentVowelMap, ...independentVowelMap };

    // --- Vowel Sequence Handling Logic ---
    // Map for replacements like i+i -> ii, e+i -> ai, etc.
    // Structure: { precedingDevChar: { currentKey: replacementDevChar } }
    export const vowelReplacementMap = {
        // Dependent Matra Replacements
        'ि': { 'i': 'ी', 'e': 'ी' }, // short i + i/e -> long ii/ee
        'ु': { 'u': 'ू', 'o': 'ू' }, // short u + u/o -> long uu/oo
        'े': { 'e': 'ी', 'i': 'ै' }, // e + e -> ee, e + i -> ai
        'ो': { 'o': 'ू', 'u': 'ौ', 'i': 'ौ' }, // o + o -> oo, o + u/i -> au
        'ृ': { 'I': 'ॄ', 'i': 'ॄ' }, // Rri + I/i -> RrI
        'ॢ': { 'I': 'ॣ', 'i': 'ॣ' }, // Lli + I/i -> LlI
        'ा': { 'a': 'ा', 'E': 'ॅ', 'O': 'ॉ' }, // aa + a -> aa, aa + E -> aE Candra, aa + O -> aO Candra
        // Independent Vowel Replacements
        'इ': { 'i': 'ई', 'I': 'ई', 'e': 'ई', 'E': 'ई' }, // short I + i/I/e/E -> long II/EE
        'उ': { 'u': 'ऊ', 'U': 'ऊ', 'o': 'ऊ', 'O': 'ऊ' }, // short U + u/U/o/O -> long UU/OO
        'ए': { 'e': 'ई', 'E': 'ई', 'i': 'ऐ', 'I': 'ऐ' }, // E + e/E -> EE, E + i/I -> AI
        'ओ': { 'o': 'ऊ', 'O': 'ऊ', 'u': 'औ', 'U': 'औ' }, // O + o/O -> OO, O + u/U -> AU
        'अ': { 'a': 'आ', 'A': 'आ', 'E': 'ॲ', 'O': 'ऑ'}, // A + a/A -> AA, A + E -> AE(Marathi), A + O -> AO(Marathi)
        'ऋ': { 'I': 'ॠ' }, // RRi + I -> RRI
        'ऌ': { 'I': 'ॡ' }, // LLi + I -> LLI
    };


    // --- Miscellaneous Mappings ---
    export const miscMap = { // VOWEL MODIFIERS(m), HALANT(H), NUKTA(N), NUMBERS, CURRENCY etc.
        // Single Key Modifiers / Symbols
        'M': ANUSVARA,      // 'ं'
        'H': VISARGA,       // 'ः'
        'F': AVAGRAHA,      // 'ऽ'
        'q': HALANT,        // '्' (Explicit Halant ONLY - applies differently than Halant+ZWNJ)
        ' ': ' ',
        '.': '.',           // Period
        'f': DANDA,         // '।', Note: 'f' is also consonant 'फ'
        '0': '०', '1': '१', '2': '२', '3': '३', '4': '४',
        '5': '५', '6': '६', '7': '७', '8': '८', '9': '९',
        'W': ZWJ,           // '\u200D' (Zero Width Joiner)
        'w': ZWNJ,          // '\u200C' (Zero Width Non-Joiner)

        // Sequences (Handled in handleInput based on last key)
        'MM': CANDRABINDU,  // 'ँ' (Replaces Anusvara)
        '.N': NUKTA,        // '◌़' (Applies to preceding consonant)
        'ff': DOUBLE_DANDA, // '॥' (Replaces Danda)
        'om': OM,           // 'ॐ'

        // --- Keys needing assignment for unmapped chars ---
        // Choose appropriate keys and uncomment/add here if needed
        // Example assignments:
        // '\'': '\u0970', // DEVANAGARI ABBREVIATION SIGN
        // '_': '\u0971',  // DEVANAGARI SIGN HIGH SPACING DOT
        // '^': '\u0900',  // DEVANAGARI SIGN INVERTED CANDRABINDU
        // '+': '\u0951',  // DEVANAGARI STRESS SIGN UDATTA
        // '=': '\u0952',  // DEVANAGARI STRESS SIGN ANUDATTA
    };

    // Helper Map for simple direct insertions (no context needed beyond the key itself)
    // Includes digits, space, period, ZWJ, ZWNJ, Avagraha, and any assigned simple symbols
    export const simpleInsertMap = {
        ' ': ' ', '.': '.',
        '0': '०', '1': '१', '2': '२', '3': '३', '4': '४',
        '5': '५', '6': '६', '7': '७', '8': '८', '9': '९',
        'W': ZWJ, 'w': ZWNJ,
        'F': AVAGRAHA, // Avagraha can usually be inserted directly
        // Add keys for other simple insertions if assigned in miscMap
        // '\'': '\u0970', '_': '\u0971', '^': '\u0900', '+': '\u0951', '=': '\u0952',
    };

    // --- Sequence Prefix Information ---
    // Helps identify potential multi-character sequences
    // Structure: { key: potentialNextKey[] }
    // ** Define the base object first **
    export const sequencePrefixes = {
        // Vowel prefixes
        'R': ['r', 'R', 'i', 'I'], // For Rr, RR, Rri, RRI
        'L': ['l', 'L', 'i', 'I'], // For Ll, LL, Lli, LLI
        'z': ['e', 'o', 'a', 'E', 'A', 'O', 'U'], // For ze, zo, za, zE, zA etc.
        'a': ['a', 'e', 'i', 'u', 'E', 'O'], // For aa, ae, ai, au, aE, aO
        'A': ['A', 'E', 'I', 'O', 'U'], // For AA, AE, AI, AO, AU
        'e': ['e', 'i'], // For ee, ei (ai)
        'E': ['E', 'I'], // For EE, EI (ai)
        'i': ['i', 'e'], // For ii, ie (ee)
        'I': ['I', 'E'], // For II, IE (ee)
        'o': ['o', 'u', 'i'], // For oo, ou (au), oi (au?) - ** Initial definition **
        'O': ['O', 'U', 'I'], // For OO, OU (au), OI (au?)
        'u': ['u', 'o'], // For uu, uo (oo)
        'U': ['U', 'O'], // For UU, UO (oo)

        // Misc prefixes
        '.': ['N'], // For Nukta sequence .N
        'M': ['M'], // For Chandrabindu sequence MM
        'f': ['f'], // For Double Danda sequence ff
        // Add 'A', 'U' prefixes if needed for 'AUM' later
    };

    // ** Modify the object after definition **
    // Add 'm' to the potential keys following 'o' for the 'om' sequence
    sequencePrefixes['o'] = [...(sequencePrefixes['o'] || []), 'm'];
    // If you were implementing AUM:
    // sequencePrefixes['A'] = [...(sequencePrefixes['A'] || []), 'U']; // If A can start AU and AUM
    // sequencePrefixes['U'] = [...(sequencePrefixes['U'] || []), 'M']; // If U can start UU and follow A in AUM


    // --- Helper Functions ---

    // Insert Character Sequence (Generic)
    export function insertCharacter(input, devanagariRef, charToInsert, cursorPosition) {
        const currentValue = input.value;
        const newValue =
        currentValue.slice(0, cursorPosition) +
        charToInsert +
        currentValue.slice(cursorPosition);
        const newCursorPosition = cursorPosition + charToInsert.length;

        devanagariRef.value = newValue;
        input.value = newValue;
        input.setSelectionRange(newCursorPosition, newCursorPosition);
        console.log(`Inserted: ${charToInsert}`);
        logCharactersBeforeCursor(input);
    }

    // Replace Previous Characters (Generic)
    export function replacePreviousChars(input, devanagariRef, charsToRemove, charToInsert, cursorPosition) {
        const currentValue = input.value;
        const startReplacePos = cursorPosition - charsToRemove;

        // Ensure we don't go below index 0
        if (startReplacePos < 0) {
            console.error(`replacePreviousChars: Attempting to remove ${charsToRemove} chars from pos ${cursorPosition}.`);
            return; // Or handle differently
        }

        const newValue =
        currentValue.slice(0, startReplacePos) +
        charToInsert +
        currentValue.slice(cursorPosition); // Slice from original cursor pos

        // New cursor position: start of replacement + length of inserted char
        const newCursorPosition = startReplacePos + charToInsert.length;

        devanagariRef.value = newValue;
        input.value = newValue;
        input.setSelectionRange(newCursorPosition, newCursorPosition);
        console.log(`Replaced ${charsToRemove} chars with ${charToInsert}`);
        logCharactersBeforeCursor(input);
    }

    // Helper to apply Dependent Vowel (Matra)
    export function applyDependentVowel(input, devanagariRef, matra, cursorPosition) {
        const currentValue = input.value;
        // Context assumes: Base (charM3) + Halant (charM2) + ZWNJ (charM1) before cursor
        const baseConsonant = currentValue[cursorPosition - 3];
        const charsToRemove = 3; // Base + Halant + ZWNJ
        const charToInsert = baseConsonant + matra;

        // Use the generic replace function
        replacePreviousChars(input, devanagariRef, charsToRemove, charToInsert, cursorPosition);
        console.log(`Applied Matra: ${matra} to ${baseConsonant}`);
    }

    // Insert Consonant Sequence (Base + Halant + ZWNJ)
    export function insertConsonantSequence(input, devanagariRef, baseChar, cursorPosition) {
        const currentValue = input.value;
        const sequence = baseChar + HALANT + ZWNJ;
        const sequenceLength = sequence.length; // Should be 3

        const newValue =
        currentValue.slice(0, cursorPosition) +
        sequence +
        currentValue.slice(cursorPosition);

        const newCursorPosition = cursorPosition + sequenceLength;

        devanagariRef.value = newValue;
        input.value = newValue;
        input.setSelectionRange(newCursorPosition, newCursorPosition);
        console.log(`Inserted ${baseChar} + Halant + ZWNJ`);
        logCharactersBeforeCursor(input);
    }

    // Replace previous sequence with new Consonant Sequence (Base + Halant + ZWNJ)
    export function replaceConsonantSequence(input, devanagariRef, baseChar, cursorPosition, charsToRemove) {
        const currentValue = input.value;
        const sequence = baseChar + HALANT + ZWNJ;
        const sequenceLength = sequence.length; // Should be 3

        const newValue =
        currentValue.slice(0, cursorPosition - charsToRemove) +
        sequence +
        currentValue.slice(cursorPosition);

        // New cursor position: original position - removed chars + inserted chars
        const newCursorPosition = cursorPosition - charsToRemove + sequenceLength;

        devanagariRef.value = newValue;
        input.value = newValue;
        input.setSelectionRange(newCursorPosition, newCursorPosition);
        console.log(`Replaced ${charsToRemove} chars with ${baseChar} + Halant + ZWNJ`);
        logCharactersBeforeCursor(input);
    }

    // Handle insertion of a single consonant character
    export function handleSingleConsonant(event, devanagariRef, devanagariChar) {
    const input = event.target;
    const cursorPosition = input.selectionStart;
    const currentValue = input.value;
    const characterRelativeMinus1 = currentValue[cursorPosition - 1];

    // No preventDefault needed here, it's handled in handleInput

    if (characterRelativeMinus1 === ZWNJ) {
        // If ZWNJ is just before cursor (e.g., after explicit H+ZWNJ),
        // replace the ZWNJ with the new consonant sequence.
        replacePreviousChars(input, devanagariRef, 1, devanagariChar + HALANT + ZWNJ, cursorPosition);
        console.log(`Replaced ZWNJ with ${devanagariChar} + Halant + ZWNJ`);

    } else {
        // Standard insertion: Append Consonant + Halant + ZWNJ
        insertConsonantSequence(input, devanagariRef, devanagariChar, cursorPosition);
    }
    }

    // Apply Nukta to the preceding consonant (C+H+ZWNJ -> C+Nukta+H+ZWNJ)
    export function applyNukta(input, devanagariRef, cursorPosition) {
        const currentValue = input.value;
        // Context: Base (charM3) + Halant (charM2) + ZWNJ (charM1)
        // Basic check to prevent errors if context is wrong, though handleInput should verify
        if (cursorPosition < 3 || currentValue[cursorPosition - 1] !== ZWNJ || currentValue[cursorPosition - 2] !== HALANT) {
            console.error("applyNukta called with invalid context.");
            return;
        }
        const baseConsonant = currentValue[cursorPosition - 3];
        const charsToRemove = 3; // Base + Halant + ZWNJ
        // Insert Base + Nukta + Halant + ZWNJ
        const charToInsert = baseConsonant + NUKTA + HALANT + ZWNJ;

        replacePreviousChars(input, devanagariRef, charsToRemove, charToInsert, cursorPosition);
        console.log(`Applied Nukta to ${baseConsonant}`);
    }


    ================================================
    FILE: frontend/src/router/index.js
    ================================================
    import { createRouter, createWebHistory } from 'vue-router'

    const router = createRouter({
    history: createWebHistory(import.meta.env.BASE_URL),
    routes: [
        {
        path: '/',
        name: 'welcome',
        component: () => import('../views/LandingPage.vue'),
        },
        {
        path: '/annotation',
        name: 'annotation-view',
        component: () => import('../views/AnnotationView.vue'),
        children: [
            {
            path: '/annotation/upload',
            name: 'upload-manuscript',
            component: () => import('../components/archive/UploadForm.vue'),
            alias: '/annotation',
            },
            {
            path: '/annotation/annotate',
            name: 'annotation-section',
            component: () => import('../components/archive/AnnotationSection.vue'),
            },
            {
            path: '/annotation/semi-segment',
            name: 'semi-segment',
            component: () => import('../components/archive/SemiSegmentationSection.vue'),
            },
        ],
        },


        {
        path: '/new',
        name: 'new-annotation-view',
        component: () => import('../views/new-AnnotationView.vue'),
        children: [
            {
            path: '/new/upload',
            name: 'new-manuscript',
            component: () => import('../components/new-UploadForm.vue'),
            },
            {
            path: '/new/img-2-txt',                
            name: 'img-2-txt',                      
            component: () => import('../components/new-IMG2TXT.vue'),
            },
            {
            path: '/new/semi-segment',             
            name: 'new-semi-segment',               
            component: () => import('../components/new-SemiSegmentationSection.vue'),
            }
        ],
        },

        
        {
        path: '/uploads',
        name: 'uploaded-manuscripts',
        component: () => import('../views/UploadedManuscriptsView.vue'),
        },

    ],
    })

    export default router



    ================================================
    FILE: frontend/src/stores/annotationStore.js
    ================================================
    //annotationStore.js
    import { acceptHMRUpdate, defineStore } from 'pinia'
    import { ref, computed as vueComputed } from 'vue' // Use vueComputed to avoid naming clash
    import * as zip from "@zip.js/zip.js";

    export const useAnnotationStore = defineStore('annotations', () => {
    const modelName = ref();
    const recognitions = ref({}); // Structure: { manuscriptName: { pageId: pageData, ... } }
    const userAnnotations = ref([]);
    const currentPage = ref(); // Stores the ID of the current page, e.g., "001"

    // Helper to get the current (assumed single) manuscript name
    const currentManuscriptName = vueComputed(() => {
        const keys = Object.keys(recognitions.value);
        return keys.length > 0 ? keys[0] : null;
    });

    // Helper to get sorted page IDs for the current manuscript
    const sortedPageIds = vueComputed(() => {
        const manuscript = currentManuscriptName.value;
        // Ensure recognitions.value[manuscript] exists and is an object before trying to get keys
        if (manuscript && recognitions.value[manuscript] && typeof recognitions.value[manuscript] === 'object') {
        return Object.keys(recognitions.value[manuscript]).sort((a, b) => {
            const numA = parseInt(a, 10);
            const numB = parseInt(b, 10);
            // If both are parseable as numbers, sort numerically
            if (!isNaN(numA) && !isNaN(numB)) {
            return numA - numB;
            }
            // Otherwise, fall back to lexicographical sort (e.g., for "page1a", "page1b")
            return a.localeCompare(b);
        });
        }
        return [];
    });

    /**
    * Sets the current page.
    * @param {string} pageId - The ID of the page to set as current.
    */
    function setCurrentPage(pageId) {
        if (currentPage.value !== pageId) {
        console.log(`AnnotationStore: Setting current page to ${pageId}`);
        currentPage.value = pageId;
        }
    }

    /**
    * Navigates to the next page in the current manuscript.
    */
    function nextPage() {
        const pages = sortedPageIds.value;
        
        if (pages.length === 0) {
        console.log("AnnotationStore: No pages available to navigate.");
        return;
        }

        if (!currentPage.value) {
            // If current page is not set, default to the first page.
            console.log("AnnotationStore: Current page not set, navigating to the first available page.");
            setCurrentPage(pages[0]);
            return;
        }

        const currentIndex = pages.indexOf(currentPage.value);
        if (currentIndex === -1) {
        console.warn(`AnnotationStore: Current page "${currentPage.value}" not found in available pages. Navigating to the first page.`);
        setCurrentPage(pages[0]); // Default to first page if current is invalid
        return;
        }

        if (currentIndex < pages.length - 1) {
        setCurrentPage(pages[currentIndex + 1]);
        } else {
        console.log("AnnotationStore: Already on the last page.");
        // Optionally, you could add logic to loop back to the first page if desired
        }
    }

    /**
    * Navigates to the previous page in the current manuscript.
    */
    function previousPage() {
        const pages = sortedPageIds.value;

        if (pages.length === 0) {
        console.log("AnnotationStore: No pages available to navigate.");
        return;
        }
        
        if (!currentPage.value) {
            // If current page is not set, it's ambiguous where "previous" should go.
            // Could go to last page, first page, or do nothing. Let's do nothing or go to first.
            console.log("AnnotationStore: Current page not set. Cannot determine previous page. Navigating to first page.");
            setCurrentPage(pages[0]);
            return;
        }

        const currentIndex = pages.indexOf(currentPage.value);
        if (currentIndex === -1) {
        console.warn(`AnnotationStore: Current page "${currentPage.value}" not found in available pages. Navigating to the first page.`);
        setCurrentPage(pages[0]); // Default to first page
        return;
        }

        if (currentIndex > 0) {
        setCurrentPage(pages[currentIndex - 1]);
        } else {
        console.log("AnnotationStore: Already on the first page.");
        // Optionally, you could add logic to loop back to the last page if desired
        }
    }

    /**
    * Sets an initial page, typically the first page of the manuscript.
    * Call this after `recognitions` data is loaded.
    */
    function setInitialPage() {
        const manuscript = currentManuscriptName.value;
        if (manuscript && recognitions.value[manuscript]) {
            const pages = sortedPageIds.value;
            if (pages.length > 0) {
                // Set to first page if current page is not already set or not in the list of pages
                if (!currentPage.value || !pages.includes(currentPage.value)) {
                    console.log(`AnnotationStore: Setting initial page to ${pages[0]}.`);
                    setCurrentPage(pages[0]);
                }
            } else {
                console.log("AnnotationStore: No pages available to set an initial page.");
                setCurrentPage(undefined); // Clear current page if no pages exist
            }
        } else {
            console.log("AnnotationStore: No manuscript data available to set an initial page.");
            setCurrentPage(undefined); // Clear current page if no manuscript
        }
    }

    // --- Existing functions (with minor robustness checks) ---

    function levenshteinDistance(str1 = '', str2 = '') {
        const track = Array(str2.length + 1)
        .fill(null)
        .map(() => Array(str1.length + 1).fill(null))
        for (let i = 0; i <= str1.length; i += 1) {
        track[0][i] = i
        }
        for (let j = 0; j <= str2.length; j += 1) {
        track[j][0] = j
        }
        for (let j = 1; j <= str2.length; j += 1) {
        for (let i = 1; i <= str1.length; i += 1) {
            const indicator = str1[i - 1] === str2[j - 1] ? 0 : 1
            track[j][i] = Math.min(
            track[j][i - 1] + 1, // deletion
            track[j - 1][i] + 1, // insertion
            track[j - 1][i - 1] + indicator, // substitution
            )
        }
        }
        return track[str2.length][str1.length]
    }

    function calculateLevenshteinDistances() {
        for (const annotationsObject of userAnnotations.value) {
        const manuscript_name = annotationsObject['manuscript_name'];
        if (!recognitions.value[manuscript_name]) {
            console.warn(`Recognitions not found for manuscript: ${manuscript_name} during Levenshtein calculation.`);
            continue;
        }
        for (const page in annotationsObject['annotations']) {
            if (!recognitions.value[manuscript_name][page]) {
                console.warn(`Recognitions not found for page: ${page} in manuscript: ${manuscript_name} during Levenshtein calculation.`);
                continue;
            }
            for (const line in annotationsObject['annotations'][page]) {
            const recognitionLine = recognitions.value[manuscript_name][page][line];
            const annotationLine = annotationsObject['annotations'][page][line];

            if (recognitionLine && annotationLine &&
                typeof recognitionLine['predicted_label'] === 'string' &&
                typeof annotationLine['ground_truth'] === 'string'
                ) {
                annotationLine['levenshtein_distance'] =
                levenshteinDistance(
                    recognitionLine['predicted_label'],
                    annotationLine['ground_truth'],
                );
            } else {
                // console.warn(`Missing data for Levenshtein calculation: manuscript ${manuscript_name}, page ${page}, line ${line}`);
            }
            }
        }
        }
    }

    function exportToTxt() {
        const manuscript = currentManuscriptName.value;
        if (!manuscript || !recognitions.value[manuscript] || typeof recognitions.value[manuscript] !== 'object') {
        console.error("No valid manuscript data to export or manuscript name not found.");
        alert("No data available to export.");
        return;
        }

        const zipWriter = new zip.ZipWriter(new zip.BlobWriter("application/zip"));
        const pageKeys = Object.keys(recognitions.value[manuscript]);

        if (pageKeys.length === 0) {
            console.warn("No pages found in the manuscript to export.");
            alert("No pages found in the manuscript to export.");
            return;
        }

        pageKeys.forEach(pageName => {
        let lines = "";
        const pageContent = recognitions.value[manuscript][pageName]; // This is the page data { "0": {predicted_label: ""}, ... }
        
        // Assuming pageContent is an object where keys are line numbers/IDs (e.g. "0", "1")
        // and values are objects with a 'predicted_label'
        if (pageContent && typeof pageContent === 'object') {
            // Sort line keys numerically if possible, otherwise lexicographically
            const lineKeys = Object.keys(pageContent).sort((a, b) => {
                const numA = parseInt(a, 10); const numB = parseInt(b, 10);
                return (!isNaN(numA) && !isNaN(numB)) ? numA - numB : a.localeCompare(b);
            });

            lineKeys.forEach(lineKey => {
                const lineData = pageContent[lineKey];
                if (lineData && typeof lineData.predicted_label === 'string') {
                lines += lineData.predicted_label + "\n";
                } else {
                lines += "\n"; // Add an empty line if no label or incorrect format
                }
            });
        }
        zipWriter.add(`${pageName}.txt`, new zip.TextReader(lines));
        });

        zipWriter.close().then(blob => {
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = `${manuscript}_recognitions.zip`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(link.href); // Clean up blob URL
        }).catch(err => {
            console.error("Error creating zip file:", err);
            alert("Error creating zip file. Check console for details.");
        });
    }

    function reset() {
        modelName.value = null;
        recognitions.value = {};
        userAnnotations.value = [];
        currentPage.value = undefined; // Explicitly set to undefined
        console.log("AnnotationStore: Reset complete.");
    }

    return { 
        // State
        recognitions, 
        userAnnotations, 
        modelName, 
        currentPage, 
        
        // Computed (can be used by components if needed)
        currentManuscriptName, 
        sortedPageIds,

        // Actions
        setCurrentPage,
        nextPage,
        previousPage,
        setInitialPage, // Important for initializing after data load

        // Existing functions
        calculateLevenshteinDistances, 
        exportToTxt, 
        reset 
    };
    });

    // HMR (Hot Module Replacement)
    if (import.meta.hot) {
    import.meta.hot.accept(acceptHMRUpdate(useAnnotationStore, import.meta.hot))
    }


    ================================================
    FILE: frontend/src/views/AnnotationView.vue
    ================================================
    <script setup>
    import { RouterView } from 'vue-router'
    </script>

    <template>
    <div class="annotationView-container">
        <header>
        <h1>Manuscript Annotation Tool</h1>
        </header>
        <RouterView />
    </div>
    </template>

    <style>
    .annotationView-container {
    padding: 1em;
    }
    </style>



    ================================================
    FILE: frontend/src/views/LandingPage.vue
    ================================================
    <script setup>
    import { useAnnotationStore } from '@/stores/annotationStore';
    import { useCssModule } from 'vue'
    import { RouterLink } from 'vue-router'
    const landingPage = useCssModule()
    const annotationStore = useAnnotationStore();
    annotationStore.reset()
    </script>

    <template>
    <div :class="landingPage['landing-container']">
        <header>
        <img src="/flame-logo.svg" alt="Flame logo" :class="landingPage.logo" />
        <h1>Manuscript Annotation Tool</h1>
        </header>
        <main :class="landingPage.main">
        <div :class="landingPage.links">

        <RouterLink :to="{ name: 'new-manuscript' }" class="btn btn-primary m-2">
        New Manuscript / Map
        </RouterLink>

        <RouterLink :to="{ name: 'upload-manuscript' }" class="btn btn-sm btn-secondary text-gray-700 bg-gray-200 m-2">
        old version -- Annotate
        </RouterLink>

        <RouterLink :to="{ name: 'uploaded-manuscripts' }" class="btn btn-sm btn-secondary text-gray-700 bg-gray-200 m-2">
        old version -- Uploaded Manuscripts
        </RouterLink>

        </div>
        </main>
    </div>
    </template>

    <style module>

    .landing-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100vh;
    }

    .logo {
    height: 35vh;
    display: block;
    margin: auto;
    }
    </style>



    ================================================
    FILE: frontend/src/views/new-AnnotationView.vue
    ================================================
    <script setup>
    import { RouterView } from 'vue-router'
    </script>

    <template>
    <div class="annotationView-container">
        <header>
        <h1>Historical OCR Tool</h1>
        </header>
        <RouterView />
    </div>
    </template>

    <style>
    .annotationView-container {
    padding: 1em;
    }
    </style>



    ================================================
    FILE: frontend/src/views/UploadedManuscriptsView.vue
    ================================================
    <script setup>
    import { useAnnotationStore } from '@/stores/annotationStore'
    import { ref } from 'vue'
    import { useRouter } from 'vue-router'

    const annotationStore = useAnnotationStore()
    const router = useRouter()

    const manuscripts = ref([])
    const models = ref([])
    const manuscript_name = ref()
    const model = ref()

    const RECOGNITION_URL = import.meta.env.VITE_BACKEND_URL + '/recognise'

    function fetch_manuscript() {
    fetch(RECOGNITION_URL, {
        method: 'POST',
        headers: {
        'Content-Type': 'application/json',
        },
        body: JSON.stringify({ manuscript_name: manuscript_name.value, model: model.value }),
    })
        .then((response) => response.json())
        .then((object) => {
        const manuscript_name = Object.values(object)[0][0].manuscript_name
        const selected_model = Object.values(object)[0][0].selected_model
        annotationStore.recognitions[manuscript_name] = {}

        for (const page of Object.keys(object)) {
            annotationStore.recognitions[manuscript_name][page] = {}
            for (const line in object[page]) {
            const line_name = object[page][line]['line']
            annotationStore.recognitions[manuscript_name][page][line_name] = {}
            annotationStore.recognitions[manuscript_name][page][line_name]['predicted_label'] =
                object[page][line]['predicted_label']
            annotationStore.recognitions[manuscript_name][page][line_name]['image_path'] =
                object[page][line]['image_path']
            annotationStore.recognitions[manuscript_name][page][line_name]['confidence_score'] =
                object[page][line]['confidence_score']
            }
        }
        annotationStore.userAnnotations.push({
            manuscript_name: manuscript_name,
            selected_model: selected_model,
            annotations: {},
        })

        router.push({ name: 'annotation-section' })
        })
    }

    fetch(import.meta.env.VITE_BACKEND_URL + '/uploaded-manuscripts')
    .then((response) => response.json())
    .then((object) => {
        manuscripts.value = object
        manuscript_name.value = manuscripts.value[0]
    })

    fetch(import.meta.env.VITE_BACKEND_URL + '/models')
    .then((response) => response.json())
    .then((object) => {
        models.value = object
        model.value = models.value[0]
    })
    </script>

    <template>
    <div class="uploadedManuscriptsView-container">
        <header>
        <h1>Manuscript Annotation Tool</h1>
        </header>
        <form
        v-if="manuscripts.length && models.length"
        class="mb-3"
        @submit.prevent="fetch_manuscript"
        >
        <label for="page" class="form-label">Manuscript</label>
        <select
            v-model="manuscript_name"
            class="form-select"
            id="page"
            placeholder="Select a manuscript"
        >
            <option v-for="manuscript in manuscripts" :key="manuscript">
            {{ manuscript }}
            </option>
        </select>
        <label for="model" class="form-label">Model</label>
        <select class="form-select mb-3" id="model" v-model="model" placeholder="Select a model">
            <option disabled hidden value="">Select a model</option>
            <option v-for="model in models" :key="model" :value="model">{{ model }}</option>
        </select>
        <button type="submit" class="btn btn-primary">Find</button>
        </form>
    </div>
    </template>

    <style>
    .uploadedManuscriptsView-container {
    padding: 1em;
    }
    </style>


