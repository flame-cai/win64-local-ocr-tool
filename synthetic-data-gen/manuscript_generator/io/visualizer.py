import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def render_page(points: np.ndarray, labels_textbox: np.ndarray, labels_textline: np.ndarray, 
                page_dims: Tuple[float, float], output_path: str, config: dict):
    """
    Renders the point cloud of a page and saves it as a PNG.
    """
    if points.shape[0] == 0:
        return # Do not render empty pages

    page_width, page_height = page_dims
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(page_width / config['dpi'], page_height / config['dpi']), dpi=config['dpi'])
    fig.patch.set_facecolor(config['background_color'])
    ax.set_facecolor(config['background_color'])

    # Determine color scheme
    if config['color_by'] == 'textline':
        labels = labels_textline
        cmap = plt.cm.get_cmap('hsv', np.max(labels) + 1)
    else: # Default to textbox
        labels = labels_textbox
        cmap = plt.cm.get_cmap('viridis', np.max(labels) + 1)
        
    # Scatter plot
    scatter = ax.scatter(
        points[:, 0], 
        points[:, 1], 
        c=labels,
        s=points[:, 2] * config['point_size_multiplier'],
        cmap=cmap,
        alpha=0.8
    )

    ax.set_xlim(0, page_width)
    ax.set_ylim(0, page_height)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis() # (0,0) is top-left
    plt.axis('off')

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)