# /manuscript_generator/utils/plotter.py

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from manuscript_generator.core.page import Page
from manuscript_generator.configs.base_config import VisualizationConfig

# Colorblind-friendly palette
COLORS = [
    '#4363d8', '#f58231', '#ffe119', '#3cb44b', '#e6194B',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
    '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000',
    '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9'
]

def visualize_page(page: Page, config: VisualizationConfig, output_path: Path):
    """Generates and saves a PNG visualization of the page."""
    if page.points.shape[0] == 0:
        return

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10 * page.width / page.height, 10))

    if config.coloring == "textbox":
        labels = page.textbox_labels
    elif config.coloring == "textline":
        labels = page.textline_labels
    else:
        raise ValueError(f"Invalid coloring option: {config.coloring}")

    unique_labels = np.unique(labels)
    colors = [COLORS[label % len(COLORS)] for label in labels]

    # Point sizes are proportional to their font size
    # We need to scale them appropriately for the plot
    avg_font_size = np.mean(page.points[:, 2])
    sizes = (page.points[:, 2] / avg_font_size) * config.point_size_multiplier * 10

    ax.scatter(page.points[:, 0], page.points[:, 1], c=colors, s=sizes, marker='.')
    
    # Set page boundaries
    ax.set_xlim(0, page.width)
    ax.set_ylim(0, page.height)
    ax.set_aspect('equal', adjustable='box')
    
    # Invert Y-axis to have origin at top-left, common for images
    ax.invert_yaxis()
    
    # Add page border
    rect = plt.Rectangle((0, 0), page.width, page.height, linewidth=2, edgecolor='w', facecolor='none')
    ax.add_patch(rect)
    
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)