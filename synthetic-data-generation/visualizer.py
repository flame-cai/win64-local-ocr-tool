# visualizer.py (Corrected)

import matplotlib.pyplot as plt
import numpy as np

# Use a qualitative colormap for good visual distinction between textboxes
# We add black at the start for any potential background points (ID -1)
# FIXED: Convert the color tuples to lists before concatenating.
COLOR_MAP = ['#000000'] + list(plt.cm.get_cmap('tab20').colors) + list(plt.cm.get_cmap('tab20b').colors)

def visualize_page(page, output_path, final_points, textbox_labels):
    """
    Renders the page layout to a PNG file.
    
    Args:
        page (Page): The page object with all its contents.
        output_path (str): Path to save the PNG file.
        final_points (list[tuple]): List of (x, y, font_size) for all points.
        textbox_labels (list[int]): List of textbox IDs for each point.
    """
    fig, ax = plt.subplots(figsize=(10, 10 * (page.height / page.width)))
    ax.set_facecolor('white')

    if not final_points:
        print("Warning: No points to visualize.")
        # Save an empty image
        ax.set_xlim(0, page.width)
        ax.set_ylim(0, page.height)
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    points_arr = np.array(final_points)
    labels_arr = np.array(textbox_labels)

    # Plot points, color-coded by textbox ID
    scatter = ax.scatter(
        points_arr[:, 0],
        points_arr[:, 1],
        s=(points_arr[:, 2] / 5)**2,  # Scale marker size by font size
        c=[COLOR_MAP[label % len(COLOR_MAP)] for label in labels_arr],
        alpha=0.8,
        linewidths=0
    )

    # For debugging: plot the oriented bounding boxes
    for textbox in page.textboxes:
        poly = textbox.get_oriented_bounding_box()
        x, y = poly.exterior.xy
        ax.plot(x, y, color=COLOR_MAP[textbox.box_id % len(COLOR_MAP)], linestyle='--', linewidth=1.5, alpha=0.7)


    ax.set_xlim(0, page.width)
    ax.set_ylim(0, page.height)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis() # Top-left origin, common in image processing
    plt.axis('off')
    plt.tight_layout(pad=0)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)