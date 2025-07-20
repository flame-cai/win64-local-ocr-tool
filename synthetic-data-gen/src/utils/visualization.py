import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

from ..core.data_structures import GeneratedPage, TextBoxBlueprint
from ..config.config_models import VisualizationConfig


class Visualizer:
    def __init__(self, config: VisualizationConfig):
        self.config = config

    def visualize_page(self, page: GeneratedPage, output_path: Path):
        """Renders the generated page to a PNG file."""
        if not self.config.enabled:
            return

        fig, ax = plt.subplots(figsize=(10, 15), dpi=self.config.dpi)
        
        # Set plot limits based on page dimensions, with a small margin
        margin = 50
        ax.set_xlim(-page.page_width / 2 - margin, page.page_width / 2 + margin)
        ax.set_ylim(-page.page_height / 2 - margin, page.page_height / 2 + margin)
        ax.set_aspect('equal', adjustable='box')
        ax.invert_yaxis() # Top-left origin
        ax.set_facecolor('lightgrey')

        # Get colors for each point based on textbox ID
        colors = self._get_point_colors(page.textbox_labels, page.text_boxes)

        # Plot all points
        if page.points.shape[0] > 0:
            ax.scatter(page.points[:, 0], page.points[:, 1], c=colors, s=self.config.point_size, marker='.')

        # Draw OBBs if enabled
        if self.config.draw_obbs:
            for bp in page.text_boxes:
                self._draw_obb(ax, bp)
        
        # Draw textline boundaries if enabled
        if self.config.draw_textlines:
             self._draw_textlines(ax, page)


        plt.title(f"Layout: {page.layout_strategy} | Seed: {page.seed}")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)

    def _get_point_colors(self, textbox_labels, blueprints: list[TextBoxBlueprint]):
        if textbox_labels.shape[0] == 0:
            return []
        
        # Create a mapping from textbox_idx to color
        idx_to_color = {}
        for i, bp in enumerate(blueprints):
            box_type_str = bp.box_type.value
            color = self.config.color_map.get(box_type_str, self.config.color_map['default'])
            idx_to_color[i] = color
            
        return [idx_to_color[label] for label in textbox_labels]

    def _draw_obb(self, ax, blueprint: TextBoxBlueprint):
        obb_corners = blueprint.get_obb()
        color = self.config.color_map.get(blueprint.box_type.value, self.config.color_map['default'])
        polygon = patches.Polygon(obb_corners, closed=True, fill=False, edgecolor=color, linewidth=1, linestyle='--')
        ax.add_patch(polygon)
    
    def _draw_textlines(self, ax, page: GeneratedPage):
        """Draws convex hulls of points for each textline."""
        from scipy.spatial import ConvexHull
        
        unique_line_ids = np.unique(page.textline_labels)
        line_colors = plt.cm.get_cmap('viridis', len(unique_line_ids))

        for i, line_id in enumerate(unique_line_ids):
            points_in_line = page.points[page.textline_labels == line_id]
            if points_in_line.shape[0] >= 3:
                try:
                    hull = ConvexHull(points_in_line[:, :2])
                    for simplex in hull.simplices:
                        ax.plot(points_in_line[simplex, 0], points_in_line[simplex, 1], color=line_colors(i), lw=0.5)
                except Exception:
                    # ConvexHull can fail for collinear points
                    pass