from pathlib import Path
import numpy as np

from ..core.data_structures import GeneratedPage
from ..core.coordinate_systems import normalize_page_coordinates
from ...config.config_models import Config
from ..utils.visualization import Visualizer

class FileWriter:
    """Handles writing all output files for a generated sample."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_sample(self, page: GeneratedPage, sample_id: str, config: Config = None):
        """
        Saves all files for a single generated page.

        Args:
            page (GeneratedPage): The completed page data.
            sample_id (str): A unique identifier for the sample (e.g., "sample_00001").
            config (Config, optional): The configuration used for generation,
                                       needed for visualization.
        """
        sample_path = self.output_dir / sample_id
        sample_path.mkdir(exist_ok=True)
        
        # Save unnormalized inputs
        np.savetxt(sample_path / "inputs_unnormalized.txt", page.points, fmt="%.2f %.2f %d")
        
        # Normalize and save normalized inputs
        points_normalized = normalize_page_coordinates(page.points, page.page_width, page.page_height)
        np.savetxt(sample_path / "inputs_normalized.txt", points_normalized, fmt="%.6f")

        # Save labels
        np.savetxt(sample_path / "labels_textbox.txt", page.textbox_labels, fmt="%d")
        np.savetxt(sample_path / "labels_textline.txt", page.textline_labels, fmt="%d")
        
        # Save visualization if a config is provided and visualization is enabled
        if config and config.visualization.enabled:
            visualizer = Visualizer(config.visualization)
            vis_path = sample_path / f"{sample_id}.png"
            visualizer.visualize_page(page, vis_path)