import os
import json
import numpy as np
from .visualizer import render_page
from ..core.classes import PageData
from ..core.utils import normalize_page_data

class DataWriter:
    """Handles saving all generated artifacts for a sample to disk."""

    def __init__(self, output_dir: str, visualize_cfg: dict):
        self.output_dir = output_dir
        self.visualize_cfg = visualize_cfg
        os.makedirs(self.output_dir, exist_ok=True)

    def save_sample(self, page_data: PageData):
        """Saves all files for a single PageData object."""
        sample_dir = os.path.join(self.output_dir, page_data.sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        points_unnormalized = page_data.points_global
        points_normalized = normalize_page_data(points_unnormalized, page_data.page_dims)

        # Save text files
        np.savetxt(os.path.join(sample_dir, "inputs_unnormalized.txt"), points_unnormalized, fmt="%.2f %.2f %.2f")
        np.savetxt(os.path.join(sample_dir, "inputs_normalized.txt"), points_normalized, fmt="%.6f %.6f %.6f")
        np.savetxt(os.path.join(sample_dir, "labels_textbox.txt"), page_data.labels_textbox, fmt="%d")
        np.savetxt(os.path.join(sample_dir, "labels_textline.txt"), page_data.labels_textline, fmt="%d")

        # Save metadata
        with open(os.path.join(sample_dir, "meta.json"), 'w') as f:
            # Convert numpy types to native python types for JSON serialization
            meta_serializable = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in page_data.meta.items()}
            json.dump(meta_serializable, f, indent=4)

        # Save visualization
        if self.visualize_cfg['visualize']:
            png_path = os.path.join(sample_dir, f"{page_data.sample_id}.png")
            render_page(
                points=points_unnormalized,
                labels_textbox=page_data.labels_textbox,
                labels_textline=page_data.labels_textline,
                page_dims=page_data.page_dims,
                output_path=png_path,
                config=self.visualize_cfg
            )

    def save_summary(self, summary: dict):
        """Saves the final dataset summary report."""
        report_path = os.path.join(self.output_dir, "dataset_summary.json")
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"\nDataset summary report saved to {report_path}")