# generate.py (Improved Version)

import argparse
import yaml
import numpy as np
import os
from datetime import datetime

import layout_generator
import ambiguous_layouts
import augmentations
import visualizer # Import here

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Synthetic Manuscript Layout Generator")
    parser.add_argument("--config", type=str, default="configs/default_params.yaml", help="Path to the config file.")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of samples to generate.")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save the generated samples.")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility.")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config.get('seed')
    if seed is None:
        seed = np.random.randint(0, 2**32 - 1)
    print(f"Using random seed: {seed}")
    rng = np.random.default_rng(seed)

    for i in range(args.n_samples):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_id = f"sample_{timestamp}_{i:04d}"
        sample_output_dir = os.path.join(args.output_dir, sample_id)

        mode = config.get('generation_mode', 'standard')
        if mode == 'standard':
            page = layout_generator.generate_standard_layout(config, rng)
        elif mode == 'grid':
            page = ambiguous_layouts.generate_grid_layout(config, rng)
        elif mode == 'concentric':
            page = ambiguous_layouts.generate_concentric_layout(config, rng)
        else:
            raise ValueError(f"Unknown generation mode: {mode}")

        augmentations.apply_augmentations(page, config, rng)
        
        # Finalize and save text files, getting back data for visualization
        unnormalized_points, textbox_labels = page.finalize_and_save_text(
            sample_output_dir, sample_id, config, rng
        )

        # Now, call the visualizer from the main script
        image_output_path = os.path.join(sample_output_dir, f"{sample_id}.png")
        visualizer.visualize_page(page, image_output_path, unnormalized_points, textbox_labels)
        print(f"Successfully visualized sample '{sample_id}'")

if __name__ == "__main__":
    main()