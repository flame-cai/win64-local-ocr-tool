# /manuscript_generator/generator.py

import yaml
import time
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import collections

from manuscript_generator.configs.base_config import Config
from manuscript_generator.core.registry import LAYOUT_STRATEGIES
from manuscript_generator.utils.distribution_sampler import sample_from_distribution
from manuscript_generator.utils.plotter import visualize_page

class SyntheticManuscriptGenerator:
    """Orchestrates the entire data generation process."""

    def __init__(self, config: Config):
        self.config = config
        self.output_dir = Path(self.config.generation.output_dir)
        self.stats = collections.defaultdict(int)
        self.page_aspect_ratios = []

    def _generate_single_sample(self, sample_index: int, master_seed: int):
        """Generates, saves, and optionally visualizes a single sample (or a pair for grid)."""
        # Create a unique, reproducible random state for this sample
        seed = master_seed + sample_index
        rng = np.random.default_rng(seed)

        try:
            # 1. Select layout strategy
            strategy_name = sample_from_distribution(self.config.page.layout_strategy, rng)
            layout_strategy_func = LAYOUT_STRATEGIES[strategy_name]
            
            # 2. Generate page(s) using the strategy
            pages = layout_strategy_func(self.config, rng)
            
            # 3. Finalize and save each page
            for i, page in enumerate(pages):
                sample_id = f"sample_{sample_index:06d}"
                if len(pages) > 1: # For ambiguous layouts like grid
                    sample_id += f"_{i}"

                # Finalize (combine textboxes, apply phase 3 augs)
                page.finalize(self.config, rng)

                # Save all files
                page.save(self.output_dir, sample_id)

                # Update stats
                self.stats['total_pages'] += 1
                self.stats['total_points'] += page.points.shape[0]
                self.stats[f'pages_from_{strategy_name}'] += 1
                for box in page.textboxes:
                    self.stats[f'textbox_{box.box_type.value}'] += 1
                if page.height > 0:
                    self.page_aspect_ratios.append(page.width / page.height)


                # 4. Visualize if enabled
                is_dry_run = self.config.generation.num_samples <= self.config.generation.dry_run_num_samples
                if self.config.visualization.enabled:
                    if not self.config.visualization.render_on_dry_run_only or is_dry_run:
                        viz_path = self.output_dir / sample_id / f"{sample_id}.png"
                        visualize_page(page, self.config.visualization, viz_path)

            return None # Success
        except Exception as e:
            return f"Error generating sample {sample_index}: {e}"

    def generate_dataset(self, dry_run: bool = False):
        """
        Generates the full dataset.
        
        Args:
            dry_run: If True, generates a small number of samples for testing.
        """
        start_time = time.time()
        num_samples = self.config.generation.dry_run_num_samples if dry_run else self.config.generation.num_samples
        master_seed = self.config.generation.base_seed

        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Starting dataset generation...")
        print(f" - Mode: {'Dry Run' if dry_run else 'Full Generation'}")
        print(f" - Number of samples to generate: {num_samples}")
        print(f" - Output directory: {self.output_dir.resolve()}")
        
        num_workers = self.config.generation.num_workers
        if num_workers == -1:
            num_workers = cpu_count()
        
        worker_func = partial(self._generate_single_sample, master_seed=master_seed)

        if num_workers > 1:
            print(f" - Using {num_workers} parallel workers.")
            with Pool(processes=num_workers) as pool:
                results = list(tqdm(pool.imap(worker_func, range(num_samples)), total=num_samples, desc="Generating Samples"))
        else:
            print(" - Using a single process (sequential).")
            results = [worker_func(i) for i in tqdm(range(num_samples), desc="Generating Samples")]
        
        errors = [r for r in results if r is not None]
        if errors:
            print(f"\nEncountered {len(errors)} errors during generation:")
            for error in errors[:5]: # Print first 5 errors
                print(f"  - {error}")
        
        duration = time.time() - start_time
        print(f"\nDataset generation completed in {duration:.2f} seconds.")
        
        self.print_summary_report()

    def print_summary_report(self):
        """Prints an aggregate statistics report for the generated dataset."""
        print("\n--- Dataset Summary Report ---")
        if self.stats['total_pages'] == 0:
            print("No pages were generated.")
            return

        print(f"Total Pages Generated: {self.stats['total_pages']}")
        print(f"Total Points Generated: {self.stats['total_points']:,}")
        avg_points = self.stats['total_points'] / self.stats['total_pages']
        print(f"Average Points per Page: {avg_points:,.2f}")
        
        print("\nLayout Strategy Distribution:")
        for key, val in self.stats.items():
            if key.startswith("pages_from_"):
                strategy = key.replace("pages_from_", "")
                percentage = (val / self.stats['total_pages']) * 100
                print(f"  - {strategy}: {val} pages ({percentage:.1f}%)")

        print("\nTextBox Type Distribution:")
        total_boxes = sum(v for k, v in self.stats.items() if k.startswith("textbox_"))
        if total_boxes > 0:
            for key, val in self.stats.items():
                if key.startswith("textbox_"):
                    box_type = key.replace("textbox_", "")
                    percentage = (val / total_boxes) * 100
                    print(f"  - {box_type}: {val} boxes ({percentage:.1f}%)")
            avg_boxes = total_boxes / self.stats['total_pages']
            print(f"Average TextBoxes per Page: {avg_boxes:.2f}")

        print("\nPage Aspect Ratio (Width/Height):")
        if self.page_aspect_ratios:
            ratios = np.array(self.page_aspect_ratios)
            print(f"  - Min: {np.min(ratios):.2f}")
            print(f"  - Mean: {np.mean(ratios):.2f}")
            print(f"  - Max: {np.max(ratios):.2f}")
        print("-----------------------------\n")