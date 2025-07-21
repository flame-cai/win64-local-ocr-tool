import argparse
import yaml
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm
from pydantic import ValidationError

from .config import AppConfig
from .generation.page_generator import PageGenerator
from .io.writer import DataWriter

def load_config(config_path: str) -> AppConfig:
    """Loads and validates the YAML configuration file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    try:
        return AppConfig(**config_dict)
    except ValidationError as e:
        print("Configuration Error:")
        print(e)
        exit(1)

def generate_single_sample(seed: int, config: AppConfig) -> List[Dict[str, Any]]:
    """Worker function to generate one geometric sample (which may yield multiple PageData objects)."""
    page_gen = PageGenerator(config, seed)
    results = []
    # A single call might yield multiple PageData objects (for ambiguous layouts)
    for page_data in page_gen.generate_page():
        results.append(page_data)
    return results

def main_cli():
    parser = argparse.ArgumentParser(description="Synthetic Manuscript Data Generator")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--output-dir", type=str, help="Directory to save the generated samples (overrides config).")
    parser.add_argument("--num-samples", type=int, help="Number of samples to generate (overrides config).")
    parser.add_argument("--num-workers", type=int, help="Number of parallel workers (overrides config).")
    parser.add_argument("--seed", type=int, help="Master random seed (overrides config).")
    parser.add_argument("--dry-run", action="store_true", help="Generate 10 samples with visualization for testing.")
    args = parser.parse_args()

    config = load_config(args.config)

    # Override config with CLI arguments
    if args.output_dir: config.generation.output_dir = args.output_dir
    if args.num_samples: config.generation.num_samples = args.num_samples
    if args.num_workers: config.generation.num_workers = args.num_workers
    if args.seed: config.generation.seed = args.seed
    if args.dry_run:
        config.generation.dry_run = True
        config.generation.num_samples = 10
        config.output.visualize = True
        print("--- Running in Dry-Run mode ---")

    output_dir = config.generation.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    num_workers = config.generation.num_workers
    if num_workers == -1:
        num_workers = os.cpu_count() or 1

    print(f"Starting generation of {config.generation.num_samples} samples using {num_workers} workers...")
    print(f"Output directory: {output_dir}")
    print(f"Master seed: {config.generation.seed}")

    start_time = time.time()
    
    # Create a seed sequence to ensure workers have different, but reproducible, seeds
    seed_sequence = np.random.SeedSequence(config.generation.seed)
    worker_seeds = seed_sequence.spawn(config.generation.num_samples)

    writer = DataWriter(output_dir, config.output.model_dump())
    
    all_meta_data = []
    sample_id_counter = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(generate_single_sample, seed, config): seed for seed in worker_seeds}
        
        progress = tqdm(as_completed(futures), total=config.generation.num_samples, desc="Generating Pages")
        for future in progress:
            try:
                page_data_list = future.result()
                for page_data in page_data_list:
                    if page_data.points_global.shape[0] > 0:
                        page_data.sample_id = f"sample_{sample_id_counter:06d}"
                        writer.save_sample(page_data)
                        all_meta_data.append(page_data.meta)
                        sample_id_counter += 1
            except Exception as e:
                print(f"A worker failed with error: {e}")
    
    # --- Generate Summary Report ---
    if all_meta_data:
        summary = {
            "total_samples_generated": len(all_meta_data),
            "generation_config": config.generation.model_dump(),
            "layout_strategy": config.layout.strategy,
        }
        # Add more stats here as needed
        writer.save_summary(summary)

    end_time = time.time()
    print(f"\nGeneration complete. Total time: {end_time - start_time:.2f} seconds.")
    print(f"Successfully generated {len(all_meta_data)} samples.")

if __name__ == "__main__":
    main_cli()