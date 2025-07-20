import typer
import yaml
from pathlib import Path
from typing_extensions import Annotated
import random
from tqdm import tqdm
import multiprocessing

from ..config.config_models import Config
from .generators.page_generator import PageGenerator
from .io.file_writer import FileWriter
from .utils.visualization import Visualizer

app = typer.Typer(help="Synthetic Manuscript Layout Generator CLI")

def generate_sample_worker(args):
    """Worker function for multiprocessing."""
    worker_id, output_dir, config_path, base_seed, dry_run, vis_config = args
    
    # Ensure each process has a unique random state
    # but derived deterministically from the base seed
    seed = base_seed + worker_id if base_seed is not None else None
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = Config(**config_dict)

    # Override visualization if disabled by CLI
    if not vis_config['enabled']:
        config.visualization.enabled = False

    random_state = random.Random(seed)
    page_generator = PageGenerator(config, random_state)
    
    generated_pages = page_generator.generate_page()
    
    if dry_run:
        if generated_pages:
            print(f"\n--- [DRY RUN] Worker {worker_id} (Seed: {seed}) ---")
            first_page = generated_pages[0]
            print(f"  Layout Strategy: {first_page.layout_strategy}")
            print(f"  Page Dimensions: {first_page.page_width:.2f} x {first_page.page_height:.2f}")
            print(f"  Total Points: {len(first_page.points)}")
            print(f"  Number of TextBoxes: {len(first_page.text_boxes)}")
            
            visualizer = Visualizer(config.visualization)
            dry_run_path = Path(output_dir) / f"dry_run_sample_s{seed}.png"
            dry_run_path.parent.mkdir(parents=True, exist_ok=True)
            visualizer.visualize_page(first_page, dry_run_path)
            print(f"  Visualization saved to: {dry_run_path}")
        return None

    writer = FileWriter(output_dir)
    saved_paths = []
    for page in generated_pages:
        # Each page needs a unique ID, even from the same geometric generation
        sample_id = f"sample_{base_seed}_{worker_id}_{page.sub_id}"
        writer.save_sample(page, sample_id)
        saved_paths.append(output_dir / sample_id)
    return saved_paths


@app.command()
def generate(
    num_samples: Annotated[int, typer.Option(help="Number of samples to generate.")] = 10,
    output_dir: Annotated[Path, typer.Option(help="Directory to save the generated samples.")] = Path("output"),
    config_path: Annotated[Path, typer.Option(help="Path to the YAML configuration file.")] = Path("config/default_config.yaml"),
    seed: Annotated[int, typer.Option(help="Master random seed for reproducibility.")] = 42,
    dry_run: Annotated[bool, typer.Option(help="Run for one sample, print metadata, and save only the visualization.")] = False,
    no_vis: Annotated[bool, typer.Option(help="Disable visualization file generation.")] = False,
    workers: Annotated[int, typer.Option(help="Number of parallel processes to use.")] = 1,
):
    """
    Generate a dataset of synthetic manuscript layouts.
    """
    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}")
        raise typer.Exit(code=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualization config dict to pass to workers
    vis_config = {'enabled': not no_vis}

    if dry_run:
        print("Executing in --dry-run mode. Generating one sample preview.")
        generate_sample_worker((0, output_dir, config_path, seed, True, vis_config))
        return

    print(f"Generating {num_samples} samples using {workers} worker(s)...")
    print(f"Output directory: {output_dir}")
    print(f"Using configuration: {config_path}")
    print(f"Master seed: {seed}")

    tasks = [(i, output_dir, config_path, seed, False, vis_config) for i in range(num_samples)]
    
    if workers > 1:
        with multiprocessing.Pool(processes=workers) as pool:
            list(tqdm(pool.imap(generate_sample_worker, tasks), total=num_samples, desc="Generating Samples"))
    else:
        # Single-threaded execution for easier debugging
        for task in tqdm(tasks, desc="Generating Samples"):
            generate_sample_worker(task)

    print("\nDataset generation complete.")


if __name__ == "__main__":
    app()