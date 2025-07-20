import typer
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json

app = typer.Typer()

@app.command()
def create_report(
    dataset_dir: Path = typer.Argument(..., help="Path to the root directory of the generated dataset."),
    output_file: Path = typer.Option("dataset_summary.json", help="Path to save the summary report.")
):
    """
    Analyzes a generated dataset and creates a summary report.
    """
    if not dataset_dir.is_dir():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        raise typer.Exit(code=1)

    print(f"Analyzing dataset at: {dataset_dir}")
    
    sample_dirs = [d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith('sample_')]
    if not sample_dirs:
        print("No valid sample directories found.")
        raise typer.Exit()
        
    stats = {
        "total_samples": len(sample_dirs),
        "points_per_page": [],
        "textboxes_per_page": [],
        "textlines_per_page": [],
    }

    for sample_dir in tqdm(sample_dirs, desc="Processing samples"):
        try:
            points_file = sample_dir / "inputs_unnormalized.txt"
            tb_labels_file = sample_dir / "labels_textbox.txt"
            tl_labels_file = sample_dir / "labels_textline.txt"

            if not all([points_file.exists(), tb_labels_file.exists(), tl_labels_file.exists()]):
                continue

            points = np.loadtxt(points_file)
            tb_labels = np.loadtxt(tb_labels_file, dtype=int, ndmin=1)
            tl_labels = np.loadtxt(tl_labels_file, dtype=int, ndmin=1)
            
            stats["points_per_page"].append(points.shape[0])
            
            if tb_labels.size > 0:
                stats["textboxes_per_page"].append(len(np.unique(tb_labels)))
            else:
                 stats["textboxes_per_page"].append(0)

            if tl_labels.size > 0:
                stats["textlines_per_page"].append(len(np.unique(tl_labels)))
            else:
                stats["textlines_per_page"].append(0)

        except Exception as e:
            print(f"\nWarning: Could not process sample {sample_dir.name}: {e}")
            continue

    # Calculate aggregate statistics
    summary = {
        "total_samples": stats["total_samples"],
        "avg_points_per_page": np.mean(stats["points_per_page"]),
        "min_points_per_page": int(np.min(stats["points_per_page"])),
        "max_points_per_page": int(np.max(stats["points_per_page"])),
        "avg_textboxes_per_page": np.mean(stats["textboxes_per_page"]),
        "avg_textlines_per_page": np.mean(stats["textlines_per_page"]),
    }
    
    print("\n--- Dataset Summary ---")
    for key, value in summary.items():
        print(f"{key.replace('_', ' ').title()}: {value:.2f}")

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=4)
        
    print(f"\nReport saved to: {output_file}")


if __name__ == "__main__":
    app()