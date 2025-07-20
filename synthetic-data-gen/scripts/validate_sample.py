import typer
from pathlib import Path
import yaml
import random

from src.utils.validation import validate_sample_files
from src.utils.visualization import Visualizer
from src.io.file_writer import FileWriter
from src.generators.page_generator import PageGenerator
from config.config_models import Config


app = typer.Typer()

@app.command()
def validate(
    sample_dir: Path = typer.Argument(..., help="Path to the sample directory to validate."),
    rerender: bool = typer.Option(False, "--rerender", help="Re-render the visualization from the data files.")
):
    """
    Validates the integrity of a generated sample's files.
    """
    if not sample_dir.is_dir():
        print(f"Error: Directory not found: {sample_dir}")
        raise typer.Exit(code=1)
        
    print(f"--- Validating Sample: {sample_dir.name} ---")
    
    results = validate_sample_files(sample_dir)
    
    if results["status"] == "OK":
        print("✅ Validation Successful: All files are present and correctly formatted.")
    elif results["status"] == "WARNING":
        print("⚠️ Validation Warning:")
        for warn in results["errors"]:
            print(f"  - {warn}")
    else: # ERROR
        print("❌ Validation Failed:")
        for err in results["errors"]:
            print(f"  - {err}")
    
    if rerender:
        print("\n--- Re-rendering visualization ---")
        # This is a complex task as it requires reversing the generation to get metadata
        # For now, we will just print a message. A full implementation would
        # require saving metadata alongside the sample.
        print("Re-rendering from raw data is not fully supported in this script.")
        print("Use the --dry-run feature of the generator for accurate visualizations.")


if __name__ == "__main__":
    app()