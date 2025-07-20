# Synthetic Manuscript Layout Generator

This project provides a highly configurable, modular, and ablation-friendly Python tool for generating synthetic page layouts of handwritten manuscripts. The output is a point cloud representing characters, along with ground truth labels for textboxes and text lines, suitable for training and evaluating machine learning models for document layout analysis.

## Core Design Principles

-   **Domain Randomization:** Every variable parameter of the generation process (e.g., page size, number of textboxes, spacing, orientation) is sampled from a pre-defined probability distribution to ensure high variance in the output data.
-   **Modularity:** The code is organized into logical, decoupled components (TextBox types, layout strategies, augmentations, visualizer). This allows for easy extension, debugging, and maintenance.
-   **Ablation-Friendly:** Augmentations and layout features can be easily enabled or disabled via a configuration file, facilitating ablation studies.
-   **Scalability:** The generation process for a single page is self-contained, allowing for trivial parallelization to generate large datasets.
-   **Reproducibility:** All stochastic processes are controlled by a single random seed for perfect reproducibility.

## Features

-   **Hybrid OOP-to-Vectorized Workflow:** Uses an object-oriented approach for logical content creation and switches to high-performance NumPy for all geometric transformations.
-   **Strict Coordinate System Management:** Explicitly separates Local, Global, and Normalized coordinate systems to prevent bugs and ensure clarity.
-   **Pluggable Architecture:** New layout strategies and augmentations can be added using a simple decorator-based registry system without modifying core code.
-   **Multiple Layout Strategies:** Includes standard "rejection sampling" for realistic layouts and specialized "grid" and "concentric" generators for creating ambiguous test cases.
-   **Cascaded Augmentation Pipeline:** Augmentations are applied in three distinct phases:
    1.  **Content & Micro-Variations:** Font size, spacing, text alignment, local jitter.
    2.  **Geometric Distortion:** Shear, Stretch, Warp/Curl on a per-textbox basis.
    3.  **Page-Level Effects:** Global jitter, point dropout (missing characters).
-   **Pydantic Configuration:** Uses Pydantic for robust, type-safe configuration loading and validation from a `YAML` file.
-   **Comprehensive Tooling:**
    -   A `typer`-based CLI for easy control over generation.
    -   A `--dry-run` mode for quick testing of configuration changes.
    -   Optional visualization of generated pages with debug overlays (OBBs, text lines).
    -   A validation script to check the integrity of generated samples.
    -   A summary report script to get aggregate statistics over a dataset.

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd synthetic_data_generator
    ```

2.  Create and activate a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Generating a Dataset

The main entry point is `scripts/generate_dataset.py`. Use the `--help` flag to see all options.

```bash
python scripts/generate_dataset.py --help
```

Example: Generate 100 samples in the output/my_dataset directory with a specific seed.
Generated bash
python scripts/generate_dataset.py --num-samples 100 --output-dir output/my_dataset --seed 42
Use code with caution.
Bash
Dry Run (Testing a Config)
To generate a single sample, save its visualization, and print metadata to the console without creating the full data files, use the --dry-run flag. This is useful for quickly previewing the effect of configuration changes.
Generated bash
python scripts/generate_dataset.py --dry-run --config-path config/default_config.yaml
Use code with caution.
Bash
Validating a Sample
You can validate the format and integrity of a single generated sample.

Generated bash
python scripts/validate_sample.py output/my_dataset/sample_00000
Use code with caution.
Bash


Creating a Dataset Summary
After generating a dataset, you can create a summary report.
Generated bash
```
python scripts/create_summary_report.py output/my_dataset
```
