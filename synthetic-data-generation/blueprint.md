# Project Blueprint: Synthetic Manuscript Layout Generator (v2.1 FINAL)

## 1. Overview

This document outlines the specifications for a Python-based synthetic data generator. The goal is to create diverse, script-agnostic page layouts of handwritten manuscripts for training and evaluating machine learning models for document layout analysis.

Each character on a page is abstracted as a `Point` with three features: `(x, y, font_size)`. The generator will produce these point clouds along with corresponding ground-truth labels for textboxes and text lines, enabling supervised learning.

The system is designed with core principles of Domain Randomization, modularity, and scalability to generate millions of unique samples.

## 2. Core Design Principles

-   **Domain Randomization:** Every variable parameter of the generation process (e.g., page size, number of textboxes, spacing, orientation) will be sampled from a pre-defined probability distribution to ensure high variance in the output data.
-   **Modularity:** The code will be organized into logical, decoupled components (structures, generators, augmentations, visualizer). This allows for easy logging, debugging, maintenance and extension.
-   **Ablation-Friendly:** Augmentations and specific layout features will be implemented in a way that allows them to be easily enabled or disabled via a configuration file, facilitating ablation studies on model performance.
-   **Scalability:** The generation process for a single page will be self-contained, allowing for trivial parallelization across multiple CPU cores to generate large datasets.
-   **Reproducibility:** All stochastic processes will be controlled by a single random seed to ensure that any generated sample can be perfectly reproduced for debugging and validation.


## 3. System Architecture

### 3.1. File Structure

```
manuscript-generator/
├── generate.py                # Main script to run generation jobs (accepts --seed)
├── layout_generator.py        # Core logic for standard layouts
├── ambiguous_layouts.py       # Logic for special grid/circular layouts
├── structures.py              # Class definitions (Point, Word, TextLine, TextBox, Page)
├── augmentations.py           # Functions for data augmentation (warp, missing chars)
├── visualizer.py              # Code to plot and save a visual representation of a page
├── configs/
│   └── default_params.yaml    # Defines all probability distributions and parameters
└── output/
    └── {sample_id}/
        ├── input_unnormalized.txt
        ├── input_normalized.txt
        ├── labels_textbox.txt
        └── labels_textline.txt
        └── {sample_id}.png
```

input_unnormalized.txt should look like this (with x, y, and font_size)
```
700 752 28
462 754 28
418 755 28
376 756 34
so on...
```
### 3.2. Class Definitions (`structures.py`)

-   **`Point(x, y, font_size)`**: The fundamental unit. Stores unnormalized local coordinates and font size.
-   **`Word(points: list[Point])`**: A collection of `Point` objects.
-   **`TextLine(words: list[Word])`**: A collection of `Word` objects.
-   **`TextBox(text_lines: list[TextLine], box_type: str, position: tuple, orientation_deg: float, width: float, height: float)`**: The primary container.
    -   `box_type`: A string identifier, e.g., `'main_text'`, `'marginalia'`, `'interlinear_gloss'`.
    -   `position`: The `(x, y)` coordinates of the center of the un-rotated box on the page.
    -   `orientation_deg`: The rotation of the box in degrees.
    -   `width`, `height`: The dimensions of the un-rotated bounding box.
    -   **Method `get_oriented_bounding_box()`**: Returns the 4 corner coordinates after rotation and translation. Used for collision detection.
    -   **Method `collides_with(other_box)`**: Implements the Separating Axis Theorem (SAT) to check for overlap.
-   **`Page(width: float, height: float, textboxes: list[TextBox])`**: The top-level object.
    -   **Maintains a global counter for `textline_id`** to ensure uniqueness across all textboxes on the page.
    -   **Method `add_textbox(textbox)`**: Attempts to add a textbox, first checking for collisions.
    -   **Method `apply_augmentations(config)`**: Applies a sequence of augmentation functions.
    -   **Method `finalize_and_save(output_dir, sample_id)`**: Performs point transformation, normalization, and writes all output files.

### 3.3. Generation Pipeline

The generation of a single page follows these steps:
1.  **Initialize Page**: Sample page aspect ratio. Set a large base height (e.g., 1000 units).
2.  **Generate TextBoxes**: Generate and attempt to place a main textbox and a variable number of ancillary textboxes. Placement uses **rejection sampling**: a random position is chosen, and if it results in a collision, a new position is sampled up to a maximum number of attempts.
3.  **Apply Augmentations**: The complete `Page` object is passed to a pipeline of augmentation functions.
4.  **Finalize and Save**: This crucial final step ensures all outputs are consistent. The `Page.finalize_and_save()` method will:
    a. Initialize empty lists for final points (normalized and unnormalized) and labels.
    b. Iterate through its `textboxes` in a fixed order. For each `textbox`:
    c. Iterate through its `text_lines`. For each `text_line`:
    d. Iterate through its `words` and `points`. For each `point`:
        i. Apply the parent textbox's rotation and translation to the point's local coordinates to get the final page coordinates.
        ii. Append the unnormalized point, normalized point, textbox ID, and unique text-line ID to their respective lists.
    e. Write the contents of these finalized, consistently-ordered lists to the output `.txt` files.

## 4. Core Generation Logic & Parameters

Generation is controlled by a configuration file (`default_params.yaml`).

| Parameter | Distribution & Example | Description |
| :--- | :--- | :--- |
| **Page Aspect Ratio** | `Uniform(0.7, 1.4)` | Defines page shape. |
| **Number of Ancillary Boxes** | `Poisson(lambda=2)` | Number of textboxes besides the main one. |
| **TextBox Type** | `Categorical/Choice` | `p={'marginalia':0.6, ...}` |
| **TextBox Orientation** | `Uniform(-5, 5)` for main, `Uniform(-180, 180)` for others. | Angle of rotation. |
| **Base Font Size** | `LogNormal(mu=2.5, sigma=0.3)` | Base font size for a textbox. Unnormalized. |
| **Character Spacing** | `Normal(mu=0.8, sigma=0.1)` | Multiplier of font size. |
| **Line Spacing** | `Normal(mu=1.5, sigma=0.2)` | Multiplier of font size. **Note:** The mean (`μ`) must be kept significantly larger than the mean of `Character Spacing` to ensure layouts are generally readable, with congestion emerging from the variance (`σ`). |
| **Positional Jitter (X, Y)**| `Normal(mu=0, sigma=0.05)` | Per-character positional noise, multiplier of font size. |
| **Line Break Position** | `Uniform(0.7, 1.0)` | A line ends after filling this fraction of its width. |
| **Content Counts** | `Poisson(lambda)` | Controls chars/word, words/line, lines/box. e.g. `words_per_line: Poisson(lambda=8)`. |

## 5. Data Augmentations

Implemented in `augmentations.py`. Their application will be controlled by boolean flags in the config file.

-   **Textbox-Level Warp/Curl**: `warp_textbox(textbox, config)` applies a non-linear sinusoidal transformation to the points *within* a textbox to simulate local paper curl.
-   **Missing Characters**: `remove_points(page, probability)` iterates through all final points and deletes each one with a given `probability`.
-   **Interlinear Gloss**: Implemented as a layout feature in `layout_generator.py`. There is a probability of generating a small `interlinear_gloss` textbox positioned relative to a line in the `main_text`.

## 6. Special Ambiguous Layouts (`ambiguous_layouts.py`)

These generators are designed to create challenging cases for layout analysis algorithms by making local neighbor relationships ambiguous.

### 6.1. Grid Layout

-   **Goal**: Create a perfect grid of points where horizontal and vertical spacing are equal, making the reading order (horizontal vs. vertical) impossible to infer from spacing alone.
-   **Generation**: Generate points at `(x_0 + i*S, y_0 + j*S)` for `i` in `range(N)` and `j` in `range(M)`, where `S` is the constant spacing.
-   **Labeling Interpretation**: To generate ground truth, **one interpretation must be chosen**. The default interpretation will be that **text lines run horizontally**. Therefore, all points with the same `j` index will be grouped into a single `TextLine` and share the same `textline_id`. The entire grid will belong to a single `TextBox`.

### 6.2. Concentric Circles Layout

-   **Goal**: Create points arranged in concentric circles where the spacing between points along a circle's circumference is approximately equal to the radial spacing between circles. This creates ambiguity between a "circular" reading order and a "radial" (spoke-like) reading order.
-   **Generation**: Iterate through radii `r` from a starting radius `r_0` with a step `S`. For each `r`, calculate the number of points `N = floor(2*pi*r / S)` that can fit on the circumference with an arc length of approximately `S`. Place these `N` points evenly around the circle of radius `r`.
-   **Labeling Interpretation**: As with the grid, **one interpretation must be chosen for labeling**. The default interpretation will be that **each circle constitutes a text line**. Therefore, all points on the circle with radius `r` will be grouped into a single `TextLine` and share the same `textline_id`. The entire layout will belong to a single `TextBox`.

## 7. Output Specification

For each sample, a directory `{sample_id}` is created with:
-   **`input_unnormalized.txt`**: `x y font_size` in raw generation units.
-   **`input_normalized.txt`**: `x_norm y_norm font_size_norm`.
    -   `x_norm`, `y_norm`: Coords scaled such that the longest page dimension maps to `[0, 1]`.
    -   `font_size_norm`: Font size divided by page height.
-   **`labels_textbox.txt`**: `textbox_id` (integer).
-   **`labels_textline.txt`**: `textline_id` (globally unique integer).
-   **`{sample_id}.png`**: Visualization of the page, color-coded by textbox ID.

## 8. Development & Implementation Strategy

### 8.1. Recommended Implementation Order

1.  **`structures.py`**: Implement all core data classes.
2.  **`visualizer.py`**: Implement a basic visualizer to plot points from a `Page` object.
3.  **Simple Generator**: In `layout_generator.py`, create a function that generates only a single, un-rotated main textbox and successfully saves all outputs.
4.  **Full Parameterization**: Integrate the `config.yaml` file to control all generation parameters via distributions.
5.  **Multi-Box & Collision**: Add logic for placing multiple textboxes with varying orientations, including the SAT-based collision detection.
6.  **`augmentations.py`**: Implement the augmentation functions one by one, verifying each visually.
7.  **`ambiguous_layouts.py`**: Implement the special case generators.

### 8.2. Configuration for Ablation Studies

The `config.yaml` file will be structured to allow for easy enabling/disabling of features.

**Example `config.yaml` structure:**
```yaml
# --- Feature Flags ---
features:
  allow_interlinear_gloss: true

augmentations:
  use_textbox_warp: true
  use_missing_chars: false # This augmentation is turned off

# --- Parameters for Enabled Features ---
textbox_warp:
  amplitude: Uniform(0.5, 2.0)
  frequency: Uniform(0.1, 0.5)

missing_chars:
  probability: 0.02

# ... all other distributions
```

### 8.3. Reproducibility

The main script `generate.py` must accept a command-line argument, e.g., `--seed 42`. This integer will be used to initialize the `numpy.random` generator at the start of the program, guaranteeing that the same command will always produce the identical output.