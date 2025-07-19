# Project Blueprint: Synthetic Manuscript Layout Generator

## 1. Overview

This document outlines the specifications for a Python-based synthetic data generator. The goal is to create diverse, script-agnostic page layouts of handwritten manuscripts for training and evaluating machine learning models for document layout analysis.

Each character on a page is abstracted as a `Point` with three features: `(x, y, font_size)`. The generator will produce these point clouds along with corresponding ground-truth labels for textboxes and text lines, enabling supervised learning.

The system is designed with core principles of Domain Randomization, modularity, and scalability to generate millions of unique samples.

## 2. Core Design Principles

-   **Domain Randomization:** Every variable parameter of the generation process (e.g., page size, number of textboxes, spacing, orientation) will be sampled from a pre-defined probability distribution to ensure high variance in the output data.
-   **Modularity:** The code will be organized into logical, decoupled components (structures, generators, augmentations, visualizer). This allows for easy maintenance and extension.
-   **Ablation-Friendly:** Augmentations and specific layout features will be implemented in a way that allows them to be easily enabled or disabled, facilitating ablation studies on model performance.
-   **Scalability:** The generation process for a single page will be self-contained, allowing for trivial parallelization across multiple CPU cores to generate large datasets.

## 3. System Architecture

### 3.1. File Structure

```
manuscript-generator/
├── generate.py                # Main script to run generation jobs
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

### 3.2. Class Definitions (`structures.py`)

-   **`Point(x, y, font_size)`**: The fundamental unit. Stores unnormalized coordinates and font size.
-   **`Word(points: list[Point])`**: A collection of `Point` objects.
-   **`TextLine(words: list[Word])`**: A collection of `Word` objects.
-   **`TextBox(text_lines: list[TextLine], box_type: str, position: tuple, orientation_deg: float, width: float, height: float)`**: The primary container.
    -   `box_type`: A string identifier, e.g., `'main_text'`, `'marginalia'`, `'page_no'`, `'footnote'`, `'interlinear_gloss'`.
    -   `position`: The `(x, y)` coordinates of the center of the un-rotated box.
    -   `orientation_deg`: The rotation of the box in degrees.
    -   `width`, `height`: The dimensions of the un-rotated bounding box.
    -   **Method `get_oriented_bounding_box()`**: Returns the 4 corner coordinates after rotation and translation. Used for collision detection.
    -   **Method `collides_with(other_box)`**: Implements the Separating Axis Theorem (SAT) to check for overlap between oriented bounding boxes.
-   **`Page(width: float, height: float, textboxes: list[TextBox])`**: The top-level object representing a single sample.
    -   **Method `add_textbox(textbox)`**: Attempts to add a textbox, first checking for collisions.
    -   **Method `apply_augmentations(config)`**: Applies a sequence of augmentation functions.
    -   **Method `finalize_and_save(output_dir, sample_id)`**: Performs normalization and writes all output files.

### 3.3. Generation Pipeline

The generation of a single page follows these steps:
1.  **Initialize Page**: Sample page aspect ratio from its distribution. Set a large base height (e.g., 1000 units) to work with unnormalized coordinates.
2.  **Generate Main TextBox**: Generate and place the primary text block. This box is typically large and has a near-zero orientation.
3.  **Generate Ancillary TextBoxes**: Iteratively generate and place additional textboxes (`marginalia`, `footnote`, etc.).
    -   For each new textbox, sample its type, dimensions, orientation, and internal properties (font size, spacing).
    -   Attempt to place it on the page by sampling a position.
    -   If it collides with any existing textbox, re-sample the position up to a maximum number of attempts. If it fails, discard the textbox.
4.  **Apply Augmentations**: The complete `Page` object is passed to a pipeline of augmentation functions (e.g., warp, point removal), which modify the textboxes or points in place.
5.  **Finalize and Save**: The `Page` object's save method is called to compute final labels, perform coordinate normalization, and write all output files.

## 4. Core Generation Logic & Parameters

Generation will be controlled by a configuration file (`default_params.yaml`) that specifies the distributions for all stochastic parameters.

| Parameter | Distribution & Example | Description |
| :--- | :--- | :--- |
| **Page Aspect Ratio** | `Uniform(0.7, 1.4)` | Defines page shape (portrait to landscape). |
| **Number of Ancillary Boxes** | `Poisson(lambda=2)` | Number of textboxes in addition to the main one. |
| **TextBox Type** | `Categorical/Choice` | `p={'marginalia':0.6, 'footnote':0.2, 'page_no':0.1, ...}` |
| **TextBox Orientation** | `Uniform(-5, 5)` for main, `Uniform(-180, 180)` for others. | Angle of rotation in degrees. |
| **Base Font Size** | `LogNormal(mu=2.5, sigma=0.3)` | Base font size for a textbox (e.g., 12pt). Unnormalized. |
| **Font Size Jitter** | `Normal(mu=0, sigma=0.05)` | Per-character font size variation, as a multiplier of base font size. |
| **Character Spacing** | `Normal(mu=0.8, sigma=0.1)` | Multiplier of font size. |
| **Word Spacing** | `Normal(mu=2.5, sigma=0.3)` | Multiplier of character spacing. |
| **Line Spacing** | `Normal(mu=1.5, sigma=0.2)` | Multiplier of font size. **Crucially, `mu` should be larger than character spacing's `mu`**. |
| **Positional Jitter (X, Y)**| `Normal(mu=0, sigma=0.05)` | Per-character positional noise, as a multiplier of font size. |
| **Line Break Position** | `Uniform(0.7, 1.0)` | A line ends after filling this fraction of the textbox width. |
| **Chars/Words/Lines** | `Poisson(lambda)` | Used to determine counts for words per line, lines per box, etc. |

## 5. Data Augmentations (Ablation-Friendly)

Augmentations will be implemented in `augmentations.py` as functions that take a `TextBox` or `Page` object and return a modified version. This allows them to be selectively applied in the `generate.py` script.

### 5.1. Textbox-Level Warp/Curl

-   **Function**: `warp_textbox(textbox, config)`
-   **Logic**: Applies a non-linear spatial transformation *only to the points within a single textbox*. This simulates local paper curl or distortion.
-   A simple implementation uses a sinusoidal function: `y' = y + amplitude * sin(frequency * x + phase)`. The amplitude, frequency, and phase parameters will themselves be sampled from distributions defined in the config.

### 5.2. Missing Characters

-   **Function**: `remove_points(page, probability)`
-   **Logic**: Iterates through all points on the finalized page. Each point has a `probability` of being deleted. This simulates ink fade, holes, or other forms of manuscript damage.

### 5.3. Interlinear Gloss

-   **Implementation**: This is a **layout feature**, not a post-processing augmentation. It must be integrated into the `layout_generator.py` logic.
-   **Logic**: After a `main_text` line is generated, there is a probability of generating a corresponding `interlinear_gloss` textbox.
    -   This gloss box will have a much smaller font size.
    -   Its position will be calculated relative to the main text line it annotates (e.g., placed directly above it with a small offset).
    -   It must be added to the page's collision detection system like any other textbox.

## 6. Special Ambiguous Layouts

To be implemented in `ambiguous_layouts.py`. These generators are designed to create challenging cases for layout analysis algorithms.

### 6.1. Grid Layout

-   **Goal**: Create a perfect grid of points where horizontal and vertical spacing are equal.
-   **Generation**: Generate points at `(x_0 + i*S, y_0 + j*S)` for `i` in `range(N)` and `j` in `range(M)`. `S` is the constant spacing.
-   **Labeling**: For labeling purposes, **one interpretation must be chosen**. The default will be to label **horizontal rows as text lines**. Each row `j` will receive a unique text-line ID. The entire grid will belong to a single textbox ID.

### 6.2. Concentric Circles Layout

-   **Goal**: Create points arranged in concentric circles where radial spacing is approximately equal to circumferential spacing.
-   **Generation**: Iterate through radii `r` from a starting radius `r_0` with a step `S`. For each `r`, calculate the number of points `N` that can fit on the circumference with an arc length of `S`. Place `N` points evenly around the circle of radius `r`.
-   **Labeling**: As with the grid, **one interpretation will be chosen for labeling**. The default will be to label **each circle as a text line**. All points on the circle with radius `r` will share the same text-line ID. The entire layout will belong to a single textbox ID.

## 7. Output Specification

For each generated sample, a directory will be created containing four text files and one image.

### 7.1. File Formats

-   **`input_unnormalized.txt`**:
    -   Format: `x y font_size` (space-separated).
    -   Values: Raw, unnormalized coordinates and font sizes used during generation. One point per line.

-   **`input_normalized.txt`**:
    -   Format: `x_norm y_norm font_size_norm` (space-separated).
    -   Normalization:
        -   The page is mapped to a `[0, 1] x [0, 1]` canvas, preserving aspect ratio. The longest side of the page is scaled to length 1.
        -   `x_norm = x / page.longest_side`
        -   `y_norm = y / page.longest_side`
        -   `font_size_norm = font_size / page.height`
    -   One point per line, in the same order as the unnormalized file.

-   **`labels_textbox.txt`**:
    -   Format: `textbox_id`
    -   Values: A single integer ID for each point, corresponding to its parent textbox. All points in the same textbox share the same ID.
    -   One integer label per line, in the same order as the input files.

-   **`labels_textline.txt`**:
    -   Format: `textline_id`
    -   Values: A single integer ID for each point, corresponding to its parent text line. All points in the same text line share the same ID. IDs are unique across the entire page.
    -   One integer label per line, in the same order as the input files.

### 7.2. Visualization Output

-   **`{sample_id}.png`**: A visual rendering of the generated page.
    -   Points should be plotted as dots.
    -   Dot size should be proportional to `point.font_size`.
    -   Dot color should correspond to a categorical label (e.g., color-coded by `textbox_id` by default).
    -   The visualizer should have an option to draw the oriented bounding boxes of textboxes for debugging purposes.