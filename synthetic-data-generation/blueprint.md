You are an expert in Domain Randomization and synthetic data generation.
This document outlines the specifications for a Python-based ablation friendly, modular synthetic data generator. The goal is to create diverse, script-agnostic page layouts of handwritten manuscripts for training and evaluating machine learning models for document layout analysis.
Each character on a page is abstracted as a `Point` with three features: `(x, y, font_size)`. The generator will produce these point clouds along with corresponding ground-truth labels for textboxes and text lines, enabling supervised learning.
The code should have classes for points (characters), words (a collection of points), text lines (a collection of words), and text box (a collection of text lines)

Class Definitions
-   **`Point(x, y, font_size)`**: The fundamental unit. Stores unnormalized local coordinates and font size.
-   **`Word(points: list[Point])`**: A collection of `Point` objects.
-   **`TextLine(words: list[Word])`**: A collection of `Word` objects.
-   **`TextBox(text_lines: list[TextLine], box_type: str, position: tuple, orientation_deg: float, width: float, height: float

TextBox can be of various types: main_text, marginalia, page number, interlinear_gloss (There is a probability of generating a small `interlinear_gloss` textbox positioned relative to a line in the `main_text`.)

## The system is designed with the following core design  principles:

-   **Domain Randomization:** Every variable parameter of the generation process (e.g., page size, number of textboxes, spacing, orientation) will be sampled from a pre-defined probability distribution to ensure high variance in the output data.
-   **Modularity:** The code will be organized into logical, decoupled components (TextBox types, augmentations, visualizer). This allows for easy logging, debugging, maintenance and extension. We want to be able to add/remove/edit augmentations and text box types in the future.
-   **Ablation-Friendly:** Augmentations and specific layout features will be implemented in a way that allows them to be easily enabled or disabled via a configuration file, facilitating ablation studies on model performance.
-   **Scalability:** The generation process for a single page will be self-contained, allowing for trivial parallelization across multiple CPU cores to generate large datasets.
-   **Reproducibility:** All stochastic processes will be controlled by a single random seed to ensure that any generated sample can be perfectly reproduced for debugging and validation.


## To generate random layouts we can use: 
can be configured in the config.. We should also be add additional strategies in the future.
### 1) simple rejection sampling:
prioritizes dense layouts (text boxes close to each other). ** Using Adjacency-Biased Placement**
Instead of complex space-partitioning algorithms (like BSP trees or quadtrees to track free rectangles), we can use a simpler, highly effective heuristic that achieves the desired "dense packing" behavior:
1. **Place the main_text first:** This box is typically the largest and most central. It can be placed randomly but biased towards the page center.
2. **For subsequent textboxes:**
    - With a high probability p_adjacent (e.g., 0.8, defined in config), attempt to place the new box adjacent to an existing one.
        - a. Randomly select an already-placed textbox.
        - b. Randomly select one of its four sides (top, bottom, left, right).
        - c. Sample a position for the new box's center along that side, slightly offset by a random margin.
        - d. Perform a collision check against **all** other existing boxes. If it collides, retry a few times (e.g., pick a different side or existing box) before giving up on adjacency.
    - With probability 1 - p_adjacent, or if the adjacent placement fails, fall back to placing the box randomly on the page (standard rejection sampling).
The p_adjacent parameter gives us a tunable knob to control the "denseness" of the layout, which is excellent for Domain Randomization. 
This encourages a clustered, dense layout, mimicking how a scribe might fill a page.
Use Seperating Axis Theorem (SAT) algorithm for collision detection between two convex polygons. 
### 2) special ambiguous layouts: 
These generators are designed to create the following challenging cases for layout analysis algorithms by making local neighbor relationships ambiguous.
#### Grid Layout
-   **Goal**: Create a perfect grid of points where horizontal and vertical spacing are equal, making the reading order (horizontal vs. vertical) impossible to infer from spacing alone.
-   **Generation**: Generate points at `(x_0 + i*S, y_0 + j*S)` for `i` in `range(N)` and `j` in `range(M)`, where `S` is the constant spacing.
-   **Labeling Interpretation**: For each generated layout, please save input-label pairs for both interpretations of reading order.  Hence generate **two separate samples** from a single geometric arrangement.
- **Augmentations**: all augmentations for textboxes can OPTIONALLY be applied here too
#### Concentric Circles Layout
-   **Goal**: Create points arranged in concentric circles where the spacing between points along a circle's circumference is approximately equal to the radial spacing between circles. This creates ambiguity between a "circular" reading order and a "radial" (spoke-like) reading order.
-   **Generation**: To create true ambiguity between circular and radial readings, I suggest a "polar grid" approach:
	1. **Fix the number of spokes (radial lines), K.** This will be a randomized parameter.
	2. **Define the spoke angles:** theta_k = k * (2*pi / K) for k from 0 to K-1. These angles are now constant for the entire layout.
	3. **Iterate through radii:** For each radius r (from r_0 with step S), place one point at each of the K spoke angles.
-   **Labeling Interpretation**:  For each generated layout, please save input-label pairs for both interpretations of reading order.  Hence generate **two separate samples** from a single geometric arrangement.
- **Augmentations**: all augmentations for textboxes can OPTIONALLY be applied here too

---

## Following are are all the ways in which a page can "vary"
We need to model the variations in following augmentations, using appropriate **Probability Distributions.**
It should be like code written by an expert in Domain Randomization. The code should be modular, and we should have a config. file for this. 

The generator will model variations using a cascaded augmentation pipeline. Augmentations are organized into three distinct phases, applied sequentially. Every parameter mentioned is to be sampled from a probability distribution defined in the configuration file (e.g., config.), enabling fine-grained control and Domain Randomization.

to begin, we want to achieve high variance in page sizes and aspect ratios

---

### **Phase 1: TextBox Content & Micro-Variations (Applied in Local Coordinates)**

This phase focuses on generating the "ideal" content of a single textbox as a point cloud centered at (0,0). It defines the internal structure and introduces small-scale, handwriting-like imperfections.

- **Core Text Structure Parameters:**
    
    - **Font Sizes:** A base font_size is sampled for the textbox.
        
    - **Spacing Parameters:** The fundamental spacing is defined here, with the initial state typically being character_spacing < word_spacing < line_spacing.
        
        - Character Spacing: The distance between points within a word.
            
        - Word Spacing: The distance between words on a line.
            
        - Line Spacing (Leading): The vertical distance between text lines.
            
    - **Content Density:**
        
        - Words per TextLine: The number of words to generate for a typical line. Supports single-word "lines" for logographic scripts.
            
        - Lines per TextBox: The number of text lines to generate within the box.
            
    - **Text Alignment:** Determines how lines are positioned relative to each other within the textbox's width.
        
        - Options: left, right, center, justify. For justify, we can calculate the "slack" (the difference between the textbox width and the line's natural width). This slack can then be distributed by increasing the word_spacing for that specific line. 
            
- **Structural Variations:**
    
    - **Line Breaks:** A probabilistic model for introducing line breaks within a single logical text line. Controlled by two parameters: probability_of_break and a distribution for break_location.
        
- **Micro-level Jitter (Handwriting Imperfection):**
    
    - **TextLine-Level Variation:** A slight, random variation is applied to the font_size of each individual TextLine within the textbox.
        
    - **Point-Level Jitter:** A small random offset is added to the local (x, y) coordinates of each individual Point to simulate the unsteadiness of a human hand.
        

---

### **Phase 2: TextBox Geometric Distortion (Applied in Local Coordinates)**

After the points for a textbox are generated in their ideal grid-like layout, this phase applies large-scale geometric distortions to the entire point cloud of the textbox, simulating physical properties of the writing surface. These transformations are applied sequentially as a chain.

- **Shear:** Skews the textbox along the X or Y axis, transforming the rectangular shape into a parallelogram.
    
- **Stretch:** Applies non-uniform scaling, stretching or squashing the textbox more along one axis than the other.
    
- **Warp / Curl:** Applies non-linear, wave-like distortions to the point coordinates, simulating a curved or wrinkled page surface. This is typically implemented using sine functions applied to the points' coordinates.
    

Each of these augmentations is controlled by a probability of being applied and a set of parameters sampled from the config (e.g., shear_factor, curl_amplitude, curl_frequency).

---

### **Phase 3: Page-Level Augmentations (Applied in Global Coordinates)**

This final phase is applied after all textboxes have been generated, distorted, and placed at their final position and orientation on the page. These augmentations affect the entire point cloud and can create interactions between different textboxes.


- **Textbox Placement & Orientation:** While not strictly an augmentation, the layout strategy's placement of textboxes (with randomized positions and orientations) is the primary page-level variation. This includes placing textboxes very close to or touching each other.
    
- **Interlinear Gloss Placement:** A special case where a small interlinear_gloss textbox is probabilistically generated and its position is calculated relative to a specific text line within the main_text box.
    
- **Missing Characters (Point Dropout):** Iterates through all final points on the page and removes each one with a given probability. This simulates ink fade or physical damage.
    
- **Global Jitter (Congestion Simulation):** This critical final step simulates a congested or hurried writing style. A small random offset is added to the **global** coordinates of every point on the page. This is distinct from the local point-level jitter because it can cause a point from one text line to become closer to a point in an adjacent line (or even an adjacent textbox) than to its neighbors in its own line, breaking the initial spacing rules and creating challenging cases for layout analysis algorithms.

---
## INPUTS and LABELS

For each sample page, a directory `{sample_id}` is created with:
-   **`inputs_unnormalized.txt`**: `x y font_size` in raw generation units.
-   **`inputs_normalized.txt`**: `x_norm y_norm font_size_norm`.
    -   `x_norm`, `y_norm`: Coords scaled such that the longest page dimension maps to `[0, 1]`.
    -   `font_size_norm`: Font size divided by page height.
-   **`labels_textbox.txt`**: `textbox_id` (integer).
-   **`labels_textline.txt`**: `textline_id` (globally unique integer).
-   **`{sample_id}.png`**: Visualization of the page, color-coded by textbox ID. (Before we generate millions of data points, we should optionally be able visualize and save images of a small number of data points just to verify if the code is working.) We also want optional rendering flags (controlled via config or CLI) to draw:
	- The **oriented bounding box** of each TextBox. This is crucial for verifying placement, orientation, and collision detection logic.
	- The **text line boundaries** within a textbox, color-coded differently.

for example input_unnormalized.txt should look like this (with x, y, and font_size)

```

700 752 28

462 754 28

418 755 28

376 756 34

so on...

```
Ambiguous Layout Labeling: For the Grid and Concentric layouts, we can treat the entire grid/concentric circles as a single TextBox, and then each row/circle can be a TextLine.
For each one of such generated layouts we want to save two data points, with everything but the labels different. It's okay if the input txt files are duplicate.



## Important Core Considerations:
- **Coordinate System Management**
To avoid ambiguity and bugs, all positional data will be handled within one of three explicitly defined coordinate systems. All augmentations and transformations are strictly bound to a specific system.
Local Coordinates (TextBox-centric): Each TextBox's point cloud is initially generated in its own coordinate system, with its origin (0, 0) typically at the box's center. All internal content generation and distortions (Phase 1 and Phase 2 augmentations like local jitter, shear, and warp) are performed exclusively in this system.
Global Coordinates (Page-centric): After a TextBox is fully populated and distorted in its local system, its points are transformed (rotated and translated) into the final page's coordinate system. This step places the box at its intended position and orientation on the page. All page-wide augmentations (Phase 3, such as global jitter and point dropout) that can affect interactions between textboxes happen in this system.
Normalized Coordinates (Output-centric): As the final step before saving, the global coordinates of all points on the page are scaled to fit a [0, 1] range. The longest page dimension (either width or height) is mapped to 1.0. This is the coordinate system used for the inputs_normalized.txt output file, providing a consistent scale for machine learning models.
This strict separation is vital for modularity, reproducibility, and correctness.
- **The Hybrid OOP-to-Vectorized Workflow**
The generator will employ a "best of both worlds" approach, leveraging the strengths of both Object-Oriented Programming (OOP) and vectorized computation with NumPy.
Phase 1 - Content Generation (OOP): The initial creation of text content will use the clear, logical class structure (TextLine contains Words, Word contains Points). This approach is ideal for handling the complex rules of text layout, such as character/word/line spacing, alignment, and line breaks. The code in this phase is highly readable and easy to reason about.
The "Consolidation" Step: Once the ideal text content for a TextBox is generated as a hierarchy of objects, a crucial one-time "consolidation" method is called. This method traverses the object structure (TextLines -> Words -> Points) and flattens all character data into a single, highly efficient NumPy array of shape (N, 3) for (x, y, font_size). A corresponding NumPy array of shape (N,) for textline_id labels is also created.
Phase 2 & 3 - Geometric Augmentation & Finalization (Vectorized): From the moment of consolidation onwards, all subsequent operations are performed on these NumPy arrays.
Performance: Applying geometric distortions (Phase 2: Shear, Stretch, Warp), global transformations (rotation/translation), and page-level augmentations (Phase 3: Global Jitter, Dropout) as vectorized NumPy operations is orders of magnitude faster than iterating through Python objects.
Clarity: The code for these mathematical transformations becomes a direct, clean, and often one-line implementation of the underlying formula (e.g., a single matrix multiplication for rotation).
This hybrid workflow gives us the organizational clarity of OOP for complex layout logic and the raw computational performance of vectorized NumPy for all heavy-duty geometric manipulations, creating a system that is both scalable and maintainable.


##  On the Overall Generation Flow

Here is a high-level, step-by-step process for generating a single sample page, incorporating the hybrid workflow:
Initialization: (No change)
Page Generation (generate_page):
Step 1: Setup Page: (No change)
Step 2: Choose Layout Strategy: (No change)
Step 3: Plan Layout: (No change)
Step 4: Populate & Distort TextBoxes: Iterate through the TextBox blueprints. For each one:
a. Create a TextBox object.
b. (Phase 1) Call a textbox.populate_content() method. This method executes the TextBox Content Generation using the OOP structure (TextLine, Word, Point) to create the ideal text layout in local coordinates.
c. (Consolidation & Phase 2) Call a textbox.consolidate_and_distort() method. This method first converts the object hierarchy into powerful NumPy arrays for points and labels. Then, it applies the TextBox Geometric Augmentations (Shear, Stretch, Warp) as vectorized operations on these arrays.
Step 5: Assemble Page: Create a Page object. Iterate through the populated and distorted TextBox objects:
a. Take the textbox's NumPy point array and apply its global position and orientation via a final vectorized transformation (rotation and translation). This converts the points from local to global coordinates.
b. Collect all transformed NumPy arrays of points and labels into page-wide master arrays.
Step 6: Apply Final Page Augmentations (Phase 3): Apply Missing Characters and Global Jitter as vectorized operations on the final, aggregated NumPy point array.
Step 7: Handle Special Layouts: (No change)
Step 8: Save Outputs: (No change - normalization is now understood to be the conversion from Global to Normalized coordinates).

## MISC
- **Configuration File:** Use Pydantic for configuration loading and validation.
- **CLI** Use a library like Click or Typer to build a simple but effective CLI.
- **Distribution Sampler Utility:** Create a utility function sample_from_config(config_dict, random_state) that can parse distribution definitions from the YAML file.
- **Extensible Enums**: For parameters like box_type or text_alignment, use Python's StrEnum (available in enum since Python 3.11, or as a backport). This allows you to use strings in the config file (e.g., alignment: "justify") while getting the type safety and auto-completion benefits of enums in the code.
- **Validator Script**: A separate utility script validate_sample.py {sample_id} that loads a generated sample's files, checks for format errors (e.g., mismatched line counts), and re-renders the visualization. This is a crucial sanity check.
- **Dry-Run Mode: A CLI flag** --dry-run that generates a single sample, prints its metadata to the console, and saves the visualization, but doesn't write the full dataset. This is perfect for quickly testing config changes.
- **Extensibility via Factories/Registries:** For modularity, we can use a simple dictionary-based registry for layout strategies and augmentations. This avoids messy if/elif/else chains and allows new components to be added just by defining them in their respective modules. Create explicit registry objects.
LAYOUT_STRATEGIES = {}
AUGMENTATIONS = {}
A function/class can be added to a registry using a simple decorator. This makes the system "pluggable." To add a new layout strategy, a developer simply creates a new Python file, defines their function, and adds @register_layout('my_new_strategy') above it. The main generator code never needs to be touched.
- **Data Classes:** Using Python's dataclasses will make the Point, Word, TextLine, and TextBox classes cleaner and more robust.
- **Type Hinting:** Use Python's type hints throughout the codebase. It improves readability and allows for static analysis.
Core Implementation Strategy: From Logical Objects to Computational Arrays
To achieve both conceptual clarity and high performance, the generator will be built on two foundational strategies: strict coordinate system management and a hybrid object-oriented/vectorized workflow.
- **A Dataset Summary Report**
Problem: After generating 100,000 samples, it's hard to know if the output distribution matches our intent.
Suggestion: After a generation run, create a summary_report.md or summary.json in the root output directory. This report would contain aggregate statistics over the entire dataset:
Histogram of textbox types generated.
Distribution of page aspect ratios.
Average/min/max points per page.
Average number of textboxes per page.
This provides a high-level sanity check on the generated data.


Before you code, please study this in depth the following and ask me if you have any doubts, clarifications:
- Please think about the implementation details and the perfect flow of generation.
- Please also suggest additional miscellaneous improvements which can be done.




Before writing the code, I want you to please write me a professional blueprint prompt which will document all the detailed specifications of this synthetic layout generator. Please format the prompt as a .md files so that I copy it easily.