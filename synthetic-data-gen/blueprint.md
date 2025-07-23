You are an expert in Domain Randomization and synthetic data generation.
This document outlines the specifications for a Python-based ablation friendly, modular synthetic data generator. The goal is to create diverse, script-agnostic page layouts of handwritten manuscripts for training and evaluating machine learning models for document layout analysis.

Each character on a page is abstracted as a `Point` with three features: `(x, y, font_size)`. The generator will produce these point clouds along with corresponding ground-truth labels for textboxes and text lines, enabling supervised learning.
The code should have classes for points (characters), words (a collection of points), text lines (a collection of words), and text box (a collection of text lines)

Please study this blueprint, and think about an fast and efficient, step-by-step process for generating sample pages (perhaps many pages at once in parallel), without compromising on any functionality. Please decide the directory structure, and code ENTIRE files without skipping any. Please DO NOT make the codebase a CLI, just python code.

Class Definitions
-   **`Point(x, y, font_size)`**: The fundamental unit. Stores unnormalized local coordinates and font size.
-   **`Word(points: list[Point])`**: A collection of `Point` objects.
-   **`TextLine(words: list[Word])`**: A collection of `Word` objects.
-   **`TextBox(text_lines: list[TextLine], box_type: str, position: tuple, orientation_deg: float, width: float, height: float

TextBox can be of various types: main_text, marginalia, page number, interlinear_gloss (There is a probability of generating a small `interlinear_gloss` textbox positioned relative to a line in the `main_text`.)

## The system is designed with the following core design  principles:

-   **Domain Randomization:** Every variable parameter of the generation process (e.g., page size, number of textboxes, spacing, orientation) will be sampled from a pre-defined probability distribution to ensure high variance in the output data.
-   **Modularity:** The code will be organized into logical, decoupled components (TextBox types, augmentations, layout generation strategies). This allows for easy logging, debugging, maintenance and extension.
-   **Ablation-Friendly:** Augmentations and specific layout features will be implemented in a way that allows them to be easily enabled or disabled via a configuration file, facilitating ablation studies on model performance.
-   **Scalability:** The generation process for a single page will be self-contained, allowing for trivial parallelization across multiple CPU cores to generate large datasets.
-   **Reproducibility:** All stochastic processes will be controlled by a single random seed to ensure that any generated sample can be perfectly reproduced for debugging and validation.


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
    
    - **Font Size Variation:** A slight, random variation is applied to the font_size of each individual point within the textbox.
        
    - **Point-Level Jitter:** A slight random offset is added to the local (x, y) coordinates of each individual Point to simulate the unsteadiness of a human hand.

    - **Congestion Jitter:** This step simulates a congested or hurried writing style. Based on the character spacing and line spacing of a TextBox, a small random offset is added to the coordinates of every point on the TextBox. We want to cause a configurable percentage of points to become closer to points in adjacent lines (or even an adjacent textbox) than their neighbors in their own line, thus breaking the usual spacing rule of the character spacing being less than the line spacing. We do this do simulate challenging cases for layout analysis algorithms. Thus we want a specific configurable percentage of points to be randomly selected and jittered in this manner.

---

### **Phase 2: TextBox Geometric Distortion (Applied in Local Coordinates)**

After the points for a textbox are generated in their ideal grid-like layout, this phase applies large-scale geometric distortions to the entire point cloud of the textbox, simulating physical properties of the writing surface. These transformations can be applied in any random order.

- **Shear:** Skews the textbox along the X or Y axis, transforming the rectangular shape into a parallelogram.
    
- **Stretch:** Applies non-uniform scaling, stretching or squashing the textbox more along one axis than the other.
    
- **Warp / Curl:** Applies non-linear, wave-like distortions to the point coordinates, simulating a curved or wrinkled page surface. This is typically implemented using sine functions applied to the points' coordinates.
    

Each of these augmentations is controlled by a probability of being applied and a set of parameters sampled from the config (e.g., shear_factor, curl_amplitude, curl_frequency).

---

### **Phase 3: Page-Level Augmentations (Applied in Global Coordinates)**

This final phase is applied after all textboxes have been generated, distorted, and placed at their final position and orientation on the page. These augmentations affect the entire point cloud and can create interactions between different textboxes.

- **Textbox Placement & Orientation:** While not strictly an augmentation, the layout strategy's placement of textboxes (with randomized positions and orientations) is the primary page-level variation.
Text box orientations can be 0 +-45 degrees, 90 +-45 degrees, -90 +-45 degrees to the page x-axis. 0, 90, -90 should be much more common orientations thought.
    
- **Interlinear Gloss Placement:** A special case where a small interlinear_gloss textbox is probabilistically generated and its position is calculated relative to a specific text line within the main_text box. It can be placed above or below a text line.
    
- **Missing Characters (Point Dropout):** Iterates through all final points on the page and removes each one with a given probability. This simulates ink fade or physical damage.
    


## To generate random layouts we can use the following layout generation strategies: 
One of the below layout generation strategies (rejection_sampling, grid) can be configured in the config file. We should also be able to add other strategies in the future.
### 1) Rejection Sampling
First generate text boxes and perform augmentations on it, and then draw a polygon (Convex hull of all points) around it. When checking if two textboxes overlap, we should check if the polygons overlap (they can touch or be very close to each other, but no ovelap)
For example, for a TextBox we first do the Phase 1 and Phase 2 Augmentations. Then we do the Page level Phase 3 augmentation of Textbox Placement & Orientation. If the final location of the polygon of this TextBox Overlaps with existing Textboxe polygon, we can do the Phase 3 augmentation of Textbox Placement & Orientation again, and check again. If it still fails after N number of times, THEN perhaps we attempt again K times from scratch with a fresh TextBox.
We want some pages to have only one main textbox, but we also watch some pages to be really dense. Hence the number of (N and K) attempts to try to fit a TextBox can also vary from page to page.
Also have Page boundary checks to ensure textboxes don't extend outside page dimensions.

### 2) special ambiguous layouts: 
These generators are designed to create the following challenging cases for layout analysis algorithms by making local neighbor relationships ambiguous. In other words, based only on the coordinates of the points of these layouts, two reading orders can be interpreted.
These layouts should be independent of Rejection Sampling approach. We only want one grid per page in this case.

For these layouts, only apply the Point-Level Jitter, Font Size Variation, Point Dropout and Rotation augmentations. Do not apply any other augmentations.
#### Grid Layout
-   **Goal**: Create a perfect grid of points where horizontal and vertical spacing are equal, making the reading order (horizontal vs. vertical) impossible to infer from spacing alone.
-   **Generation**: Generate points at `(x_0 + i*S, y_0 + j*S)` for `i` in `range(N)` and `j` in `range(M)`, where `S` is the constant spacing.
-   **Labeling Interpretation**: For each generated layout, please save input-label pairs for both interpretations of reading order. Hence generate **two separate input-label data points with completely separate IDs** from a single geometric arrangement.



## INPUTS and LABELS

For each sample page, a directory `{sample_id}` is created with:
-   **`meta_data.txt`**: To store the Page Dimensions and other stats.
-   **`inputs_unnormalized.txt`**: `x y font_size` in raw generation units.
-   **`inputs_normalized.txt`**: `x_norm y_norm font_size_norm`.
    -   `x_norm`, `y_norm`: Coords scaled such that the longest page dimension maps to `[0, 1]`.
    -   `font_size_norm`: Font size divided by page height.
-   **`labels_textbox.txt`**: `textbox_id` (integer).
-   **`labels_textline.txt`**: `textline_id` (globally unique integer).
-   **`{sample_id}.png`**: Visualization of the page (Before we generate millions of data points, we should optionally be able visualize and save images of a small number of data points just to verify if the code is working.) 
We also want optional rendering flags (controlled via config):
 - color-coded by textboxes on a page (all points in a text box the same color)
 - color-coded by text lines on a page (all points in a text line the same color)
 In both cases, the size of the points as per the font size. Also please mark the borders of the page as black. And use bold colorblind friendly contrasting colors: '#ffe119', '#4363d8', '#f58231', '#dcbeff', '#800000', '#000075', '#a9a9a9', '#ffffff', '#000000'

Ambiguous Layout Labeling: For the Grid layouts, we can treat the entire grid as a single TextBox, and then each row/column can be labelled as TextLine. For each one of such generated layouts we want to save two data points, with everything but the labels different. It's okay if the input txt files are duplicate.

for example input_unnormalized.txt should look like this (with x, y, and font_size)

```

700 752 28

462 754 28

418 755 28

376 756 34

so on...

```




## Important Core Considerations:
- **Coordinate System Management**
To avoid ambiguity and bugs, all positional data will be handled within one of three explicitly defined coordinate systems. All augmentations and transformations are strictly bound to a specific system.
Local Coordinates (TextBox-centric): Each TextBox's point cloud is initially generated in its own coordinate system, with its origin (0, 0) typically at the box's center. All internal content generation and distortions (Phase 1 and Phase 2 augmentations) are performed exclusively in this system.
Global Coordinates (Page-centric): After a TextBox is fully populated and distorted in its local system, its points are transformed (rotated and translated) into the final page's coordinate system. This step places the box at its intended position and orientation on the page. All page-wide augmentations (Phase 3) that can affect interactions between textboxes happen in this system.
Normalized Coordinates (Output-centric): As the final step before saving, the global coordinates of all points on the page are scaled to fit a [0, 1] range. The longest page dimension (either width or height) is mapped to 1.0. This is the coordinate system used for the inputs_normalized.txt output file, providing a consistent scale for machine learning models.
This strict separation is vital for modularity, reproducibility, and correctness. Please write checks and "assert" statements to ensure robust coordinate system handling, logging and debugging. Note that font size remains constant across coordinate system transformations. It only gets normalized in the final step by dividing by page height.

- **The Hybrid OOP-to-Vectorized Workflow**
The generator will employ a "best of both worlds" approach, leveraging the strengths of both Object-Oriented Programming (OOP) and vectorized computation with NumPy.
Phase 1 - Content Generation (OOP): The initial creation of text content will use the clear, logical class structure (TextLine contains Words, Word contains Points). This approach is ideal for handling the complex rules of text layout, such as character/word/line spacing, alignment, and line breaks. The code in this phase is highly readable and easy to reason about.
The "Consolidation" Step: Once the ideal text content for a TextBox is generated as a hierarchy of objects, a crucial one-time "consolidation" method is called. This method traverses the object structure (TextLines -> Words -> Points) and flattens all character data into a single, highly efficient NumPy array of shape (N, 3) for (x, y, font_size). A corresponding NumPy array of shape (N,) for textline_id labels is also created.
Phase 2 & 3 - Geometric Augmentation & Finalization (Vectorized): From the moment of consolidation onwards, all subsequent operations are performed on these NumPy arrays.
Performance: Applying geometric distortions (Phase 2), and page-level augmentations such as Textbox Placement & Orientation (Phase 3) as vectorized NumPy operations is orders of magnitude faster than iterating through Python objects.
Clarity: The code for these mathematical transformations becomes a direct, clean, and often one-line implementation of the underlying formula (e.g., a single matrix multiplication for rotation).
This hybrid workflow gives us the organizational clarity of OOP for complex layout logic and the raw computational performance of vectorized NumPy for all heavy-duty geometric manipulations, creating a system that is both scalable and maintainable.


## MISC
- **Configuration File:** Use Pydantic for configuration loading and validation.
- **Distribution Sampler Utility:** Create a utility function sample_from_config(config_dict, random_state) that can parse distribution definitions from the YAML file.
- **Extensible Enums**: For parameters like box_type or text_alignment, use Python's StrEnum (available in enum since Python 3.11, or as a backport). This allows you to use strings in the config file (e.g., alignment: "justify") while getting the type safety and auto-completion benefits of enums in the code.
- **Dry-Run Mode** --dry-run that generates only 10 samples with random seed and saves the visualization. This is perfect for quickly testing config changes.
- **Extensibility via Factories/Registries:** For modularity, we can use a simple dictionary-based registry for layout strategies, augmentations, and TextBox Types. This avoids messy if/elif/else chains and allows new components to be added just by defining them in their respective modules. Create explicit registry objects.
LAYOUT_STRATEGIES = {}
AUGMENTATIONS = {}
TEXTBOX_TYPES = {}
A function/class can be added to a registry using a simple decorator. This makes the system "pluggable." To add a new layout strategy, a developer simply creates a new Python file, defines their function, and adds @register_layout('my_new_strategy') above it. The main generator code never needs to be touched.
- **Data Classes:** Using Python's dataclasses will make the Point, Word, TextLine, and TextBox classes cleaner and more robust.
- **Type Hinting:** Use Python's type hints throughout the codebase. It improves readability and allows for static analysis.
Core Implementation Strategy: From Logical Objects to Computational Arrays
To achieve both conceptual clarity and high performance, the generator will be built on two foundational strategies: strict coordinate system management and a hybrid object-oriented/vectorized workflow.
- **Create a Dataset Summary Report**
This report would contain aggregate statistics over the entire dataset:
- Histogram of textbox types generated.
- Distribution of page aspect ratios.
- Average/min/max points per page.
- Average number of textboxes per page.
This provides a high-level sanity check on the generated data.


TODO
- interlinear gloss is not being placed between the text lines placement. Interlinear gloss should be a part of the Text Box class. Not a text box type. Hence the probability of interlinear gloss should be be independent of probability of text box types occuring.
- bug in the grid special ambiguous layout: the grid is always at the same place




 please study this in depth the following and ask me if you have any doubts, clarifications:
- Please think about the implementation details and the perfect flow of generation.
- Please also suggest additional miscellaneous improvements which can be done.
Before writing the code, I want you to please write me a professional blueprint prompt which will document all the detailed specifications of this synthetic layout generator. Please format the prompt as a .md files so that I copy it easily.