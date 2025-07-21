import numpy as np
from typing import Generator
from ..core.classes import PageData
from ..core.registries import register_layout
from ..core.distributions import sample_from_config
from ..config import AppConfig

@register_layout("grid")
def generate_grid(config: AppConfig, rng: np.random.Generator) -> Generator[PageData, None, None]:
    """
    Generates a grid of points with two possible reading order interpretations.
    Yields two PageData objects with the same points but different labels.
    """
    grid_cfg = config.layout.grid
    rows = sample_from_config(grid_cfg.rows.model_dump(), rng)
    cols = sample_from_config(grid_cfg.cols.model_dump(), rng)
    spacing = sample_from_config(grid_cfg.spacing.model_dump(), rng)
    
    # Generate points
    x = np.arange(cols) * spacing
    y = np.arange(rows) * spacing
    xv, yv = np.meshgrid(x, y)
    
    # Apply micro-jitter
    jitter = rng.normal(0, spacing * 0.05, size=xv.shape + (2,))
    xv_jittered = xv + jitter[..., 0]
    yv_jittered = yv + jitter[..., 1]
    
    points = np.stack([
        xv_jittered.ravel(), 
        yv_jittered.ravel(), 
        np.full(rows * cols, 20.0) # Fixed font size for ambiguity
    ], axis=1)
    
    # Center points
    points[:, :2] -= points[:, :2].mean(axis=0)

    page_width = cols * spacing * 1.5
    page_height = rows * spacing * 1.5
    page_dims = (page_width, page_height)
    
    # --- Interpretation 1: Row-major (standard reading order) ---
    row_ids = np.arange(rows).reshape(-1, 1).repeat(cols, axis=1).ravel()
    
    # --- Interpretation 2: Column-major ---
    col_ids = np.arange(cols).reshape(1, -1).repeat(rows, axis=0).ravel()
    # Shift IDs to not overlap with row IDs
    col_ids += rows 

    # All points belong to a single textbox
    textbox_labels = np.ones(points.shape[0], dtype=int)
    
    meta = {"layout_strategy": "grid"}
    
    # Yield first interpretation
    yield PageData(
        sample_id="", points_global=points, labels_textbox=textbox_labels,
        labels_textline=row_ids, page_dims=page_dims, meta={**meta, "interpretation": "row-major"}
    )
    # Yield second interpretation
    yield PageData(
        sample_id="", points_global=points, labels_textbox=textbox_labels,
        labels_textline=col_ids, page_dims=page_dims, meta={**meta, "interpretation": "column-major"}
    )


@register_layout("concentric")
def generate_concentric(config: AppConfig, rng: np.random.Generator) -> Generator[PageData, None, None]:
    """
    Generates points in concentric circles with two interpretations (circular vs. radial).
    Yields two PageData objects.
    """
    conc_cfg = config.layout.concentric
    num_spokes = sample_from_config(conc_cfg.num_spokes.model_dump(), rng)
    num_circles = sample_from_config(conc_cfg.num_circles.model_dump(), rng)
    radius_step = sample_from_config(conc_cfg.radius_step.model_dump(), rng)
    start_radius = sample_from_config(conc_cfg.start_radius.model_dump(), rng)

    points_list = []
    circle_labels_list = []
    spoke_labels_list = []

    for i in range(num_circles):
        radius = start_radius + i * radius_step
        for k in range(num_spokes):
            angle = k * (2 * np.pi / num_spokes)
            
            # Add some jitter for realism
            r_jitter = rng.normal(0, radius_step * 0.05)
            a_jitter = rng.normal(0, (2 * np.pi / num_spokes) * 0.05)
            
            r = radius + r_jitter
            a = angle + a_jitter

            x = r * np.cos(a)
            y = r * np.sin(a)
            
            points_list.append([x, y, 20.0]) # Fixed font size
            circle_labels_list.append(i + 1) # Circle ID
            spoke_labels_list.append(k + 1 + num_circles) # Spoke ID (offset)

    points = np.array(points_list)
    page_dims = (points[:,0].max() * 2.2, points[:,1].max() * 2.2)
    textbox_labels = np.ones(len(points), dtype=int)
    
    meta = {"layout_strategy": "concentric"}
    
    # Yield first interpretation
    yield PageData(
        sample_id="", points_global=points, labels_textbox=textbox_labels,
        labels_textline=np.array(circle_labels_list), page_dims=page_dims, meta={**meta, "interpretation": "circular"}
    )
    # Yield second interpretation
    yield PageData(
        sample_id="", points_global=points, labels_textbox=textbox_labels,
        labels_textline=np.array(spoke_labels_list), page_dims=page_dims, meta={**meta, "interpretation": "radial"}
    )