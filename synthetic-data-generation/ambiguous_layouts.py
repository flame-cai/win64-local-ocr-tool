# ambiguous_layouts.py

import numpy as np
from structures import Page, TextBox, TextLine, Word, Point
from augmentations import _sample_from_config

def generate_grid_layout(config, rng):
    """Generates a grid of points with ambiguous reading order."""
    page_h = 1000
    page_w = 1000
    page = Page(width=page_w, height=page_h)
    
    cfg = config['grid_layout']
    rows = max(1, _sample_from_config(cfg['rows'], rng))
    cols = max(1, _sample_from_config(cfg['cols'], rng))
    spacing = _sample_from_config(cfg['spacing'], rng)
    font_size = _sample_from_config(cfg['font_size'], rng)

    text_lines = []
    total_width = (cols - 1) * spacing
    total_height = (rows - 1) * spacing
    
    start_x = (page_w - total_width) / 2
    start_y = (page_h - total_height) / 2
    
    for j in range(rows):
        points = []
        for i in range(cols):
            x = start_x + i * spacing
            y = start_y + j * spacing
            points.append(Point(x, y, font_size))
        
        # Per blueprint, horizontal lines are the ground truth
        # We group points into a "word" and then a "line"
        text_lines.append(TextLine(words=[Word(points=points)]))
        
    # All points belong to a single textbox
    # The points are already in global page coordinates, so we treat the textbox
    # as a large, unrotated box centered on the page. We must shift the points
    # to be local to the textbox center.
    box_center = (page_w / 2, page_h / 2)
    for line in text_lines:
        for word in line.words:
            for p in word.points:
                p.x -= box_center[0]
                p.y -= box_center[1]

    textbox = TextBox(
        box_type='grid',
        position=box_center,
        width=total_width + spacing,
        height=total_height + spacing,
        text_lines=text_lines
    )
    page.add_textbox(textbox, collision_check=False)
    return page

def generate_concentric_layout(config, rng):
    """Generates concentric circles of points with ambiguous reading order."""
    page_h = 1000
    page_w = 1000
    page = Page(width=page_w, height=page_h)
    
    cfg = config['concentric_layout']
    num_circles = max(1, _sample_from_config(cfg['num_circles'], rng))
    radius = _sample_from_config(cfg['start_radius'], rng)
    spacing = _sample_from_config(cfg['radial_spacing'], rng)
    font_size = _sample_from_config(cfg['font_size'], rng)

    text_lines = []
    center_x, center_y = page_w / 2, page_h / 2
    max_radius = 0
    
    for _ in range(num_circles):
        points = []
        circumference = 2 * np.pi * radius
        num_points = max(1, int(circumference / spacing))
        
        for i in range(num_points):
            angle = (2 * np.pi * i) / num_points
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            points.append(Point(x, y, font_size))

        # Per blueprint, each circle is a text line
        text_lines.append(TextLine(words=[Word(points=points)]))
        max_radius = radius
        radius += spacing

    # Shift points to be local to the textbox center
    box_center = (page_w / 2, page_h / 2)
    for line in text_lines:
        for word in line.words:
            for p in word.points:
                p.x -= box_center[0]
                p.y -= box_center[1]
                
    textbox = TextBox(
        box_type='concentric',
        position=box_center,
        width=2*max_radius,
        height=2*max_radius,
        text_lines=text_lines
    )
    page.add_textbox(textbox, collision_check=False)
    return page