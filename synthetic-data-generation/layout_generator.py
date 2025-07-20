# layout_generator.py (Corrected for KeyError: 'orientation_deg')

import numpy as np
from structures import Page, TextBox, TextLine, Word, Point
from augmentations import _sample_from_config

def _generate_text_content(box_width, config, box_cfg, rng):
    """Generates the hierarchical content (lines, words, points) for a textbox."""
    lines = []
    
    base_font_size = max(1, _sample_from_config(box_cfg['base_font_size'], rng))
    num_lines = max(0, int(_sample_from_config(box_cfg['lines_per_box'], rng)))

    line_spacing = base_font_size * _sample_from_config(box_cfg['line_spacing_multiplier'], rng)
    char_spacing = base_font_size * _sample_from_config(box_cfg['char_spacing_multiplier'], rng)
    word_spacing = char_spacing * box_cfg['word_spacing_multiplier']

    current_y = 0
    for _ in range(num_lines):
        words = []
        num_words = max(0, int(_sample_from_config(box_cfg['words_per_line'], rng)))
        line_break_width = box_width * _sample_from_config(box_cfg['line_break_fraction'], rng)
        
        current_x = 0
        for _ in range(num_words):
            if current_x > line_break_width:
                break

            points = []
            num_chars = max(0, int(_sample_from_config(box_cfg['chars_per_word'], rng)))
            
            for _ in range(num_chars):
                font_size = base_font_size
                jitter_x = font_size * _sample_from_config(box_cfg['positional_jitter'], rng)
                jitter_y = font_size * _sample_from_config(box_cfg['positional_jitter'], rng)
                
                point = Point(x=current_x + jitter_x, y=current_y + jitter_y, font_size=font_size)
                points.append(point)
                current_x += char_spacing
            
            if points:
                words.append(Word(points=points))
            current_x += word_spacing
        
        if words:
            lines.append(TextLine(words=words))
        current_y += line_spacing

    return lines, current_y

def _create_textbox(box_type, config, rng, position=(0,0), width=None, height=None, orientation=None):
    """Factory function to create a single, fully populated TextBox."""
    
    # --- FIXED LOGIC ---
    # 1. Start with the most basic global defaults.
    base_defaults = {**config['content'], **config['spacing']}
    
    # 2. Use the 'main_text' config as the primary set of defaults for any box.
    #    This guarantees that keys like 'orientation_deg' are always present.
    main_text_defaults = config['box_specific']['main_text']
    
    # 3. Get the specific overrides for the current box type (e.g., 'marginalia').
    box_specific_overrides = config['box_specific'].get(box_type, {})
    
    # 4. Merge them in the correct order of precedence:
    #    specific_overrides > main_text_defaults > base_defaults
    final_cfg = {**base_defaults, **main_text_defaults, **box_specific_overrides}
    # --- END OF FIX ---

    w = width if width is not None else rng.uniform(200, 800)
    
    text_lines, h = _generate_text_content(w, config, final_cfg, rng)
    h = height if height is not None else h
    
    for line in text_lines:
        for word in line.words:
            for p in word.points:
                p.x -= w / 2
                p.y -= h / 2

    # This line is now safe because 'orientation_deg' is guaranteed to be in final_cfg.
    orient = orientation if orientation is not None else _sample_from_config(final_cfg['orientation_deg'], rng)
    
    return TextBox(
        box_type=box_type,
        position=position,
        width=w, height=h,
        orientation_deg=orient,
        text_lines=text_lines
    )

def generate_standard_layout(config, rng):
    """Generates a standard page layout with a main text and ancillary boxes."""
    aspect_ratio = _sample_from_config(config['page']['aspect_ratio'], rng)
    page_h = config['page']['base_height']
    page_w = page_h * aspect_ratio
    page = Page(width=page_w, height=page_h)
    
    main_box = _create_textbox('main_text', config, rng)
    main_box.position = (page_w / 2, page_h / 2)
    page.add_textbox(main_box, collision_check=False)

    num_ancillary = _sample_from_config(config['textbox']['ancillary_box_count'], rng)
    all_types = list(config['textbox']['type_probabilities'].keys())
    
    # Filter for types that can be placed as ancillary boxes
    ancillary_types = [t for t in all_types if t not in ['main_text', 'interlinear_gloss']]
    
    # Create a new probability distribution just for the ancillary types
    prob_map = config['textbox']['type_probabilities']
    ancillary_probs_unnormalized = [prob_map[t] for t in ancillary_types]
    ancillary_probs = np.array(ancillary_probs_unnormalized) / sum(ancillary_probs_unnormalized)

    for _ in range(num_ancillary):
        if not ancillary_types: continue
        box_type = rng.choice(ancillary_types, p=ancillary_probs)
        ancillary_box = _create_textbox(box_type, config, rng)
        
        for _ in range(config['max_placement_attempts']):
            pos_x = rng.uniform(0, page.width)
            pos_y = rng.uniform(0, page.height)
            ancillary_box.position = (pos_x, pos_y)
            if page.add_textbox(ancillary_box):
                break
    
    if config['features']['allow_interlinear_gloss']:
        main_box = page.textboxes[0]
        newly_added_gloss_lines = []
        for i in range(len(main_box.text_lines) - 1):
            if rng.random() < config['textbox']['type_probabilities']['interlinear_gloss']:
                line1 = main_box.text_lines[i]
                line2 = main_box.text_lines[i+1]
                
                y1 = np.mean([p.y for w in line1.words for p in w.points]) if any(w.points for w in line1.words) else 0
                y2 = np.mean([p.y for w in line2.words for p in w.points]) if any(w.points for w in line2.words) else y1 + 50
                
                gloss_box = _create_textbox('interlinear_gloss', config, rng, width=main_box.width * 0.4)
                
                gloss_y_offset = (y1 + y2) / 2
                
                for gloss_line in gloss_box.text_lines:
                   for gloss_word in gloss_line.words:
                        for p in gloss_word.points:
                            p.y += gloss_y_offset
                
                newly_added_gloss_lines.extend(gloss_box.text_lines)
        
        main_box.text_lines.extend(newly_added_gloss_lines)

    return page