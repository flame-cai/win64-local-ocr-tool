# augmentations.py

import numpy as np
from structures import TextBox

def _sample_from_config(param, rng):
    """Helper to sample a value from a config dictionary."""
    if isinstance(param, dict):
        dist_type = param['type']
        if dist_type == 'Uniform':
            return rng.uniform(param['min'], param['max'])
        elif dist_type == 'Normal':
            return rng.normal(param['mu'], param['sigma'])
        elif dist_type == 'LogNormal':
            return rng.lognormal(param['mu'], param['sigma'])
        elif dist_type == 'Poisson':
            return rng.poisson(param['lambda'])
        elif dist_type == 'Constant':
            return param['value']
    return param # It's a constant value

def apply_augmentations(page, config, rng):
    """Applies all selected augmentations to the page."""
    
    # Textbox-level augmentations
    if config['augmentations']['use_textbox_warp']:
        for textbox in page.textboxes:
            warp_textbox(textbox, config['textbox_warp'], rng)

def warp_textbox(textbox: TextBox, warp_config: dict, rng: np.random.Generator):
    """
    Applies a non-linear sinusoidal warp to the points within a textbox.
    This modifies the local coordinates of the points in-place.
    """
    amp_x_mult = _sample_from_config(warp_config['amplitude_multiplier'], rng)
    freq_x_mult = _sample_from_config(warp_config['frequency_multiplier'], rng)
    amp_y_mult = _sample_from_config(warp_config['amplitude_multiplier'], rng)
    freq_y_mult = _sample_from_config(warp_config['frequency_multiplier'], rng)

    # Amplitudes are relative to box size
    amp_x = textbox.height * amp_x_mult
    freq_x = freq_x_mult / textbox.height if textbox.height > 0 else 0
    amp_y = textbox.width * amp_y_mult
    freq_y = freq_y_mult / textbox.width if textbox.width > 0 else 0

    phase_x = rng.uniform(0, 2 * np.pi)
    phase_y = rng.uniform(0, 2 * np.pi)

    for text_line in textbox.text_lines:
        for word in text_line.words:
            for point in word.points:
                # Distort x based on y, and y based on x
                dx = amp_x * np.sin(2 * np.pi * freq_x * point.y + phase_x)
                dy = amp_y * np.sin(2 * np.pi * freq_y * point.x + phase_y)
                point.x += dx
                point.y += dy