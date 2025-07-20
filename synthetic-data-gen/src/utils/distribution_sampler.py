import random
from typing import Dict, Any

from ...config.config_models import Distribution

def sample_from_config(config: Any, random_state: random.Random) -> Any:
    """
    Parses a distribution definition from the config and samples a value.
    If the config is not a distribution dict, it returns the value directly.
    """
    if isinstance(config, dict):
        config = Distribution(**config)
    
    if not isinstance(config, Distribution):
        return config

    dist_type = config.distribution.lower()

    if dist_type == "uniform":
        if config.min is None or config.max is None:
            raise ValueError("Uniform distribution requires 'min' and 'max'.")
        return random_state.uniform(config.min, config.max)
    
    elif dist_type == "normal":
        if config.mean is None or config.std is None:
            raise ValueError("Normal distribution requires 'mean' and 'std'.")
        return random_state.normalvariate(config.mean, config.std)
        
    elif dist_type == "choice":
        if config.choices is None:
            raise ValueError("Choice distribution requires 'choices'.")
        return random_state.choices(config.choices, weights=config.weights, k=1)[0]
        
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")