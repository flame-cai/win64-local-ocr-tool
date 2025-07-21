from typing import Dict, Any, Union
import numpy as np

def sample_from_config(dist_config: Dict[str, Any], rng: np.random.Generator) -> Union[int, float]:
    """
    Samples a value from a distribution defined in a configuration dictionary.

    Args:
        dist_config: A dictionary specifying the distribution and its parameters.
                     e.g., {'type': 'uniform', 'min': 0, 'max': 1}
                           {'type': 'normal', 'mean': 0, 'std': 1}
                           {'type': 'randint', 'low': 1, 'high': 10}
        rng: A NumPy random number generator instance.

    Returns:
        A single sampled value (int or float).
    """
    dist_type = dist_config.get("type")
    if dist_type == "uniform":
        return rng.uniform(low=dist_config["min"], high=dist_config["max"])
    elif dist_type == "normal":
        return rng.normal(loc=dist_config["mean"], scale=dist_config["std"])
    elif dist_type == "randint":
        low = dist_config["low"]
        high = dist_config["high"]
        return rng.integers(low=low, high=high + 1)
    elif dist_type is None: # It's a constant value
        return dist_config
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")