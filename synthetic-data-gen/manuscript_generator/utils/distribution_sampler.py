# /manuscript_generator/utils/distribution_sampler.py

import numpy as np
from typing import Any, Union
from pydantic import BaseModel
from manuscript_generator.configs.base_config import AnyDist

def sample_from_distribution(dist_config: Union[AnyDist, dict], rng: np.random.Generator) -> Any:
    """
    Samples a value from a distribution defined in the config.
    This function robustly handles both Pydantic models and raw dictionaries.

    Args:
        dist_config: A Pydantic model instance for a distribution or a dict.
        rng: A NumPy random number generator instance.

    Returns:
        A sampled value.
    """
    # --- Start of Fix ---
    # Normalize the input: If it's a Pydantic model, convert it to a dictionary.
    # This allows the rest of the function to use consistent dictionary access.
    if isinstance(dist_config, BaseModel):
        params = dist_config.model_dump()
    else:
        params = dist_config
    # --- End of Fix ---

    dist_type = params['dist']

    if dist_type == "uniform_int":
        return rng.integers(params['min'], params['max'], endpoint=True)
    elif dist_type == "uniform_float":
        return rng.uniform(params['min'], params['max'])
    elif dist_type == "normal":
        return rng.normal(params['mean'], params['std'])
    elif dist_type == "constant":
        return params['value']
    elif dist_type == "choice":
        # Use .get() for weights, as it's an optional parameter
        return rng.choice(params['choices'], p=params.get('weights'))
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")