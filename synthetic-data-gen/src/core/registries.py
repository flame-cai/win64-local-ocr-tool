from typing import Callable, Dict, Any

# --- Registry Dictionaries ---
LAYOUT_STRATEGIES: Dict[str, Callable] = {}
AUGMENTATIONS: Dict[str, Callable] = {}


# --- Decorators for Registration ---

def register_layout(name: str) -> Callable:
    """A decorator to register a new layout strategy function."""
    def decorator(func: Callable) -> Callable:
        if name in LAYOUT_STRATEGIES:
            raise ValueError(f"Layout strategy '{name}' is already registered.")
        LAYOUT_STRATEGIES[name] = func
        return func
    return decorator


def register_augmentation(name: str) -> Callable:
    """A decorator to register a new augmentation function."""
    def decorator(func: Callable) -> Callable:
        if name in AUGMENTATIONS:
            raise ValueError(f"Augmentation '{name}' is already registered.")
        AUGMENTATIONS[name] = func
        return func
    return decorator