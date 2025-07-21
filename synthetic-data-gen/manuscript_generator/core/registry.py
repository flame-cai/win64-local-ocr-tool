# /manuscript_generator/core/registry.py

from typing import Callable, Dict, Any

LAYOUT_STRATEGIES: Dict[str, Callable] = {}
AUGMENTATIONS: Dict[str, Callable] = {}

def register_layout(name: str):
    """Decorator to register a new layout strategy."""
    def decorator(func: Callable):
        if name in LAYOUT_STRATEGIES:
            raise ValueError(f"Layout strategy '{name}' is already registered.")
        LAYOUT_STRATEGIES[name] = func
        return func
    return decorator

def register_augmentation(name: str):
    """Decorator to register a new augmentation function."""
    def decorator(func: Callable):
        if name in AUGMENTATIONS:
            raise ValueError(f"Augmentation '{name}' is already registered.")
        AUGMENTATIONS[name] = func
        return func
    return decorator