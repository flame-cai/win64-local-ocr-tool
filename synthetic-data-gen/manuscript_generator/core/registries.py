import functools
from typing import Callable, Dict, TypeVar

T = TypeVar('T', bound=Callable)

LAYOUT_STRATEGIES: Dict[str, Callable] = {}
AUGMENTATIONS: Dict[str, Callable] = {}

def register_layout(name: str) -> Callable[[T], T]:
    """Decorator to register a new layout strategy."""
    def decorator(func: T) -> T:
        if name in LAYOUT_STRATEGIES:
            raise ValueError(f"Layout strategy '{name}' already registered.")
        LAYOUT_STRATEGIES[name] = func
        return func
    return decorator

def register_augmentation(name: str) -> Callable[[T], T]:
    """Decorator to register a new augmentation function."""
    def decorator(func: T) -> T:
        if name in AUGMENTATIONS:
            raise ValueError(f"Augmentation '{name}' already registered.")
        AUGMENTATIONS[name] = func
        return func
    return decorator