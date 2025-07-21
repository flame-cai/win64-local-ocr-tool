from typing import Generator
import numpy as np

from ..config import AppConfig
from ..core.classes import PageData
from ..core.registries import LAYOUT_STRATEGIES

from .. import layouts
from .. import augmentations

class PageGenerator:
    """
    Top-level class to generate a single page (or set of ambiguous pages)
    by invoking the appropriate layout strategy.
    """

    def __init__(self, config: AppConfig, seed: int):
        self.config = config
        self.rng = np.random.default_rng(seed)

    def generate_page(self) -> Generator[PageData, None, None]:
        """
        Selects and runs the configured layout strategy.

        Yields:
            One or more PageData objects for a single geometric arrangement.
        """
        strategy_name = self.config.layout.strategy
        if strategy_name not in LAYOUT_STRATEGIES:
            raise ValueError(f"Unknown layout strategy: {strategy_name}")

        layout_func = LAYOUT_STRATEGIES[strategy_name]
        
        # The layout function is a generator that yields PageData objects
        yield from layout_func(self.config, self.rng)