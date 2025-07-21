"""
Layouts Package Initializer

This file ensures that all defined layout generation strategies
are registered in the central LAYOUT_STRATEGIES registry.

Importing the modules here triggers the execution of the @register_layout
decorators within them, making the layout system extensible without
modifying the core generation logic.
"""

from . import rejection_sampling
from . import ambiguous_layouts