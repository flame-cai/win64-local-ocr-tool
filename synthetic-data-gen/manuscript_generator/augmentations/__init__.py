"""
Augmentations Package Initializer

This file's primary role is to ensure that all defined augmentations
are registered in the central AUGMENTATIONS registry.

By importing the modules here, the @register_augmentation decorators
within them will be executed as soon as the 'augmentations' package
is imported anywhere in the code. This makes the system "pluggable".
"""

from . import phase1_content
from . import phase2_geometry
from . import phase3_page