"""
Generation Package Initializer

Exposes the main PageGenerator class for use by the application's entry point.
"""

from .page_generator import PageGenerator
from .textbox_generator import TextBoxGenerator

__all__ = ["PageGenerator", "TextBoxGenerator"]