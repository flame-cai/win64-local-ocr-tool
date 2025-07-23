# /manuscript_generator/core/common.py

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

class StrEnum(str, Enum):
    """Enum where members are also strings."""
    def __str__(self):
        return self.value

class TextBoxType(StrEnum):
    MAIN_TEXT = "main_text"
    MARGINALIA = "marginalia"
    PAGE_NUMBER = "page_number"
    GRID = "grid" # For ambiguous layouts

class TextAlignment(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"
    JUSTIFY = "justify"

@dataclass
class Point:
    """Fundamental unit: a character with local coords and font size."""
    x: float
    y: float
    font_size: float

@dataclass
class Word:
    """A collection of Point objects."""
    points: List[Point] = field(default_factory=list)

@dataclass
class TextLine:
    """A collection of Word objects."""
    words: List[Word] = field(default_factory=list)
    interlinear_gloss: Optional[List[Word]] = None