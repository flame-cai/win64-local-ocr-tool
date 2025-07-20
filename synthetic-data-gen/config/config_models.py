from pydantic import BaseModel, Field
from typing import Dict, Any, List, Union, Literal, Optional
from enum import Enum


class Distribution(BaseModel):
    distribution: str
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    choices: Optional[List[Any]] = None
    weights: Optional[List[float]] = None


class TextAlignment(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"
    JUSTIFY = "justify"


class TextBoxTypeConfig(BaseModel):
    width: Optional[Distribution] = None
    height: Optional[Distribution] = None
    orientation_deg: Union[Distribution, float]
    font_size: Optional[Distribution] = None
    char_spacing: Distribution
    word_spacing: Distribution
    line_spacing: Distribution
    words_per_line: Distribution
    lines_per_box: Distribution
    alignment: Distribution
    interlinear_gloss_probability: Optional[float] = Field(default=0.0)
    # For interlinear gloss specifically
    width_factor: Optional[Distribution] = None
    height_factor: Optional[Distribution] = None
    font_size_factor: Optional[Distribution] = None
    vertical_offset_factor: Optional[Distribution] = None
    horizontal_offset_factor: Optional[Distribution] = None


class RejectionSamplingConfig(BaseModel):
    max_placement_attempts: int
    generation_queue: List[Dict[str, Union[str, Distribution]]]


class GridLayoutConfig(BaseModel):
    rows: Distribution
    cols: Distribution
    spacing: Distribution


class ConcentricCirclesConfig(BaseModel):
    num_spokes: Distribution
    num_circles: Distribution
    radial_step: Distribution
    start_radius: Distribution


class LayoutStrategiesConfig(BaseModel):
    rejection_sampling: RejectionSamplingConfig
    grid: GridLayoutConfig
    concentric_circles: ConcentricCirclesConfig


class AugmentationParam(BaseModel):
    prob: float
    x_factor: Optional[Distribution] = None
    y_factor: Optional[Distribution] = None
    amplitude: Optional[Distribution] = None
    frequency: Optional[Distribution] = None
    axis: Optional[Distribution] = None


class AugmentationProfile(BaseModel):
    line_break_prob: float
    line_level_font_size_variation: Distribution
    point_level_jitter: Distribution
    shear: AugmentationParam
    stretch: AugmentationParam
    warp: AugmentationParam
    point_dropout: AugmentationParam
    global_jitter: Distribution


class VisualizationConfig(BaseModel):
    enabled: bool
    draw_obbs: bool
    draw_textlines: bool
    dpi: int
    point_size: int
    color_map: Dict[str, str]


class Config(BaseModel):
    seed: Optional[int] = None
    page_dimensions: Dict[str, Distribution]
    layout_strategy_selection: Distribution
    layout_strategies: LayoutStrategiesConfig
    textbox_types: Dict[str, TextBoxTypeConfig]
    augmentation_profiles: Dict[str, AugmentationProfile]
    visualization: VisualizationConfig