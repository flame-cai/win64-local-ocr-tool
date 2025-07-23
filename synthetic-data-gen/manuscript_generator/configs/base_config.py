# /manuscript_generator/configs/base_config.py
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Union, Literal

class Distribution(BaseModel):
    dist: str

class UniformInt(Distribution):
    dist: Literal["uniform_int"]
    min: int
    max: int

class UniformFloat(Distribution):
    dist: Literal["uniform_float"]
    min: float
    max: float

class Normal(Distribution):
    dist: Literal["normal"]
    mean: float
    std: float

class Constant(Distribution):
    dist: Literal["constant"]
    value: Any

class Choice(Distribution):
    dist: Literal["choice"]
    choices: List[Any]
    weights: List[float] = None

AnyDist = Union[UniformInt, UniformFloat, Normal, Constant, Choice]

class GenerationConfig(BaseModel):
    num_samples: int
    num_workers: int
    base_seed: int
    output_dir: str
    dry_run_num_samples: int

class VisualizationConfig(BaseModel):
    enabled: bool
    render_on_dry_run_only: bool
    coloring: Literal["textbox", "textline"]
    point_size_multiplier: float

class PageConfig(BaseModel):
    width: AnyDist
    height: AnyDist
    layout_strategy: AnyDist

class RejectionSamplingConfig(BaseModel):
    num_textboxes: AnyDist
    max_placement_attempts: AnyDist
    max_box_generation_attempts: AnyDist
    textbox_type_probabilities: Dict[str, float]

class GridLayoutConfig(BaseModel):
    rows: AnyDist
    cols: AnyDist
    spacing: AnyDist
    augmentations: List[str]

class TextBoxContentParams(BaseModel):
    font_size: AnyDist
    lines_per_box: AnyDist
    words_per_line: AnyDist
    alignment: AnyDist
    interlinear_gloss_probability: float = 0.0 # Add with a default value

class InterlinearGlossConfig(BaseModel):
    font_size_factor: AnyDist
    words_per_line: AnyDist
    vertical_offset_factor: AnyDist

class TextBoxContentConfig(BaseModel):
    main_text: TextBoxContentParams
    marginalia: TextBoxContentParams
    page_number: TextBoxContentParams
    interlinear_gloss: InterlinearGlossConfig
    character_spacing_factor: AnyDist
    word_spacing_factor: AnyDist
    line_spacing_factor: AnyDist
    line_break_probability: AnyDist
    font_size_variation: Dict[str, Any]
    point_level_jitter: Dict[str, Any]
    congestion_jitter: Dict[str, Any]
    chars_per_word: AnyDist
    interlinear_gloss: InterlinearGlossConfig

class DistortionAugmentationConfig(BaseModel):
    enabled: bool
    probability: float

class ShearConfig(DistortionAugmentationConfig):
    shear_factor_x: AnyDist
    shear_factor_y: AnyDist

class StretchConfig(DistortionAugmentationConfig):
    stretch_factor_x: AnyDist
    stretch_factor_y: AnyDist

class WarpCurlConfig(DistortionAugmentationConfig):
    amplitude_factor: AnyDist
    frequency_factor_x: AnyDist
    frequency_factor_y: AnyDist
    phase_x: AnyDist
    phase_y: AnyDist

class TextBoxDistortionConfig(BaseModel):
    shear: ShearConfig
    stretch: StretchConfig
    warp_curl: WarpCurlConfig

class PageAugmentationsConfig(BaseModel):
    orientation_deg: AnyDist
    orientation_other_range: AnyDist
    point_dropout: Dict[str, Any]

class Config(BaseModel):
    generation: GenerationConfig
    visualization: VisualizationConfig
    page: PageConfig
    rejection_sampling: RejectionSamplingConfig
    grid_layout: GridLayoutConfig
    textbox_content: TextBoxContentConfig
    textbox_distortion: TextBoxDistortionConfig
    page_augmentations: PageAugmentationsConfig