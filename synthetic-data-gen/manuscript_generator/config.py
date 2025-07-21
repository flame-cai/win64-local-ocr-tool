from pydantic import BaseModel, Field
from typing import Dict, Literal, Union

# --- Distribution Models ---
class UniformDist(BaseModel):
    type: Literal['uniform']
    min: float
    max: float

class NormalDist(BaseModel):
    type: Literal['normal']
    mean: float
    std: float

class RandintDist(BaseModel):
    type: Literal['randint']
    low: int
    high: int

Distribution = Union[UniformDist, NormalDist, RandintDist]

# --- Augmentation Models ---
class CongestionJitterConfig(BaseModel):
    enable: bool
    probability: float = Field(..., ge=0, le=1)
    strength: Distribution

class AugmentationsPhase1(BaseModel):
    character_spacing: Distribution
    word_spacing: Distribution
    line_spacing: Distribution
    font_size_variation: Distribution
    point_jitter: Distribution
    line_break_prob: float = Field(..., ge=0, le=1)
    congestion_jitter: CongestionJitterConfig

class ProbabilisticAugmentation(BaseModel):
    prob: float = Field(..., ge=0, le=1)

class ShearConfig(ProbabilisticAugmentation):
    factor_x: Distribution
    factor_y: Distribution

class StretchConfig(ProbabilisticAugmentation):
    factor_x: Distribution
    factor_y: Distribution
    
class WarpConfig(ProbabilisticAugmentation):
    amplitude_x: Distribution
    frequency_x: Distribution
    amplitude_y: Distribution
    frequency_y: Distribution

class AugmentationsPhase2(BaseModel):
    shear: ShearConfig
    stretch: StretchConfig
    warp: WarpConfig

class PointDropoutConfig(BaseModel):
    prob: float = Field(..., ge=0, le=1)

class AugmentationsPhase3(BaseModel):
    point_dropout: PointDropoutConfig

# --- Layout and TextBox Models ---
class TextBoxConfig(BaseModel):
    font_size: Distribution
    lines_per_box: RandintDist
    words_per_line: RandintDist
    text_alignment: Literal['left', 'right', 'center', 'justify']

class RejectionSamplingConfig(BaseModel):
    num_textboxes: RandintDist
    max_placement_attempts: int
    max_generation_attempts: int
    allow_overlap: bool
    textbox_type_probs: Dict[str, float]

class GridConfig(BaseModel):
    rows: RandintDist
    cols: RandintDist
    spacing: Distribution

class ConcentricConfig(BaseModel):
    num_spokes: RandintDist
    num_circles: RandintDist
    radius_step: Distribution
    start_radius: Distribution

class LayoutConfig(BaseModel):
    strategy: Literal['rejection_sampling', 'grid', 'concentric']
    rejection_sampling: RejectionSamplingConfig
    grid: GridConfig
    concentric: ConcentricConfig

class PageConfig(BaseModel):
    width: Distribution
    height: Distribution
    interlinear_gloss_prob: float = Field(..., ge=0, le=1)

class GenerationConfig(BaseModel):
    seed: int
    num_samples: int
    num_workers: int
    output_dir: str
    dry_run: bool

class VisualizationConfig(BaseModel):
    visualize: bool
    dpi: int
    point_size_multiplier: float
    color_by: Literal['textbox', 'textline']
    background_color: str

# --- Main App Configuration Model ---
class AppConfig(BaseModel):
    generation: GenerationConfig
    page: PageConfig
    layout: LayoutConfig
    textboxes: Dict[str, TextBoxConfig]
    augmentations_phase1: AugmentationsPhase1
    augmentations_phase2: AugmentationsPhase2
    augmentations_phase3: AugmentationsPhase3
    output: VisualizationConfig