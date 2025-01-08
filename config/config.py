from typing import List, Optional, Dict, Tuple, Union
from pydantic import BaseModel, Field, model_validator, field_validator

# Pre-processing configurations
class MedianFilterConfig(BaseModel):
    kernel_size: int = Field(3, ge=1, description="Kernel size must be a positive integer")

class GaussianBlurConfig(BaseModel):
    sigma: float = Field(3.0, gt=0, description="Sigma must be a positive value")

class ResizerConfig(BaseModel):
    scaling_factor: float = Field(..., gt=0, description="Scaling factor must be a positive value")
    interpolation_order: Dict[str, int] = Field(
        {"upscaling": 3, "downscaling": 1},
        description="Interpolation order for upscaling and downscaling"
    )

class PreProcessPipeConfig(BaseModel):
    pre_processors: List[str] = Field(default_factory=list, description="List of pre-processors to apply")

# Utils configurations
class ImagePadderConfig(BaseModel):
    mode: str = Field("linear_ramp", description="Padding mode must be a valid string")

# EDP center configurations
class AutoCorrelationConfig(BaseModel):
    padding_constants: Dict[str, float] = Field(
        {"mask_padding_value": 1.0},
        description="Constants used for padding in autocorrelation"
    )

class DistanceMetricsConfig(BaseModel):
    metrics_map: Dict[str, str] = Field(
        {
            "mean_squared_error": "mean_squared_error",
            "manhattan": "manhattan",
            "cosine": "cosine",
            "jaccard_distance": "jaccard_distance",
        },
        description="Mapping of distance metric names to functions"
    )
    jaccard_distance: Dict[str, Union[int, str]] = Field(
        {"scaling_factor": 255, "dtype": "uint8"},
        description="Jaccard distance configuration"
    )

    @field_validator("jaccard_distance")
    def validate_jaccard_distance(cls, v):
        if not isinstance(v.get("scaling_factor"), int) or v["scaling_factor"] <= 0:
            raise ValueError("Jaccard scaling_factor must be a positive integer")
        if v.get("dtype") not in ["uint8", "int32", "float32"]:
            raise ValueError("Jaccard dtype must be a valid data type")
        return v

class OptFuncConfig(BaseModel):
    default_mask_value: float = Field(1.0, ge=0)
    distance_metric: str = Field("manhattan", description="Must be a valid distance metric")
    supported_metrics: List[str] = Field(default_factory=lambda: ["mean_squared_error",
                                                                  "manhattan",
                                                                  "cosine",
                                                                  "jaccard_distance"])
    lru_cache_size: int = Field(128, ge=1, description="LRU cache size must be positive")

    @field_validator("distance_metric")
    def validate_distance_metric(cls, v):
        if v not in cls.supported_metrics:
            raise ValueError(f"Invalid distance metric: {v}. Supported metrics are: {cls.supported_metrics}")
        return v


class OptimizationConfig(BaseModel):
    default_initial_guess_divisor: float = Field(2.0, gt=0)
    default_options: Dict[str, Union[float, bool]] = Field(
        {"xtol": 1e-8, "disp": False},
        description="Default options for optimization"
    )
    optimization_method: str = Field("trust-constr", description="Optimization method must be valid")

# Ellipse configurations
class EllipseParamsConfig(BaseModel):
    radial_range: Tuple[float, float] = Field((0.06, 0.6))
    semi_angle_range: int = Field(10, ge=0)
    num_points: int = Field(10, gt=0)
    cosine_bounds: Dict[str, List[Union[float, str]]] = Field(
        {"lower": [0, "-inf"], "upper": ["inf", "inf"]},
        description="Bounds for cosine curve fitting"
    )
    cosine_amplitude_offset: int = Field(1, ge=0)
    phase_modulus: int = Field(180, ge=0)
    env_mean_offset: int = Field(1, ge=0)

    @field_validator("radial_range")
    def validate_radial_range(cls, v):
        if len(v) != 2 or v[0] >= v[1]:
            raise ValueError("radial_range must be a tuple (min, max) where min < max")
        return v

# Mask configurations
class RecursiveMaskConfig(BaseModel):
    target_ratio: float = Field(0.25, ge=0, le=1)
    mean_reduction_factor: int = Field(100, ge=1)
    dilate_iterations: int = Field(2, ge=0)
    dilate_kernel_size: int = Field(2, ge=1)
    erode_iterations: int = Field(20, ge=0)
    erode_kernel_size: int = Field(2, ge=1)

class GaussianBlurTreshgMaskConfig(BaseModel):
    sigma: int = Field(20, ge=1)
    iterations: int = Field(10, ge=1)

class AdaptiveTreshMaskConfig(BaseModel):
    sector_size: int = Field(101, ge=1)
    c_param: int = Field(0)

class MeanTreshMaskConfig(BaseModel):
    constant: float = Field(0.2, ge=0)

# Polar configurations
class PolarRepresentationConfig(BaseModel):
    default_radial_range: Tuple[int, int] = Field((0, 1))
    default_angular_range: Tuple[int, int] = Field((0, 359))
    polar_mask_lock_factor: float = Field(0.9, gt=0, lt=1)
    polar_mask_extension_factor: float = Field(0.05, gt=0)

    @field_validator("default_radial_range")
    def validate_ranges(cls, v):
        if v[0] >= v[1]:
            raise ValueError("Ranges must be defined as (min, max) where min < max")
        return v

class CVPolarTransformationConfig(BaseModel):
    interpolation_method: str = Field("cv2.INTER_CUBIC")
    polar_image_size_factor: str = Field("sqrt(pi)")

class RotationalIntegrationConfig(BaseModel):
    default_method: str = Field("mean", description="Must be a valid method")
    supported_methods: List[str] = Field(default_factory=lambda: ["mean", "median"])

    @field_validator("default_method")
    def validate_method(cls, v):
        if v not in cls.supported_methods:
            raise ValueError(f"Invalid method '{v}'. Must be one of {cls.supported_methods}")
        return v

# Signal processing configurations
class TaperAndFilterConfig(BaseModel):
    tukey_window_alpha: int = Field(1, ge=0)
    savgol_filter_window_length: int = Field(5, gt=0)
    savgol_filter_polyorder: int = Field(3, ge=1)

# Root configurations
class ImageProcessConfig(BaseModel):
    pre_process: Dict[str, Union[MedianFilterConfig, GaussianBlurConfig, ResizerConfig, PreProcessPipeConfig]]
    utils: Dict[str, ImagePadderConfig]
    edp_center: Dict[str, Union[AutoCorrelationConfig, DistanceMetricsConfig, OptFuncConfig, OptimizationConfig]]
    ellipse: Dict[str, EllipseParamsConfig]
    mask: Dict[str, Union[RecursiveMaskConfig, GaussianBlurTreshgMaskConfig, AdaptiveTreshMaskConfig, MeanTreshMaskConfig]]
    polar: Dict[str, Union[PolarRepresentationConfig, CVPolarTransformationConfig, RotationalIntegrationConfig]]

class SignalProcessConfig(BaseModel):
    utils: Dict[str, TaperAndFilterConfig]

class AppConfig(BaseModel):
    image_process: ImageProcessConfig
    signal_process: SignalProcessConfig

    @model_validator(mode="before")
    def validate_nested_config(cls, values):
        if not values.get("image_process"):
            raise ValueError("image_process configuration is required")
        if not values.get("signal_process"):
            raise ValueError("signal_process configuration is required")
        return values