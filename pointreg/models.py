from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

CoarseMethod = Literal["none", "pca", "fpfh", "fpfh_multiscale", "gcransac", "geotransformer"]
FineMethod = Literal["custom_icp", "point_to_plane"]


@dataclass(slots=True)
class RegistrationConfig:
    coarse_method: CoarseMethod = "fpfh"
    fine_method: FineMethod = "custom_icp"
    voxel_size: float = 0.0025
    max_correspondence_distance: float = 0.01
    trim_fraction: float = 0.8
    max_iterations: int = 60
    rmse_tolerance: float = 1e-7
    transform_tolerance: float = 1e-7
    min_correspondences: int = 20
    remove_outliers: bool = False
    random_seed: int = 42
    geotransformer_checkpoint: str | None = None
    geotransformer_num_points: int = 717
    success_rotation_deg: float = 5.0
    success_translation_ratio: float = 0.02
    adaptive_trim: bool = False
    min_trim_fraction: float = 0.35
    multi_score_points: int = 1200
    multi_grid_rotations: int = 4000
    multi_seed_count: int = 20
    multi_climb_rounds: int = 25
    multi_refine_seeds: int = 6
    multi_refine_samples: int = 20
    multi_violation_gate: float = 0.016
    multi_min_fitness: float = 0.08
    multi_lock_span_deg: float = 4.0
    multi_lock_step_deg: float = 1.3
    multi_lock_top: int = 6

    def validate(self) -> None:
        if self.voxel_size < 0 or self.max_correspondence_distance <= 0:
            raise ValueError("voxel_size must be >= 0 and correspondence distance must be > 0")
        if not 0 < self.trim_fraction <= 1:
            raise ValueError("trim_fraction must be in (0, 1]")
        if self.max_iterations < 1 or self.min_correspondences < 3:
            raise ValueError("max_iterations must be >= 1 and min_correspondences >= 3")
        if self.geotransformer_num_points < 3:
            raise ValueError("geotransformer_num_points must be >= 3")
        if self.adaptive_trim and not 0 < self.min_trim_fraction <= self.trim_fraction:
            raise ValueError("min_trim_fraction must be in (0, trim_fraction]")
        if self.multi_grid_rotations < 100 or self.multi_refine_seeds < 1 or self.multi_score_points < 100:
            raise ValueError("invalid global-search settings")


@dataclass(slots=True)
class ICPRecord:
    iteration: int
    rmse: float
    correspondences: int
    rotation_delta_deg: float
    translation_delta: float
    elapsed_ms: float
    stage: str = "direct"


@dataclass(slots=True)
class RegistrationResult:
    transformation: np.ndarray = field(default_factory=lambda: np.eye(4))
    success: bool = False
    status: str = "not_started"
    message: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    history: list[ICPRecord] = field(default_factory=list)
    source_points: int = 0
    target_points: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["transformation"] = self.transformation.tolist()
        return data
