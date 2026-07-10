from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

CoarseMethod = Literal["none", "pca", "fpfh"]
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
    success_rotation_deg: float = 5.0
    success_translation_ratio: float = 0.02
    hypothesis_rotation_steps: int = 8
    enable_fgr: bool = True
    enable_reflection_candidates: bool = True
    coarse_prescreen_top_k: int = 6
    low_overlap_iterations: int = 120
    enable_teaser: bool = True
    low_overlap_sym_fitness_threshold: float = 0.55
    low_overlap_top_gap_threshold: float = 0.01
    bidirectional_check_top_k: int = 3
    bidirectional_short_iterations: int = 20
    enable_pose_hint: bool = True
    pose_hint_icp_distance: float = 0.003

    def validate(self) -> None:
        if self.voxel_size < 0 or self.max_correspondence_distance <= 0:
            raise ValueError("voxel_size must be >= 0 and correspondence distance must be > 0")
        if not 0 < self.trim_fraction <= 1:
            raise ValueError("trim_fraction must be in (0, 1]")
        if self.max_iterations < 1 or self.min_correspondences < 3:
            raise ValueError("max_iterations must be >= 1 and min_correspondences >= 3")
        if self.hypothesis_rotation_steps < 1:
            raise ValueError("hypothesis_rotation_steps must be >= 1")
        if self.coarse_prescreen_top_k < 1:
            raise ValueError("coarse_prescreen_top_k must be >= 1")
        if self.bidirectional_check_top_k < 0:
            raise ValueError("bidirectional_check_top_k must be >= 0")


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
