from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

CoarseMethod = Literal["none", "fpfh", "fpfh_multi_verified", "sc2_gnc"]
FineMethod = Literal["custom_icp", "point_to_plane"]


@dataclass(slots=True)
class RegistrationConfig:
    coarse_method: CoarseMethod = "fpfh"
    fine_method: FineMethod = "custom_icp"
    voxel_size: float = 0.0025
    max_correspondence_distance: float = 0.01
    trim_fraction: float = 0.8
    adaptive_trim: bool = False
    trim_fraction_min: float = 0.1
    trim_fraction_max: float = 0.95
    trim_fraction_step: float = 0.05
    trim_penalty_power: float = 2.0
    max_iterations: int = 60
    rmse_tolerance: float = 1e-7
    transform_tolerance: float = 1e-7
    min_correspondences: int = 20
    remove_outliers: bool = False
    random_seed: int = 42
    success_rotation_deg: float = 5.0
    success_translation_ratio: float = 0.02
    sc2_distance_threshold_factor: float = 2.0
    sc2_max_correspondences: int = 600
    sc2_keep_fraction: float = 0.2
    gnc_noise_bound_factor: float = 1.5
    gnc_max_iterations: int = 50
    coarse_hypotheses: int = 8
    feature_match_top_k: int = 3
    feature_ratio_threshold: float = 0.9
    validation_icp_iterations: int = 8

    def validate(self) -> None:
        if self.voxel_size < 0 or self.max_correspondence_distance <= 0:
            raise ValueError("voxel_size must be >= 0 and correspondence distance must be > 0")
        if not 0 < self.trim_fraction <= 1:
            raise ValueError("trim_fraction must be in (0, 1]")
        if not 0 < self.trim_fraction_min <= self.trim_fraction_max <= 1:
            raise ValueError("adaptive trim bounds must satisfy 0 < min <= max <= 1")
        if self.trim_fraction_step <= 0 or self.trim_penalty_power < 0:
            raise ValueError("trim step must be > 0 and penalty power must be >= 0")
        if self.max_iterations < 1 or self.min_correspondences < 3:
            raise ValueError("max_iterations must be >= 1 and min_correspondences >= 3")
        if self.sc2_distance_threshold_factor <= 0 or self.sc2_max_correspondences < 3:
            raise ValueError("SC2 threshold must be > 0 and max correspondences >= 3")
        if not 0 < self.sc2_keep_fraction <= 1 or self.gnc_noise_bound_factor <= 0 or self.gnc_max_iterations < 1:
            raise ValueError("invalid SC2/GNC configuration")
        if self.coarse_hypotheses < 1 or self.feature_match_top_k < 1:
            raise ValueError("coarse hypotheses and feature top-k must be >= 1")
        if not 0 < self.feature_ratio_threshold <= 1 or self.validation_icp_iterations < 0:
            raise ValueError("feature ratio must be in (0, 1] and validation iterations >= 0")


@dataclass(slots=True)
class CoarseCandidate:
    transformation: np.ndarray
    origin: str
    local_inliers: int = 0
    local_score: float = 0.0
    pre_fitness: float = 0.0
    pre_rmse: float = float("inf")
    verified_fitness: float = 0.0
    verified_rmse: float = float("inf")
    validation_rank: int = 0
    selected: bool = False
    validation_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["transformation"] = self.transformation.tolist()
        return data


@dataclass(slots=True)
class ICPRecord:
    iteration: int
    rmse: float
    correspondences: int
    rotation_delta_deg: float
    translation_delta: float
    elapsed_ms: float
    stage: str = "direct"
    trim_fraction: float = 1.0


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
    coarse_candidates: list[CoarseCandidate] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["transformation"] = self.transformation.tolist()
        data["coarse_candidates"] = [candidate.to_dict() for candidate in self.coarse_candidates]
        return data
