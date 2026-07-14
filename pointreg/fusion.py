from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np

from .coarse import fpfh_registration, gcransac_from_correspondences
from .geotransformer import (
    DEFAULT_3DMATCH_CHECKPOINT,
    GeoTransformerCorrespondences,
    geotransformer_3dmatch_correspondences,
)
from .global_search import GlobalRegistrationSearch, GlobalSearchResult
from .metrics import alignment_metrics, pose_errors
from .models import RegistrationConfig, RegistrationResult
from .preprocessing import bounding_box_diagonal, preprocess_points
from .runtime import preload_open3d


ProgressCallback = Callable[[str], None]


def build_fusion_seed_candidates(
    output: GeoTransformerCorrespondences,
    source: np.ndarray,
    target: np.ndarray,
    voxel_size: float,
) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, str]]:
    """Build LGR, multi-run GC-RANSAC and FPFH seeds for global fusion."""
    candidates: dict[str, np.ndarray] = {"lgr": output.lgr_transform}
    inlier_counts: dict[str, int] = {}
    errors: dict[str, str] = {}
    for threshold in (0.008, 0.010, 0.012):
        for seed in (0, 1, 42):
            label = f"geotransformer_gcransac_t{threshold:.3f}_s{seed}"
            permutation = np.random.default_rng(seed).permutation(len(output.source_points))
            try:
                transform, inliers = gcransac_from_correspondences(
                    output.source_points[permutation],
                    output.target_points[permutation],
                    output.scores[permutation],
                    correspondence_distance=threshold,
                    max_iters=10000,
                    seed=seed,
                )
                candidates[label] = transform
                inlier_counts[label] = int(inliers.sum())
            except Exception as exc:
                errors[label] = str(exc)
    try:
        candidates["fpfh"] = fpfh_registration(source, target, voxel_size, seed=42)
    except Exception as exc:
        errors["fpfh"] = str(exc)
    return candidates, inlier_counts, errors


def register_low_overlap_pair(
    source: np.ndarray,
    target: np.ndarray,
    config: RegistrationConfig,
    *,
    checkpoint: str | Path | None = None,
    ground_truth: np.ndarray | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[RegistrationResult, dict[str, object]]:
    """Register a difficult pair with GeoTransformer-seeded global fusion."""
    config.validate()
    progress = progress or (lambda _: None)
    result = RegistrationResult(source_points=len(source), target_points=len(target))
    total_started = perf_counter()
    details: dict[str, object] = {}
    try:
        result.timings_ms["runtime_warmup"] = preload_open3d()
        source = np.asarray(source, dtype=float)
        target = np.asarray(target, dtype=float)
        started = perf_counter()
        source_points = preprocess_points(source, config.voxel_size, config.remove_outliers)
        target_points = preprocess_points(target, config.voxel_size, config.remove_outliers)
        result.timings_ms["preprocess"] = (perf_counter() - started) * 1000

        progress("正在提取 GeoTransformer 稠密对应…")
        started = perf_counter()
        output = geotransformer_3dmatch_correspondences(
            source_points,
            target_points,
            checkpoint=checkpoint or DEFAULT_3DMATCH_CHECKPOINT,
            voxel_size=config.voxel_size,
        )
        result.timings_ms["geotransformer"] = (perf_counter() - started) * 1000

        progress("正在生成 LGR、GC-RANSAC 与 FPFH 候选…")
        started = perf_counter()
        candidates, seed_inliers, seed_errors = build_fusion_seed_candidates(
            output, source_points, target_points, config.voxel_size
        )
        result.timings_ms["seed_candidates"] = (perf_counter() - started) * 1000

        progress("正在执行 4000 旋转 FFT 全局搜索与自由空间选优…")
        started = perf_counter()
        search = GlobalRegistrationSearch(source_points, target_points, config)
        search_result: GlobalSearchResult = search.run(candidates)
        result.timings_ms["global_fusion"] = (perf_counter() - started) * 1000

        transform = search_result.transform
        result.transformation = transform
        result.status = "converged"
        result.message = search_result.message
        result.metrics.update(
            alignment_metrics(source_points, target_points, transform, config.max_correspondence_distance)
        )
        diagonal = bounding_box_diagonal(source, target)
        result.metrics["bbox_diagonal"] = diagonal
        if ground_truth is not None:
            rotation_error, translation_error = pose_errors(transform, ground_truth)
            ratio = translation_error / diagonal if diagonal else float("inf")
            result.metrics.update(
                rotation_error_deg=rotation_error,
                translation_error=translation_error,
                translation_error_ratio=ratio,
            )
            result.success = (
                rotation_error < config.success_rotation_deg
                and ratio < config.success_translation_ratio
            )
        else:
            result.success = result.metrics.get("fitness", 0.0) > 0
        details = {
            "correspondence_count": len(output.source_points),
            "seed_inliers": seed_inliers,
            "seed_errors": seed_errors,
            "search": search_result.to_dict(),
        }
        progress("低覆盖 GeoTransformer 全局配准完成。")
    except Exception as exc:
        result.status = "failed"
        result.message = str(exc)
        result.success = False
        details = {"error": str(exc)}
    result.timings_ms["total"] = (perf_counter() - total_started) * 1000
    return result, details
