from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from .coarse import fpfh_registration, pca_registration
from .icp import custom_icp
from .io import read_points
from .metrics import alignment_metrics, pose_errors
from .models import RegistrationConfig, RegistrationResult
from .preprocessing import bounding_box_diagonal, preprocess_points, voxel_downsample
from .runtime import preload_open3d


def _point_to_plane(source: np.ndarray, target: np.ndarray, initial: np.ndarray, config: RegistrationConfig) -> tuple[np.ndarray, str, str]:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("point-to-plane ICP requires Open3D") from exc
    source_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source))
    target_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
    target_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max(config.voxel_size * 2, config.max_correspondence_distance), max_nn=30))
    result = o3d.pipelines.registration.registration_icp(
        source_cloud, target_cloud, config.max_correspondence_distance, initial,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=config.max_iterations),
    )
    status = "converged" if result.fitness > 0 else "failed"
    return np.asarray(result.transformation), status, f"Open3D fitness={result.fitness:.4f}"


def _sample_points(points: np.ndarray, limit: int) -> np.ndarray:
    if len(points) <= limit:
        return points
    return points[:: max(1, len(points) // limit)]


def _select_best_hypothesis(source: np.ndarray, target: np.ndarray, config: RegistrationConfig) -> tuple[np.ndarray, str]:
    """Run the global two-cloud pose search (see GlobalRegistrationSearch)."""
    from .global_search import GlobalRegistrationSearch
    return GlobalRegistrationSearch(source, target, config).run()


def _run_fine(source: np.ndarray, target: np.ndarray, initial: np.ndarray, config: RegistrationConfig) -> tuple[np.ndarray, list, str, str]:
    """Coarse-to-fine custom ICP: a wide-basin pass on coarser data, then full resolution."""
    transform = initial
    history: list = []
    if config.multiscale_fine:
        coarse_source = voxel_downsample(source, config.voxel_size * 2)
        coarse_target = voxel_downsample(target, config.voxel_size * 2)
        coarse_config = replace(config, max_correspondence_distance=config.max_correspondence_distance * 2,
                                max_iterations=max(10, config.max_iterations // 2))
        transform, coarse_history, status, _ = custom_icp(coarse_source, coarse_target, transform, coarse_config)
        if status == "failed":
            transform = initial
        else:
            history.extend(coarse_history)
    transform, fine_history, status, message = custom_icp(source, target, transform, config)
    history.extend(fine_history)
    return transform, history, status, message


def register_pair(source: str | Path | np.ndarray, target: str | Path | np.ndarray, config: RegistrationConfig | None = None, *, ground_truth: np.ndarray | None = None, initial: np.ndarray | None = None) -> RegistrationResult:
    config = config or RegistrationConfig()
    config.validate()
    result = RegistrationResult()
    result.timings_ms["runtime_warmup"] = preload_open3d()
    total_start = perf_counter()
    try:
        started = perf_counter()
        source_raw = read_points(source) if isinstance(source, (str, Path)) else np.asarray(source, dtype=float)
        target_raw = read_points(target) if isinstance(target, (str, Path)) else np.asarray(target, dtype=float)
        result.timings_ms["load"] = (perf_counter() - started) * 1000
        result.source_points, result.target_points = len(source_raw), len(target_raw)
        if len(source_raw) < 3 or len(target_raw) < 3:
            raise ValueError("both point clouds must contain at least three points")
        started = perf_counter()
        source_points = preprocess_points(source_raw, config.voxel_size, config.remove_outliers)
        target_points = preprocess_points(target_raw, config.voxel_size, config.remove_outliers)
        result.timings_ms["preprocess"] = (perf_counter() - started) * 1000

        transform = np.eye(4) if initial is None else np.asarray(initial, dtype=float).copy()
        hypothesis_name = ""
        started = perf_counter()
        if initial is None and config.coarse_method == "pca":
            transform = pca_registration(source_points, target_points)
        elif initial is None and config.coarse_method == "fpfh":
            transform = fpfh_registration(source_points, target_points, config.voxel_size, config.random_seed)
        elif initial is None and config.coarse_method == "multi":
            transform, hypothesis_name = _select_best_hypothesis(source_points, target_points, config)
        result.timings_ms["coarse"] = (perf_counter() - started) * 1000

        started = perf_counter()
        if initial is None and config.coarse_method == "multi":
            # 全局搜索内部已完成紧点到面终配准；宽阈值 trimmed ICP 会把低重叠
            # 正确姿态拖向高 fitness 的错误局部最优，这里不再追加精配准。
            status, message = "converged", "global search finished"
        elif config.fine_method == "custom_icp":
            transform, history, status, message = _run_fine(source_points, target_points, transform, config)
            result.history = history
        else:
            transform, status, message = _point_to_plane(source_points, target_points, transform, config)
        result.timings_ms["fine"] = (perf_counter() - started) * 1000
        if hypothesis_name:
            message = f"{message} | hypothesis={hypothesis_name}"
        result.transformation, result.status, result.message = transform, status, message
        result.metrics.update(alignment_metrics(source_points, target_points, transform, config.max_correspondence_distance))
        diagonal = bounding_box_diagonal(source_raw, target_raw)
        result.metrics["bbox_diagonal"] = diagonal
        if ground_truth is not None:
            rotation_error, translation_error = pose_errors(transform, ground_truth)
            result.metrics.update(rotation_error_deg=rotation_error, translation_error=translation_error,
                                  translation_error_ratio=translation_error / diagonal if diagonal else float("inf"))
            result.success = status != "failed" and rotation_error < config.success_rotation_deg and result.metrics["translation_error_ratio"] < config.success_translation_ratio
        else:
            result.success = status != "failed" and result.metrics["fitness"] > 0
    except Exception as exc:
        result.status, result.message, result.success = "failed", str(exc), False
    result.timings_ms["total"] = (perf_counter() - total_start) * 1000
    return result
