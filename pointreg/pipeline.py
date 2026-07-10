from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from .coarse import fpfh_registration, pca_registration
from .icp import custom_icp
from .io import read_points
from .metrics import alignment_metrics, pose_errors
from .models import ICPRecord, RegistrationConfig, RegistrationResult
from .preprocessing import bounding_box_diagonal, preprocess_points
from .runtime import preload_open3d


def _candidate_score(source: np.ndarray, target: np.ndarray, transform: np.ndarray, threshold: float) -> tuple[float, float, float]:
    metrics = alignment_metrics(source, target, transform, threshold)
    return metrics["fitness"], metrics["correspondences"], -metrics["rmse"]


def _relaxed_distance(config: RegistrationConfig) -> float:
    voxel_distance = config.voxel_size * 6 if config.voxel_size > 0 else 0.0
    return max(config.max_correspondence_distance * 3, voxel_distance, config.max_correspondence_distance)


def _select_fpfh_initial(source: np.ndarray, target: np.ndarray, config: RegistrationConfig) -> tuple[np.ndarray, str]:
    candidates: list[tuple[str, np.ndarray]] = []
    errors: list[str] = []
    for offset in range(3):
        name = "fpfh" if offset == 0 else f"fpfh_seed_{offset}"
        try:
            candidates.append((name, fpfh_registration(source, target, config.voxel_size, config.random_seed + offset)))
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    for name, builder in (("pca", lambda: pca_registration(source, target)), ("identity", lambda: np.eye(4))):
        try:
            candidates.append((name, builder()))
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    if config.voxel_size > 0:
        try:
            candidates.append((
                "fpfh_coarse_voxel",
                fpfh_registration(source, target, config.voxel_size * 2, config.random_seed + 1),
            ))
        except Exception as exc:
            errors.append(f"fpfh_coarse_voxel: {exc}")
    if not candidates:
        raise RuntimeError("; ".join(errors) or "no coarse registration candidates")
    threshold = _relaxed_distance(config)
    primary_feature_candidates = [
        item for item in candidates
        if item[0].startswith("fpfh_seed") or item[0] == "fpfh"
        if _candidate_score(source, target, item[1], threshold)[1] >= config.min_correspondences
    ]
    if primary_feature_candidates:
        best_name, best_transform = max(primary_feature_candidates, key=lambda item: _candidate_score(source, target, item[1], threshold))
        error_message = f", candidate_errors={'; '.join(errors)}" if errors else ""
        return best_transform, f"coarse candidate={best_name}, candidates={len(candidates)}{error_message}"
    feature_candidates = [
        item for item in candidates
        if item[0].startswith("fpfh") and _candidate_score(source, target, item[1], threshold)[1] >= config.min_correspondences
    ]
    scored_candidates = feature_candidates or candidates
    best_name, best_transform = max(scored_candidates, key=lambda item: _candidate_score(source, target, item[1], threshold))
    error_message = f", candidate_errors={'; '.join(errors)}" if errors else ""
    return best_transform, f"coarse candidate={best_name}, candidates={len(candidates)}{error_message}"


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


def _multiscale_custom_icp(source: np.ndarray, target: np.ndarray, initial: np.ndarray, config: RegistrationConfig) -> tuple[np.ndarray, list[ICPRecord], str, str]:
    transform = initial.copy()
    history: list[ICPRecord] = []
    final_status = "failed"
    final_message = ""
    distances = []
    executed_stages = 0
    for distance in (_relaxed_distance(config), max(config.max_correspondence_distance * 2, config.max_correspondence_distance), config.max_correspondence_distance):
        if not distances or distance < distances[-1]:
            distances.append(distance)
    for index, distance in enumerate(distances, start=1):
        executed_stages += 1
        stage_config = replace(
            config,
            max_correspondence_distance=distance,
            trim_fraction=1.0 if index == 1 else config.trim_fraction,
        )
        transform, stage_history, final_status, final_message = custom_icp(source, target, transform, stage_config)
        stage = f"multiscale:{distance:.4f}"
        for record in stage_history:
            record.iteration = len(history) + 1
            record.stage = stage
            history.append(record)
        if final_status == "failed":
            break
    message = f"multiscale ICP stages={executed_stages}; {final_message}"
    return transform, history, final_status, message


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
        started = perf_counter()
        if initial is None and config.coarse_method == "pca":
            transform = pca_registration(source_points, target_points)
            coarse_message = "coarse candidate=pca"
        elif initial is None and config.coarse_method == "fpfh":
            transform, coarse_message = _select_fpfh_initial(source_points, target_points, config)
        else:
            coarse_message = "coarse candidate=provided_initial" if initial is not None else "coarse candidate=identity"
        result.timings_ms["coarse"] = (perf_counter() - started) * 1000

        started = perf_counter()
        if config.fine_method == "custom_icp":
            if config.coarse_method == "fpfh" and initial is None:
                coarse_transform = transform.copy()
                transform, history, status, message = custom_icp(source_points, target_points, transform, config)
                metrics = alignment_metrics(source_points, target_points, transform, config.max_correspondence_distance)
                if status == "failed" or metrics["correspondences"] < config.min_correspondences:
                    transform, history, status, message = _multiscale_custom_icp(source_points, target_points, coarse_transform, config)
                message = f"{coarse_message}; {message}"
            else:
                transform, history, status, message = custom_icp(source_points, target_points, transform, config)
            result.history = history
        else:
            transform, status, message = _point_to_plane(source_points, target_points, transform, config)
        result.timings_ms["fine"] = (perf_counter() - started) * 1000
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
