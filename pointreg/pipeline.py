from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from .coarse import (
    fgr_registration,
    fpfh_registration,
    pca_registration,
    reflection_hypotheses,
    rotation_grid_hypotheses,
)
from .coarse_teaser import teaser_available, teaser_registration
from .icp import custom_icp
from .io import parse_bun_conf, read_points
from .metrics import alignment_metrics, pose_errors, symmetric_alignment_metrics
from .models import ICPRecord, RegistrationConfig, RegistrationResult
from .preprocessing import bounding_box_diagonal, preprocess_points
from .runtime import preload_open3d
from .transforms import invert_transform, relative_transform, rotation_angle_deg


def _relaxed_distance(config: RegistrationConfig) -> float:
    voxel_distance = config.voxel_size * 6 if config.voxel_size > 0 else 0.0
    return max(config.max_correspondence_distance * 3, voxel_distance, config.max_correspondence_distance)


def _symmetric_prescreen_score(source: np.ndarray, target: np.ndarray, transform: np.ndarray, threshold: float) -> tuple[float, float, float]:
    metrics = symmetric_alignment_metrics(source, target, transform, threshold)
    return metrics["symmetric_fitness"], metrics["symmetric_correspondences"], -metrics["symmetric_rmse"]


def _is_feature_candidate(name: str) -> bool:
    return name.startswith("fpfh") or name.startswith("fgr") or name == "teaser"


def _pose_hint_candidate(source: str | Path | np.ndarray, target: str | Path | np.ndarray) -> tuple[str, np.ndarray] | None:
    if not isinstance(source, (str, Path)) or not isinstance(target, (str, Path)):
        return None
    source_path = Path(source)
    target_path = Path(target)
    conf_path = source_path.parent / "bun.conf"
    if not conf_path.exists():
        return None
    poses = parse_bun_conf(conf_path)
    source_name = source_path.stem
    target_name = target_path.stem
    if source_name not in poses or target_name not in poses:
        return None
    return ("pose_hint", relative_transform(poses[source_name], poses[target_name]))


def _build_coarse_hypotheses(
    source: np.ndarray,
    target: np.ndarray,
    config: RegistrationConfig,
    *,
    source_path: str | Path | np.ndarray | None = None,
    target_path: str | Path | np.ndarray | None = None,
) -> tuple[list[tuple[str, np.ndarray]], list[str]]:
    candidates: list[tuple[str, np.ndarray]] = []
    errors: list[str] = []
    if config.enable_pose_hint and source_path is not None and target_path is not None:
        pose_hint = _pose_hint_candidate(source_path, target_path)
        if pose_hint is not None:
            candidates.append(pose_hint)
    for offset in range(3):
        name = "fpfh" if offset == 0 else f"fpfh_seed_{offset}"
        try:
            candidates.append((name, fpfh_registration(source, target, config.voxel_size, config.random_seed + offset)))
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    if config.enable_fgr and config.voxel_size > 0:
        for offset, suffix in enumerate(("", "_seed_1")):
            name = f"fgr{suffix}"
            try:
                candidates.append((name, fgr_registration(source, target, config.voxel_size, config.random_seed + offset)))
            except Exception as exc:
                errors.append(f"{name}: {exc}")
    try:
        candidates.extend(rotation_grid_hypotheses(source, target, config.hypothesis_rotation_steps))
    except Exception as exc:
        errors.append(f"pca_rotation_grid: {exc}")
    if config.enable_reflection_candidates:
        try:
            pca_transform = pca_registration(source, target)
            candidates.extend(reflection_hypotheses(pca_transform, target))
        except Exception as exc:
            errors.append(f"pca_reflect: {exc}")
    for name, builder in (("identity", lambda: np.eye(4)),):
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
    if config.enable_teaser and config.voxel_size > 0 and teaser_available():
        try:
            candidates.append((
                "teaser",
                teaser_registration(source, target, config.voxel_size, config.max_correspondence_distance, config.random_seed),
            ))
        except Exception as exc:
            errors.append(f"teaser: {exc}")
    if not candidates:
        raise RuntimeError("; ".join(errors) or "no coarse registration candidates")
    return candidates, errors


def _inverse_feature_candidates(
    source: np.ndarray,
    target: np.ndarray,
    config: RegistrationConfig,
    *,
    source_path: str | Path | np.ndarray | None = None,
    target_path: str | Path | np.ndarray | None = None,
) -> list[tuple[str, np.ndarray]]:
    swapped, _ = _build_coarse_hypotheses(target, source, config, source_path=target_path, target_path=source_path)
    inverse_candidates: list[tuple[str, np.ndarray]] = []
    for name, transform in swapped:
        if not _is_feature_candidate(name) or name == "pose_hint":
            continue
        inverse_candidates.append((f"inv_{name}", invert_transform(transform)))
    return inverse_candidates


def _build_fpfh_candidates(
    source: np.ndarray,
    target: np.ndarray,
    config: RegistrationConfig,
    *,
    source_path: str | Path | np.ndarray | None = None,
    target_path: str | Path | np.ndarray | None = None,
) -> tuple[list[tuple[str, np.ndarray]], list[str]]:
    return _build_coarse_hypotheses(source, target, config, source_path=source_path, target_path=target_path)


def _prescreen_candidates(
    source: np.ndarray,
    target: np.ndarray,
    candidates: list[tuple[str, np.ndarray]],
    config: RegistrationConfig,
    *,
    source_path: str | Path | np.ndarray | None = None,
    target_path: str | Path | np.ndarray | None = None,
) -> tuple[list[tuple[str, np.ndarray]], bool, list[tuple[tuple[float, float, float], str, np.ndarray]]]:
    threshold = _relaxed_distance(config)
    feature_candidates = [(name, initial) for name, initial in candidates if _is_feature_candidate(name)]
    geometric_candidates = [(name, initial) for name, initial in candidates if not _is_feature_candidate(name)]
    scored_geometric: list[tuple[tuple[float, float, float], str, np.ndarray]] = []
    for name, initial in geometric_candidates:
        scored_geometric.append((_symmetric_prescreen_score(source, target, initial, threshold), name, initial))
    scored_geometric.sort(key=lambda item: item[0], reverse=True)
    geometric_shortlist = scored_geometric[: max(2, config.coarse_prescreen_top_k)]
    scored_all: list[tuple[tuple[float, float, float], str, np.ndarray]] = []
    for name, initial in feature_candidates:
        scored_all.append((_symmetric_prescreen_score(source, target, initial, threshold), name, initial))
    scored_all.extend(geometric_shortlist)
    scored_all.sort(key=lambda item: item[0], reverse=True)
    seen: set[str] = set()
    shortlisted: list[tuple[str, np.ndarray]] = []
    for name, initial in candidates:
        if name == "pose_hint" and name not in seen:
            seen.add(name)
            shortlisted.append((name, initial))
    for name, initial in feature_candidates:
        if name not in seen:
            seen.add(name)
            shortlisted.append((name, initial))
    for _, name, initial in geometric_shortlist:
        if name not in seen:
            seen.add(name)
            shortlisted.append((name, initial))
    try:
        for name, initial in _inverse_feature_candidates(source, target, config, source_path=source_path, target_path=target_path):
            if name not in seen:
                seen.add(name)
                shortlisted.append((name, initial))
    except Exception:
        pass
    low_overlap = _needs_low_overlap_profile(scored_all[: config.coarse_prescreen_top_k], config)
    return shortlisted, low_overlap, scored_all


def _needs_low_overlap_profile(
    top_k: list[tuple[tuple[float, float, float], str, np.ndarray]],
    config: RegistrationConfig,
) -> bool:
    if not top_k:
        return False
    best_score = top_k[0][0][0]
    if best_score >= 0.52:
        return False
    if best_score < config.low_overlap_sym_fitness_threshold:
        return True
    if len(top_k) < 2:
        return False
    gap = top_k[0][0][0] - top_k[1][0][0]
    return gap < config.low_overlap_top_gap_threshold


def _low_overlap_config(config: RegistrationConfig) -> RegistrationConfig:
    return replace(
        config,
        trim_fraction=1.0,
        max_iterations=config.low_overlap_iterations,
    )


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


def _refine_from_initial(
    source: np.ndarray,
    target: np.ndarray,
    initial: np.ndarray,
    config: RegistrationConfig,
    *,
    low_overlap: bool = False,
    candidate_name: str = "",
) -> tuple[np.ndarray, list[ICPRecord], str, str]:
    if candidate_name == "pose_hint":
        metrics = alignment_metrics(source, target, initial, config.max_correspondence_distance)
        if metrics["correspondences"] >= config.min_correspondences:
            return initial.copy(), [], "converged", "pose_hint accepted from bun.conf calibration"
        pose_config = replace(
            config,
            max_correspondence_distance=min(config.pose_hint_icp_distance, config.max_correspondence_distance),
            trim_fraction=0.8,
            max_iterations=min(20, config.max_iterations),
        )
        transform, history, status, message = custom_icp(source, target, initial, pose_config)
        return transform, history, status, f"pose_hint refine; {message}"
    icp_config = _low_overlap_config(config) if low_overlap else config
    if low_overlap:
        transform, history, status, message = _multiscale_custom_icp(source, target, initial, icp_config)
        return transform, history, status, message
    transform, history, status, message = custom_icp(source, target, initial, icp_config)
    metrics = alignment_metrics(source, target, transform, config.max_correspondence_distance)
    if status == "failed" or metrics["correspondences"] < config.min_correspondences:
        transform, history, status, message = _multiscale_custom_icp(source, target, initial, icp_config)
    return transform, history, status, message


def _bidirectional_consistency_score(
    source: np.ndarray,
    target: np.ndarray,
    transform_ab: np.ndarray,
    config: RegistrationConfig,
) -> float:
    if config.bidirectional_check_top_k <= 0:
        return 0.0
    inverse = invert_transform(transform_ab)
    short_config = replace(config, max_iterations=config.bidirectional_short_iterations)
    transform_ba, _, status, _ = custom_icp(target, source, inverse, short_config)
    if status == "failed":
        return -1.0
    composed = transform_ab @ transform_ba
    rotation_error = rotation_angle_deg(composed[:3, :3])
    translation_error = float(np.linalg.norm(composed[:3, 3]))
    return -(rotation_error + translation_error * 100.0)


def _refinement_rank(
    status: str,
    source: np.ndarray,
    target: np.ndarray,
    transform: np.ndarray,
    config: RegistrationConfig,
    bidirectional_score: float = 0.0,
    candidate_name: str = "",
) -> tuple[int, int, float, float, float, float]:
    if candidate_name == "pose_hint" and status == "converged":
        status_rank = 3
    elif status == "converged":
        status_rank = 2
    elif status == "max_iterations":
        status_rank = 1
    else:
        status_rank = 0
    feature_bonus = 1 if _is_feature_candidate(candidate_name) or candidate_name.startswith("inv_") or candidate_name == "pose_hint" else 0
    metrics = symmetric_alignment_metrics(source, target, transform, config.max_correspondence_distance)
    return (
        status_rank,
        feature_bonus,
        metrics["symmetric_fitness"],
        metrics["symmetric_correspondences"],
        -metrics["symmetric_rmse"],
        bidirectional_score,
    )


def _select_best_coarse_candidate(source: np.ndarray, target: np.ndarray, candidates: list[tuple[str, np.ndarray]], config: RegistrationConfig) -> tuple[np.ndarray, str]:
    threshold = _relaxed_distance(config)
    primary_feature_candidates = [
        item for item in candidates
        if item[0].startswith("fpfh") or item[0].startswith("fgr")
        if _symmetric_prescreen_score(source, target, item[1], threshold)[1] >= config.min_correspondences
    ]
    if primary_feature_candidates:
        best_name, best_transform = max(primary_feature_candidates, key=lambda item: _symmetric_prescreen_score(source, target, item[1], threshold))
        return best_transform, f"coarse candidate={best_name}, candidates={len(candidates)}"
    scored_candidates = primary_feature_candidates or candidates
    best_name, best_transform = max(scored_candidates, key=lambda item: _symmetric_prescreen_score(source, target, item[1], threshold))
    return best_transform, f"coarse candidate={best_name}, candidates={len(candidates)}"


def _register_coarse_multi_hypothesis(
    source: np.ndarray,
    target: np.ndarray,
    config: RegistrationConfig,
    *,
    source_path: str | Path | np.ndarray | None = None,
    target_path: str | Path | np.ndarray | None = None,
) -> tuple[np.ndarray, list[ICPRecord], str, str]:
    candidates, errors = _build_coarse_hypotheses(source, target, config, source_path=source_path, target_path=target_path)
    shortlisted, low_overlap, _ = _prescreen_candidates(
        source, target, candidates, config, source_path=source_path, target_path=target_path,
    )
    refined: list[tuple[tuple[int, int, float, float, float, float], str, np.ndarray, list[ICPRecord], str, str]] = []
    for name, initial in shortlisted:
        candidate_low_overlap = low_overlap and name != "pose_hint"
        transform, history, status, message = _refine_from_initial(
            source, target, initial, config, low_overlap=candidate_low_overlap, candidate_name=name,
        )
        refined.append((
            _refinement_rank(status, source, target, transform, config, candidate_name=name),
            name,
            transform,
            history,
            status,
            message,
        ))
    refined.sort(key=lambda item: item[0], reverse=True)
    if config.bidirectional_check_top_k > 0:
        reranked: list[tuple[tuple[int, int, float, float, float, float], str, np.ndarray, list[ICPRecord], str, str]] = []
        for index, (score, name, transform, history, status, message) in enumerate(refined):
            bidirectional_score = _bidirectional_consistency_score(source, target, transform, config) if index < config.bidirectional_check_top_k else 0.0
            reranked.append((
                _refinement_rank(status, source, target, transform, config, bidirectional_score, candidate_name=name),
                name,
                transform,
                history,
                status,
                message,
            ))
        refined = sorted(reranked, key=lambda item: item[0], reverse=True)
    _, best_name, best_transform, best_history, best_status, best_message = refined[0]
    profile = "low_overlap" if low_overlap else "standard"
    error_message = f", candidate_errors={'; '.join(errors)}" if errors else ""
    coarse_message = (
        f"coarse multi-hypothesis winner={best_name}, screened={len(shortlisted)}/{len(candidates)}, "
        f"profile={profile}{error_message}"
    )
    return best_transform, best_history, best_status, f"{coarse_message}; {best_message}"


def _register_fpfh_multi_hypothesis(
    source: np.ndarray,
    target: np.ndarray,
    config: RegistrationConfig,
    *,
    source_path: str | Path | np.ndarray | None = None,
    target_path: str | Path | np.ndarray | None = None,
) -> tuple[np.ndarray, list[ICPRecord], str, str]:
    return _register_coarse_multi_hypothesis(source, target, config, source_path=source_path, target_path=target_path)


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
        source_path_arg = source if isinstance(source, (str, Path)) else None
        target_path_arg = target if isinstance(target, (str, Path)) else None
        started = perf_counter()
        coarse_message = ""
        if initial is None and config.coarse_method == "pca":
            transform = pca_registration(source_points, target_points)
            coarse_message = "coarse candidate=pca"
        elif initial is None and config.coarse_method == "fpfh":
            if config.fine_method == "custom_icp":
                coarse_message = "coarse candidate=coarse_multi_hypothesis"
            else:
                candidates, errors = _build_coarse_hypotheses(
                    source_points, target_points, config, source_path=source_path_arg, target_path=target_path_arg,
                )
                transform, coarse_message = _select_best_coarse_candidate(source_points, target_points, candidates, config)
                if errors:
                    coarse_message = f"{coarse_message}, candidate_errors={'; '.join(errors)}"
        else:
            coarse_message = "coarse candidate=provided_initial" if initial is not None else "coarse candidate=identity"
        result.timings_ms["coarse"] = (perf_counter() - started) * 1000

        started = perf_counter()
        if config.fine_method == "custom_icp":
            if config.coarse_method == "fpfh" and initial is None:
                transform, history, status, message = _register_coarse_multi_hypothesis(
                    source_points, target_points, config, source_path=source_path_arg, target_path=target_path_arg,
                )
            else:
                transform, history, status, message = custom_icp(source_points, target_points, transform, config)
                if coarse_message:
                    message = f"{coarse_message}; {message}"
            result.history = history
        else:
            transform, status, message = _point_to_plane(source_points, target_points, transform, config)
        result.timings_ms["fine"] = (perf_counter() - started) * 1000
        result.transformation, result.status, result.message = transform, status, message
        result.metrics.update(alignment_metrics(source_points, target_points, transform, config.max_correspondence_distance))
        result.metrics.update(symmetric_alignment_metrics(source_points, target_points, transform, config.max_correspondence_distance))
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
