from __future__ import annotations

from time import perf_counter

import numpy as np

from .models import ICPRecord, RegistrationConfig
from .nearest import nearest_neighbors
from .transforms import apply_transform, make_transform, rotation_angle_deg


def solve_rigid_svd(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("source and target must be matching (N, 3) arrays with N >= 3")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return make_transform(rotation, translation)


def custom_icp(source: np.ndarray, target: np.ndarray, initial: np.ndarray, config: RegistrationConfig) -> tuple[np.ndarray, list[ICPRecord], str, str]:
    if len(source) < 3 or len(target) < 3:
        return initial.copy(), [], "failed", "point cloud is empty or too small"
    transform = initial.copy()
    history: list[ICPRecord] = []
    previous_rmse = float("inf")
    for iteration in range(1, config.max_iterations + 1):
        started = perf_counter()
        moved = apply_transform(source, transform)
        distances, indices = nearest_neighbors(moved, target)
        valid_indices = np.flatnonzero(distances <= config.max_correspondence_distance)
        if len(valid_indices) >= config.min_correspondences and config.trim_fraction < 1:
            keep = max(config.min_correspondences, int(len(valid_indices) * config.trim_fraction))
            order = np.argsort(distances[valid_indices])[:keep]
            valid_indices = valid_indices[order]
        if len(valid_indices) < config.min_correspondences:
            return transform, history, "failed", f"only {len(valid_indices)} valid correspondences"
        delta = solve_rigid_svd(moved[valid_indices], target[indices[valid_indices]])
        transform = delta @ transform
        rmse = float(np.sqrt(np.mean(distances[valid_indices] ** 2)))
        rotation_delta = rotation_angle_deg(delta[:3, :3])
        translation_delta = float(np.linalg.norm(delta[:3, 3]))
        history.append(ICPRecord(iteration, rmse, len(valid_indices), rotation_delta, translation_delta, (perf_counter() - started) * 1000))
        rmse_change = abs(previous_rmse - rmse)
        if rmse_change < config.rmse_tolerance and max(np.radians(rotation_delta), translation_delta) < config.transform_tolerance:
            return transform, history, "converged", "convergence tolerances reached"
        previous_rmse = rmse
    return transform, history, "max_iterations", "maximum iterations reached"

