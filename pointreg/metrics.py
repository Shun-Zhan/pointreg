from __future__ import annotations

import numpy as np

from .nearest import nearest_neighbors
from .transforms import apply_transform, invert_transform, rotation_angle_deg


def pose_errors(estimated: np.ndarray, ground_truth: np.ndarray) -> tuple[float, float]:
    delta = invert_transform(ground_truth) @ estimated
    return rotation_angle_deg(delta[:3, :3]), float(np.linalg.norm(delta[:3, 3]))


def alignment_metrics(source: np.ndarray, target: np.ndarray, transform: np.ndarray, threshold: float) -> dict[str, float]:
    moved = apply_transform(source, transform)
    distances, _ = nearest_neighbors(moved, target)
    valid = distances <= threshold
    return {
        "rmse": float(np.sqrt(np.mean(distances[valid] ** 2))) if np.any(valid) else float("inf"),
        "fitness": float(np.mean(valid)) if len(valid) else 0.0,
        "correspondences": float(np.count_nonzero(valid)),
    }


def symmetric_overlap(source: np.ndarray, target: np.ndarray, ground_truth: np.ndarray, threshold: float) -> float:
    forward, _ = nearest_neighbors(apply_transform(source, ground_truth), target)
    backward, _ = nearest_neighbors(target, apply_transform(source, ground_truth))
    a = float(np.mean(forward <= threshold)) if len(forward) else 0.0
    b = float(np.mean(backward <= threshold)) if len(backward) else 0.0
    return (a + b) / 2.0

