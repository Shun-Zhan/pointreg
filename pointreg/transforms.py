from __future__ import annotations

import numpy as np


def quaternion_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        raise ValueError("quaternion must contain x, y, z, w")
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("zero quaternion is invalid")
    x, y, z, w = q / norm
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def make_transform(rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> np.ndarray:
    transform = np.eye(4)
    if rotation is not None:
        transform[:3, :3] = np.asarray(rotation, dtype=float)
    if translation is not None:
        transform[:3, 3] = np.asarray(translation, dtype=float)
    return transform


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    return points @ transform[:3, :3].T + transform[:3, 3]


def invert_transform(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return make_transform(rotation.T, -rotation.T @ translation)


def relative_transform(source_to_world: np.ndarray, target_to_world: np.ndarray) -> np.ndarray:
    return invert_transform(target_to_world) @ source_to_world


def rotation_angle_deg(rotation: np.ndarray) -> float:
    value = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))
