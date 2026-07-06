from __future__ import annotations

import numpy as np


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if voxel_size <= 0 or not len(points):
        return points.copy()
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    result = np.column_stack([
        np.bincount(inverse, weights=points[:, axis]) / counts for axis in range(3)
    ])
    return result


def preprocess_points(points: np.ndarray, voxel_size: float, remove_outliers: bool = False) -> np.ndarray:
    down = voxel_downsample(points, voxel_size)
    if not remove_outliers or len(down) < 30:
        return down
    try:
        import open3d as o3d
    except ImportError:
        return down
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(down))
    filtered, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return np.asarray(filtered.points)


def bounding_box_diagonal(*clouds: np.ndarray) -> float:
    valid = [np.asarray(c) for c in clouds if len(c)]
    if not valid:
        return 0.0
    joined = np.vstack(valid)
    return float(np.linalg.norm(joined.max(axis=0) - joined.min(axis=0)))

