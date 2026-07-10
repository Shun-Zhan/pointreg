from __future__ import annotations

from itertools import permutations, product

import numpy as np

from .nearest import nearest_neighbors
from .transforms import apply_transform, make_transform


def _rotation_around_axis(axis: np.ndarray, angle: float, center: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    t = 1.0 - c
    rotation = np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ])
    return make_transform(rotation, center - rotation @ center)


def _target_pca_axes(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = target.mean(axis=0)
    _, eigenvectors = np.linalg.eigh(np.cov((target - center).T))
    return center, eigenvectors


def pca_registration(source: np.ndarray, target: np.ndarray, sample_limit: int = 5000) -> np.ndarray:
    if len(source) < 3 or len(target) < 3:
        raise ValueError("PCA registration needs at least three points per cloud")
    cs, ct = source.mean(axis=0), target.mean(axis=0)
    _, es = np.linalg.eigh(np.cov((source - cs).T))
    _, et = np.linalg.eigh(np.cov((target - ct).T))
    es, et = es[:, ::-1], et[:, ::-1]
    sample = source[::max(1, len(source) // sample_limit)]
    best_score, best = float("inf"), np.eye(4)
    for perm in permutations(range(3)):
        target_axes = et[:, perm]
        for signs in product((-1.0, 1.0), repeat=3):
            candidate_axes = target_axes @ np.diag(signs)
            rotation = candidate_axes @ es.T
            if np.linalg.det(rotation) < 0:
                continue
            candidate = make_transform(rotation, ct - rotation @ cs)
            distances, _ = nearest_neighbors(apply_transform(sample, candidate), target)
            score = float(np.median(distances))
            if score < best_score:
                best_score, best = score, candidate
    return best


def rotation_grid_hypotheses(source: np.ndarray, target: np.ndarray, steps: int) -> list[tuple[str, np.ndarray]]:
    if steps < 1:
        return []
    pca_transform = pca_registration(source, target)
    center, eigenvectors = _target_pca_axes(target)
    axis = eigenvectors[:, 0]
    candidates: list[tuple[str, np.ndarray]] = [("pca", pca_transform)]
    for index in range(1, steps):
        angle = index * 2.0 * np.pi / steps
        rotation = _rotation_around_axis(axis, angle, center)
        candidates.append((f"pca_rot_{index}", rotation @ pca_transform))
    return candidates


def reflection_hypotheses(base_transform: np.ndarray, target: np.ndarray, name_prefix: str = "pca_reflect") -> list[tuple[str, np.ndarray]]:
    center, eigenvectors = _target_pca_axes(target)
    axis = eigenvectors[:, 1]
    flip = _rotation_around_axis(axis, np.pi, center)
    return [(name_prefix, flip @ base_transform)]


def _feature_clouds(source: np.ndarray, target: np.ndarray, voxel_size: float):
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("feature registration requires Open3D") from exc

    def features(points: np.ndarray):
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        descriptor = o3d.pipelines.registration.compute_fpfh_feature(
            cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
        return cloud, descriptor

    return features(source), features(target)


def fpfh_registration(source: np.ndarray, target: np.ndarray, voxel_size: float, seed: int = 42) -> np.ndarray:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("FPFH registration requires Open3D") from exc
    if voxel_size <= 0:
        raise ValueError("FPFH requires voxel_size > 0")
    np.random.seed(seed)
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("FPFH registration requires Open3D") from exc
    o3d.utility.random.seed(seed)
    (source_cloud, source_feature), (target_cloud, target_feature) = _feature_clouds(source, target, voxel_size)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_cloud, target_cloud, source_feature, target_feature, True,
        voxel_size * 1.5,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return np.asarray(result.transformation)


def fgr_registration(source: np.ndarray, target: np.ndarray, voxel_size: float, seed: int = 42) -> np.ndarray:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("FGR registration requires Open3D") from exc
    if voxel_size <= 0:
        raise ValueError("FGR requires voxel_size > 0")
    np.random.seed(seed)
    o3d.utility.random.seed(seed)
    (source_cloud, source_feature), (target_cloud, target_feature) = _feature_clouds(source, target, voxel_size)
    option = o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=voxel_size * 1.5,
    )
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_cloud, target_cloud, source_feature, target_feature, option,
    )
    return np.asarray(result.transformation)
