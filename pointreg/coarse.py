from __future__ import annotations

from itertools import permutations, product

import numpy as np

from .nearest import nearest_neighbors
from .transforms import apply_transform, make_transform


def pca_axis_candidates(source: np.ndarray, target: np.ndarray) -> list[np.ndarray]:
    """Enumerate the 24 proper-rotation axis alignments between two PCA frames."""
    if len(source) < 3 or len(target) < 3:
        raise ValueError("PCA registration needs at least three points per cloud")
    cs, ct = source.mean(axis=0), target.mean(axis=0)
    _, es = np.linalg.eigh(np.cov((source - cs).T))
    _, et = np.linalg.eigh(np.cov((target - ct).T))
    es, et = es[:, ::-1], et[:, ::-1]
    candidates = []
    for perm in permutations(range(3)):
        target_axes = et[:, perm]
        for signs in product((-1.0, 1.0), repeat=3):
            rotation = (target_axes @ np.diag(signs)) @ es.T
            if np.linalg.det(rotation) < 0:
                continue
            candidates.append(make_transform(rotation, ct - rotation @ cs))
    return candidates


def pca_registration(source: np.ndarray, target: np.ndarray, sample_limit: int = 5000) -> np.ndarray:
    candidates = pca_hypotheses(source, target, top_k=1, sample_limit=sample_limit)
    return candidates[0]


def pca_hypotheses(source: np.ndarray, target: np.ndarray, top_k: int = 6, sample_limit: int = 5000) -> list[np.ndarray]:
    """Return the top-k PCA axis-alignment hypotheses ranked by median residual."""
    candidates = pca_axis_candidates(source, target)
    sample = source[::max(1, len(source) // sample_limit)]
    scored = []
    for candidate in candidates:
        distances, _ = nearest_neighbors(apply_transform(sample, candidate), target)
        scored.append((float(np.median(distances)), candidate))
    scored.sort(key=lambda item: item[0])
    return [candidate for _, candidate in scored[:max(1, top_k)]]


def _axis_angle_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle_rad) * cross + (1 - np.cos(angle_rad)) * (cross @ cross)


def axis_grid_hypotheses(source: np.ndarray, target: np.ndarray, step_deg: float = 15.0) -> list[np.ndarray]:
    """Enumerate rotations about the clouds' principal inertia axes (turntable prior).

    Uses only the two input clouds: candidate axes come from the covariance
    eigenvectors of each cloud, translation aligns the centroids.
    """
    if len(source) < 3 or len(target) < 3:
        return []
    cs, ct = source.mean(axis=0), target.mean(axis=0)
    _, es = np.linalg.eigh(np.cov((source - cs).T))
    _, et = np.linalg.eigh(np.cov((target - ct).T))
    axes: list[np.ndarray] = []
    for basis in (es, et):
        for column in range(3):
            axis = basis[:, column]
            if not any(abs(float(axis @ known)) > 0.98 for known in axes):
                axes.append(axis)
    angles = np.radians(np.arange(step_deg, 360.0, step_deg))
    hypotheses = []
    for axis in axes:
        for angle in angles:
            rotation = _axis_angle_rotation(axis, float(angle))
            hypotheses.append(make_transform(rotation, ct - rotation @ cs))
    return hypotheses


def _open3d_features(points: np.ndarray, voxel_size: float):
    import open3d as o3d
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    descriptor = o3d.pipelines.registration.compute_fpfh_feature(
        cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    return cloud, descriptor


def fpfh_registration(source: np.ndarray, target: np.ndarray, voxel_size: float, seed: int = 42,
                      mutual_filter: bool = True, max_iterations: int = 200000, confidence: float = 0.9995) -> np.ndarray:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("FPFH registration requires Open3D") from exc
    if voxel_size <= 0:
        raise ValueError("FPFH requires voxel_size > 0")
    np.random.seed(seed)
    o3d.utility.random.seed(seed)
    source_cloud, source_feature = _open3d_features(source, voxel_size)
    target_cloud, target_feature = _open3d_features(target, voxel_size)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_cloud, target_cloud, source_feature, target_feature, mutual_filter,
        voxel_size * 1.5,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(max_iterations, confidence),
    )
    return np.asarray(result.transformation)


def fgr_registration(source: np.ndarray, target: np.ndarray, voxel_size: float, seed: int = 42) -> np.ndarray:
    """Fast Global Registration on FPFH features; robust on low-overlap pairs."""
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("FGR registration requires Open3D") from exc
    if voxel_size <= 0:
        raise ValueError("FGR requires voxel_size > 0")
    np.random.seed(seed)
    o3d.utility.random.seed(seed)
    source_cloud, source_feature = _open3d_features(source, voxel_size)
    target_cloud, target_feature = _open3d_features(target, voxel_size)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_cloud, target_cloud, source_feature, target_feature,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=voxel_size * 1.5))
    return np.asarray(result.transformation)
