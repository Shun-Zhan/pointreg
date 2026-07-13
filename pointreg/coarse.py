from __future__ import annotations

from itertools import permutations, product

import numpy as np

from .nearest import nearest_neighbors
from .transforms import apply_transform

from .nearest import nearest_neighbors
from .transforms import apply_transform, make_transform


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


def fpfh_registration(source: np.ndarray, target: np.ndarray, voxel_size: float, seed: int = 42) -> np.ndarray:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("FPFH registration requires Open3D") from exc
    if voxel_size <= 0:
        raise ValueError("FPFH requires voxel_size > 0")
    np.random.seed(seed)
    o3d.utility.random.seed(seed)
    def features(points: np.ndarray):
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        descriptor = o3d.pipelines.registration.compute_fpfh_feature(
            cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
        return cloud, descriptor
    source_cloud, source_feature = features(source)
    target_cloud, target_feature = features(target)
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


def _fpfh_cloud_and_features(points: np.ndarray, voxel_size: float):
    import open3d as o3d

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud = cloud.voxel_down_sample(voxel_size)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    features = o3d.pipelines.registration.compute_fpfh_feature(
        cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )
    return cloud, np.asarray(features.data).T


def _geometric_candidate_score(source: np.ndarray, target: np.ndarray, transform: np.ndarray, distance: float) -> tuple[int, int, float]:
    """Score a hypothesis using reciprocal geometric correspondences only."""
    moved = apply_transform(source, transform)
    source_distances, source_indices = nearest_neighbors(moved, target)
    target_distances, target_indices = nearest_neighbors(target, moved)
    reciprocal = target_indices[source_indices] == np.arange(len(source))
    inliers = reciprocal & (source_distances <= distance) & (target_distances[source_indices] <= distance)
    count = int(inliers.sum())
    all_count = int((source_distances <= distance).sum() + (target_distances <= distance).sum())
    error = float(np.median(source_distances[inliers])) if count else float("inf")
    return count, all_count, error


def multiscale_fpfh_registration(
    source: np.ndarray,
    target: np.ndarray,
    voxel_size: float,
    seed: int = 42,
    correspondence_distance: float = 0.01,
    trials_per_scale: int = 12,
) -> np.ndarray:
    """Run several FPFH-RANSAC hypotheses and retain the geometric-consensus winner."""
    if voxel_size <= 0:
        raise ValueError("multi-scale FPFH requires voxel_size > 0")
    scales = (voxel_size, voxel_size * 1.6, voxel_size * 2.4)
    best_transform: np.ndarray | None = None
    best_score = (-1, -1, float("-inf"))
    for scale_index, scale in enumerate(scales):
        for trial in range(trials_per_scale):
            candidate = fpfh_registration(source, target, scale, seed + scale_index * trials_per_scale + trial)
            reciprocal, total, error = _geometric_candidate_score(source, target, candidate, correspondence_distance)
            # More reciprocal inliers is primary; lower residual breaks ties.
            score = (reciprocal, total, -error)
            if score > best_score:
                best_transform, best_score = candidate, score
    if best_transform is None:
        raise RuntimeError("multi-scale FPFH did not produce a registration hypothesis")
    return best_transform


def gcransac_fpfh_registration(
    source: np.ndarray,
    target: np.ndarray,
    voxel_size: float,
    correspondence_distance: float = 0.01,
) -> np.ndarray:
    """Estimate a rigid transform with GC-RANSAC from mutual FPFH matches."""
    try:
        from scipy.spatial import cKDTree
        import pygcransac
    except ImportError as exc:
        raise RuntimeError("GC-RANSAC requires scipy and pygcransac") from exc
    source_cloud, source_features = _fpfh_cloud_and_features(source, voxel_size)
    target_cloud, target_features = _fpfh_cloud_and_features(target, voxel_size)
    source_points = np.asarray(source_cloud.points)
    target_points = np.asarray(target_cloud.points)
    target_indices = cKDTree(target_features).query(source_features)[1]
    source_indices = cKDTree(source_features).query(target_features)[1]
    matches = np.flatnonzero(source_indices[target_indices] == np.arange(len(source_points)))
    if len(matches) < 3:
        raise RuntimeError(f"GC-RANSAC found only {len(matches)} mutual FPFH correspondences")
    correspondences = np.concatenate([source_points[matches], target_points[target_indices[matches]]], axis=1).astype(np.float64)
    feature_distance = np.linalg.norm(source_features[matches] - target_features[target_indices[matches]], axis=1)
    probabilities = np.exp(-feature_distance / max(float(np.median(feature_distance)), 1e-6))
    model, inliers = pygcransac.findRigidTransform(
        correspondences,
        probabilities.astype(np.float64),
        threshold=correspondence_distance,
        conf=0.999,
        max_iters=10000,
        neighborhood=0,
        use_space_partitioning=True,
    )
    if model is None or int(np.sum(inliers)) < 3:
        raise RuntimeError("GC-RANSAC could not estimate a transform")
    # pygcransac uses row-vector homogeneous coordinates; PointReg uses active
    # column-vector transforms, so transpose its 4x4 output.
    return np.asarray(model, dtype=float).T
