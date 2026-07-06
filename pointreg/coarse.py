from __future__ import annotations

from itertools import permutations, product

import numpy as np

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
