from __future__ import annotations

import numpy as np

from .coarse import _feature_clouds
from .transforms import make_transform


def teaser_available() -> bool:
    try:
        import teaserpp_python  # noqa: F401
    except ImportError:
        return False
    return True


def teaser_registration(source: np.ndarray, target: np.ndarray, voxel_size: float, correspondence_distance: float, seed: int = 42) -> np.ndarray:
    try:
        import open3d as o3d
        import teaserpp_python
    except ImportError as exc:
        raise RuntimeError("TEASER registration requires open3d and teaserpp_python") from exc
    if voxel_size <= 0:
        raise ValueError("TEASER requires voxel_size > 0")
    np.random.seed(seed)
    o3d.utility.random.seed(seed)
    (source_cloud, source_feature), (target_cloud, target_feature) = _feature_clouds(source, target, voxel_size)
    distance = max(correspondence_distance, voxel_size * 1.5)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_cloud, target_cloud, source_feature, target_feature, True,
        distance,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    if not result.correspondence_set:
        raise RuntimeError("no feature correspondences for TEASER")
    source_points = np.asarray(source_cloud.points)
    target_points = np.asarray(target_cloud.points)
    source_corr = np.vstack([source_points[i] for i, _ in result.correspondence_set]).T
    target_corr = np.vstack([target_points[j] for _, j in result.correspondence_set]).T
    if source_corr.shape[1] < 3:
        raise RuntimeError("insufficient TEASER correspondences")
    params = teaserpp_python.RobustRegistrationSolverParams(
        cbar2=1.0,
        noise_bound=distance,
        estimate_scaling=False,
    )
    solver = teaserpp_python.RobustRegistrationSolver(params)
    solver.solve(source_corr, target_corr)
    rotation = np.asarray(solver.getRotationMatrix())
    translation = np.asarray(solver.getTranslationVector()).reshape(3)
    return make_transform(rotation, translation)
