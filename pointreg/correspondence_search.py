"""Feature-correspondence hypothesis tools for low-overlap registration.

Multi-scale bidirectional FPFH matching produces a large correspondence pool
whose inlier rate is far below 1% on low-overlap pairs, yet the true matches
are metrically consistent: given any pose within a few degrees / millimetres
of the truth they form the dominant consensus set.  These helpers therefore
do not try to *find* the pose from the graph alone (vote/clique ranking was
measured to be swamped by structured outliers on the Bunny hard pairs);
instead they anchor and polish pose hypotheses coming from the global
search, and provide a consensus metric usable during candidate selection.

Only the two input clouds are ever used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .preprocessing import voxel_downsample


@dataclass(slots=True)
class CorrespondencePool:
    """Flattened multi-scale correspondence set (source_xyz[i] <-> target_xyz[i])."""

    source_xyz: np.ndarray
    target_xyz: np.ndarray

    def __len__(self) -> int:
        return len(self.source_xyz)


def _normalized_fpfh(points: np.ndarray, voxel: float) -> np.ndarray:
    import open3d as o3d

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=30))
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 10, max_nn=120))
    matrix = np.asarray(feature.data).T
    return (matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)).astype(np.float32)


def build_correspondence_pool(source: np.ndarray, target: np.ndarray,
                              scales: tuple[tuple[float, int], ...] = ((0.0025, 6), (0.004, 8)),
                              ) -> CorrespondencePool:
    """Multi-scale bidirectional top-k FPFH matching.

    Bidirectional k-NN (source→target and target→source) is what keeps the
    true inliers in the pool at all: mutual filtering was measured to drop
    them to zero on the hardest pairs.
    """
    source_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    for voxel, k in scales:
        source_pts = voxel_downsample(source, voxel).astype(np.float32)
        target_pts = voxel_downsample(target, voxel).astype(np.float32)
        source_feat = _normalized_fpfh(source_pts, voxel)
        target_feat = _normalized_fpfh(target_pts, voxel)
        _, forward = cKDTree(target_feat).query(source_feat, k=k)
        _, backward = cKDTree(source_feat).query(target_feat, k=k)
        source_parts.append(source_pts[np.repeat(np.arange(len(source_pts)), k)])
        target_parts.append(target_pts[forward.ravel()])
        source_parts.append(source_pts[backward.ravel()])
        target_parts.append(target_pts[np.repeat(np.arange(len(target_pts)), k)])
    return CorrespondencePool(np.vstack(source_parts), np.vstack(target_parts))


def consensus_count(pool: CorrespondencePool, transform: np.ndarray, threshold: float = 0.003) -> int:
    """Number of correspondences consistent with the pose within ``threshold``."""
    rotation = transform[:3, :3].astype(np.float32)
    translation = transform[:3, 3].astype(np.float32)
    residual = np.linalg.norm(pool.target_xyz - pool.source_xyz @ rotation.T - translation, axis=1)
    return int((residual < threshold).sum())


def _svd_pose(source_pts: np.ndarray, target_pts: np.ndarray) -> np.ndarray:
    center_s, center_t = source_pts.mean(axis=0), target_pts.mean(axis=0)
    h = (source_pts - center_s).T @ (target_pts - center_t)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = center_t - rotation @ center_s
    return transform


def consensus_refine(pool: CorrespondencePool, transform: np.ndarray,
                     thresholds: tuple[float, ...] = (0.006, 0.004, 0.003),
                     min_support: int = 6) -> tuple[np.ndarray, int]:
    """Iteratively re-fit the pose on its correspondence consensus set.

    Acts like a correspondence-space ICP: each round keeps matches within the
    (shrinking) threshold and re-solves the rigid pose by SVD.  Returns the
    refined pose and the final support size; the input pose is returned
    unchanged when support is too small to trust.
    """
    current = transform
    support = 0
    for threshold in thresholds:
        rotation = current[:3, :3].astype(np.float32)
        translation = current[:3, 3].astype(np.float32)
        residual = np.linalg.norm(pool.target_xyz - pool.source_xyz @ rotation.T - translation, axis=1)
        mask = residual < threshold
        support = int(mask.sum())
        if support < min_support:
            return transform, support
        current = _svd_pose(pool.source_xyz[mask].astype(np.float64),
                            pool.target_xyz[mask].astype(np.float64))
    return current, support
