from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .nearest import NearestNeighborIndex
from .preprocessing import estimate_outward_normals
from .transforms import apply_transform, invert_transform


def _estimate_viewpoint(points: np.ndarray, normals: np.ndarray | None) -> np.ndarray:
    """Approximate the scanner location from the mean outward normal.

    A range scan only records the surface visible from one side, so the mean
    outward normal points roughly back at the sensor. Only the two input
    clouds are used; no ground-truth pose is involved.
    """
    center = points.mean(axis=0)
    diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    if normals is None or not len(normals):
        return center + np.array([0.0, 0.0, diagonal * 8.0])
    direction = normals.mean(axis=0)
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return center + np.array([0.0, 0.0, diagonal * 8.0])
    return center + direction / norm * diagonal * 8.0


@dataclass(slots=True)
class _DepthMap:
    viewpoint: np.ndarray
    axis_u: np.ndarray
    axis_v: np.ndarray
    bin_size: float
    min_depth: dict[int, float] = field(default_factory=dict)

    def _keys(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rays = points - self.viewpoint
        depths = np.linalg.norm(rays, axis=1)
        directions = rays / depths[:, None]
        u = directions @ self.axis_u
        v = directions @ self.axis_v
        keys = np.round(u / self.bin_size).astype(np.int64) * 200000 + np.round(v / self.bin_size).astype(np.int64)
        return keys, depths

    @classmethod
    def build(cls, points: np.ndarray, viewpoint: np.ndarray, bin_size: float) -> "_DepthMap":
        rays = points - viewpoint
        depths = np.linalg.norm(rays, axis=1)
        directions = rays / depths[:, None]
        forward = directions.mean(axis=0)
        forward /= np.linalg.norm(forward)
        axis_u = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        if np.linalg.norm(axis_u) < 1e-6:
            axis_u = np.cross(forward, np.array([1.0, 0.0, 0.0]))
        axis_u /= np.linalg.norm(axis_u)
        axis_v = np.cross(forward, axis_u)
        depth_map = cls(viewpoint=viewpoint, axis_u=axis_u, axis_v=axis_v, bin_size=bin_size)
        keys, depths = depth_map._keys(points)
        order = np.argsort(keys)
        sorted_keys, sorted_depths = keys[order], depths[order]
        unique_keys, starts = np.unique(sorted_keys, return_index=True)
        minima = np.minimum.reduceat(sorted_depths, starts)
        depth_map.min_depth = dict(zip(unique_keys.tolist(), minima.tolist()))
        return depth_map

    def violation_ratio(self, points: np.ndarray, margin: float) -> float:
        """Fraction of points that sit inside the scanner's observed free space."""
        keys, depths = self._keys(points)
        bad = 0
        covered = 0
        for key, depth in zip(keys.tolist(), depths.tolist()):
            min_depth = self.min_depth.get(key)
            if min_depth is None:
                continue
            covered += 1
            if depth < min_depth - margin:
                bad += 1
        return bad / max(1, covered)


class HypothesisScorer:
    """Scores candidate transforms on symmetric fitness and free-space consistency.

    The physical constraint: every ray from a scanner to a measured point passes
    through empty space. A candidate alignment that places the other cloud's
    points inside that free space is geometrically impossible, no matter how
    high its nearest-neighbor fitness is. This lets low-overlap correct poses
    (low fitness, zero violation) beat high-fitness impostor poses.
    """

    def __init__(self, source: np.ndarray, target: np.ndarray, voxel_size: float,
                 fitness_threshold: float, bin_size: float = 0.004) -> None:
        self.source = source
        self.target = target
        self.fitness_threshold = fitness_threshold
        self.margin = voxel_size * 2
        self.target_index = NearestNeighborIndex(target)
        source_normals = estimate_outward_normals(source, voxel_size * 3)
        target_normals = estimate_outward_normals(target, voxel_size * 3)
        self.source_depth = _DepthMap.build(source, _estimate_viewpoint(source, source_normals), bin_size)
        self.target_depth = _DepthMap.build(target, _estimate_viewpoint(target, target_normals), bin_size)

    def symmetric_fitness(self, transform: np.ndarray) -> tuple[float, float]:
        moved = apply_transform(self.source, transform)
        forward, _ = self.target_index.query(moved)
        backward, _ = NearestNeighborIndex(moved).query(self.target)
        fitness_forward = float(np.mean(forward <= self.fitness_threshold)) if len(forward) else 0.0
        fitness_backward = float(np.mean(backward <= self.fitness_threshold)) if len(backward) else 0.0
        inliers = forward[forward <= self.fitness_threshold]
        rmse = float(np.sqrt(np.mean(inliers ** 2))) if len(inliers) else float("inf")
        return min(fitness_forward, fitness_backward), rmse

    def violation(self, transform: np.ndarray) -> float:
        forward = self.target_depth.violation_ratio(apply_transform(self.source, transform), self.margin)
        backward = self.source_depth.violation_ratio(apply_transform(self.target, invert_transform(transform)), self.margin)
        return max(forward, backward)

    def score(self, transform: np.ndarray) -> tuple[float, float, float]:
        """Return (combined score, fitness, violation); higher score is better."""
        fitness, _ = self.symmetric_fitness(transform)
        violation = self.violation(transform)
        return fitness - 3.0 * violation, fitness, violation
