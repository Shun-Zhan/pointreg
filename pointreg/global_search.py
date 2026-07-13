from __future__ import annotations

from dataclasses import replace

import numpy as np
import scipy.fft as scipy_fft
from scipy.spatial.transform import Rotation

from .icp import custom_icp
from .models import RegistrationConfig
from .nearest import NearestNeighborIndex
from .preprocessing import estimate_outward_normals
from .scoring import HypothesisScorer, _DepthMap, _estimate_viewpoint
from .transforms import apply_transform, invert_transform, make_transform, rotation_angle_deg


def super_fibonacci_rotations(count: int) -> np.ndarray:
    """Uniform SO(3) sampling via super-Fibonacci spirals (Alexa 2022)."""
    phi = np.sqrt(2.0)
    psi = 1.533751168755204288118041
    s = np.arange(count) + 0.5
    t = s / count
    d = 2 * np.pi * s
    r = np.sqrt(t)
    big_r = np.sqrt(1 - t)
    alpha = d / phi
    beta = d / psi
    quats = np.stack([r * np.sin(alpha), r * np.cos(alpha),
                      big_r * np.sin(beta), big_r * np.cos(beta)], axis=1)
    return Rotation.from_quat(quats).as_matrix()


def axis_angle_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle_rad) * cross + (1 - np.cos(angle_rad)) * (cross @ cross)


def fibonacci_axes(count: int) -> np.ndarray:
    index = np.arange(count)
    golden = (1 + 5 ** 0.5) / 2
    z = 1 - (2 * index + 1) / count
    radius = np.sqrt(1 - z * z)
    theta = 2 * np.pi * index / golden
    return np.stack([radius * np.cos(theta), radius * np.sin(theta), z], axis=1)


def _occupancy_grid(points: np.ndarray, origin: np.ndarray, voxel: float, dims: np.ndarray) -> np.ndarray:
    grid = np.zeros(tuple(dims), dtype=np.float32)
    index = np.floor((points - origin) / voxel).astype(int)
    valid = ((index >= 0) & (index < dims)).all(axis=1)
    grid[index[valid, 0], index[valid, 1], index[valid, 2]] = 1.0
    return grid


def _signed_grid(points: np.ndarray, origin: np.ndarray, voxel: float, dims: np.ndarray,
                 free_penalty: float, normal_radius: float) -> np.ndarray:
    """Occupancy grid where voxels inside the scanner's observed free space are negative.

    A candidate pose that pushes source points through the target surface lands
    them in free space and collects the penalty, so the correlation prefers
    physically consistent contact instead of shell-through-shell impostors.
    """
    grid = np.zeros(tuple(dims), dtype=np.float32)
    normals = estimate_outward_normals(points, normal_radius)
    depth_map = _DepthMap.build(points, _estimate_viewpoint(points, normals), bin_size=0.004)
    axes = [np.arange(d) for d in dims]
    mesh = np.meshgrid(*axes, indexing="ij")
    centers = origin + (np.stack(mesh, axis=-1).reshape(-1, 3) + 0.5) * voxel
    keys, depths = depth_map._keys(centers)
    min_depths = np.array([depth_map.min_depth.get(key, np.inf) for key in keys.tolist()])
    free = depths < (min_depths - voxel * 1.5)
    grid.ravel()[free] = -free_penalty
    index = np.floor((points - origin) / voxel).astype(int)
    valid = ((index >= 0) & (index < dims)).all(axis=1)
    grid[index[valid, 0], index[valid, 1], index[valid, 2]] = 1.0
    return grid


class SignedCorrelationSolver:
    """Rotation scoring + translation recovery by FFT over a signed voxel grid.

    For a fixed rotation, the circular cross correlation between the rotated
    source occupancy grid and the target signed grid evaluates every
    translation at once; the peak value counts surface contacts minus
    free-space intrusions. Only the two input clouds are used.
    """

    def __init__(self, source: np.ndarray, target: np.ndarray, voxel: float,
                 dims: int = 64, free_penalty: float = 2.0, normal_radius: float = 0.0) -> None:
        self.voxel = voxel
        self.dims = np.array([dims, dims, dims])
        extent = self.dims * voxel
        self.source = source
        self.source_center = source.mean(axis=0)
        self.target_origin = target.mean(axis=0) - extent / 2
        self.source_origin = -extent / 2
        radius = normal_radius if normal_radius > 0 else voxel * 1.5
        self.target_spectrum = scipy_fft.rfftn(
            _signed_grid(target, self.target_origin, voxel, self.dims, free_penalty, radius), workers=4)
        self.count = len(source)

    def _correlation(self, rotation: np.ndarray) -> np.ndarray:
        rotated = (self.source - self.source_center) @ rotation.T
        grid = _occupancy_grid(rotated, self.source_origin, self.voxel, self.dims)
        spectrum = scipy_fft.rfftn(grid, workers=4)
        return scipy_fft.irfftn(spectrum.conj() * self.target_spectrum, s=tuple(self.dims), workers=4)

    def score(self, rotation: np.ndarray) -> float:
        return float(self._correlation(rotation).max()) / self.count

    def best_transform(self, rotation: np.ndarray) -> np.ndarray:
        correlation = self._correlation(rotation)
        cell = np.array(np.unravel_index(int(np.argmax(correlation)), tuple(self.dims)), dtype=float)
        shift = np.where(cell > self.dims / 2, cell - self.dims, cell)
        translation = (self.target_origin - self.source_origin) + shift * self.voxel
        return make_transform(rotation, translation - rotation @ self.source_center)


class GlobalRegistrationSearch:
    """Two-cloud global pose search with a free-space-aware objective.

    Stages: (1) uniform SO(3) grid scored by bidirectional signed correlation,
    (2) stochastic hill climbing from diverse top seeds, (3) fitness-driven
    local refinement with short trimmed-ICP inner polish, (4) tight
    point-to-plane finishing in both directions and two tightness levels,
    (5) gated selection: candidates must keep free-space violation below a
    calibrated bound with non-trivial symmetric fitness, ranked by
    fitness − 10 × violation. Only the two input clouds are ever used.
    """

    def __init__(self, source: np.ndarray, target: np.ndarray, config: RegistrationConfig) -> None:
        self.config = config
        self.voxel = config.voxel_size
        self.source = source
        self.target = target
        self.fast_source = self._sample(source, config.multi_score_points)
        self.fast_target = self._sample(target, config.multi_score_points)
        self.rng = np.random.default_rng(config.random_seed)
        self.forward = SignedCorrelationSolver(self.fast_source, self.fast_target, self.voxel * 2,
                                               normal_radius=self.voxel * 3)
        self.backward = SignedCorrelationSolver(self.fast_target, self.fast_source, self.voxel * 2,
                                                normal_radius=self.voxel * 3)
        self.scorer = HypothesisScorer(self.fast_source, self.fast_target, self.voxel, self.voxel * 2)
        # margin=voxel*0.8 的细违规打分器：捕捉毫米级切向滑移带来的轻微穿透
        self.fine_scorer = HypothesisScorer(self.fast_source, self.fast_target, self.voxel * 0.4,
                                            self.voxel * 2)
        self.fine_forward: SignedCorrelationSolver | None = None
        self.target_tree = NearestNeighborIndex(self.fast_target)

    @staticmethod
    def _sample(points: np.ndarray, limit: int) -> np.ndarray:
        if len(points) <= limit:
            return points
        return points[:: max(1, len(points) // limit)]

    # ---------- rotation-level objective ----------

    def _biscore(self, rotation: np.ndarray) -> float:
        return min(self.forward.score(rotation), self.backward.score(rotation.T))

    def _hill_climb(self, rotation: np.ndarray, score: float, rounds: int) -> tuple[np.ndarray, float]:
        radius, stall = 9.0, 0
        for _ in range(rounds):
            candidate = axis_angle_matrix(self.rng.normal(size=3),
                                          np.radians(radius) * self.rng.uniform(0.3, 1.0)) @ rotation
            candidate_score = self._biscore(candidate)
            if candidate_score > score:
                rotation, score, stall = candidate, candidate_score, 0
            else:
                stall += 1
                if stall >= 4:
                    radius, stall = max(1.5, radius * 0.6), 0
        return rotation, score

    # ---------- pose-level refinement ----------

    def _short_icp(self, transform: np.ndarray) -> np.ndarray:
        result = transform
        for distance_scale, iterations in ((1.5, 10), (1.0, 15), (0.7, 15)):
            stage = replace(self.config,
                            max_correspondence_distance=self.config.max_correspondence_distance * distance_scale,
                            max_iterations=iterations, trim_fraction=0.5, adaptive_trim=True,
                            min_correspondences=10)
            polished, _, status, _ = custom_icp(self.fast_source, self.fast_target, result, stage)
            if status != "failed":
                result = polished
        return result

    def _fine_fitness(self, transform: np.ndarray) -> float:
        moved = apply_transform(self.fast_source, transform)
        forward, _ = self.target_tree.query(moved)
        backward, _ = NearestNeighborIndex(moved).query(self.fast_target)
        threshold = self.voxel * 1.2
        return min(float(np.mean(forward <= threshold)), float(np.mean(backward <= threshold)))

    def _refine_objective(self, transform: np.ndarray) -> float:
        """Fine fitness with a free-space penalty; drifted poses gain fitness but
        park points in observed free space, so the penalty keeps the climb honest.
        The floor guard rejects escapes into zero-overlap nothingness."""
        fitness = self._fine_fitness(transform)
        if fitness < 0.03:
            return float("-inf")
        return fitness - 3.0 * self.scorer.violation(transform)

    def _fitness_refine(self, rotation: np.ndarray) -> np.ndarray:
        """Objective-climb over rotations; translation re-solved by FFT + short trimmed ICP."""
        current = self._short_icp(self.forward.best_transform(rotation))
        current_score = self._refine_objective(current)
        radius = 8.0
        for _ in range(3):
            for _ in range(self.config.multi_refine_samples):
                candidate_rotation = axis_angle_matrix(
                    self.rng.normal(size=3), np.radians(radius) * self.rng.uniform(0.1, 1.0)) @ current[:3, :3]
                candidate = self._short_icp(self.forward.best_transform(candidate_rotation))
                score = self._refine_objective(candidate)
                if score > current_score:
                    current, current_score = candidate, score
            radius = max(1.5, radius * 0.5)
        return current

    # ---------- tight finishing ----------

    def _tight_p2p(self, source_cloud, target_cloud, initial: np.ndarray,
                   distance: float, tukey_k: float, iterations: int = 60) -> np.ndarray:
        import open3d as o3d
        loss = o3d.pipelines.registration.TukeyLoss(k=tukey_k)
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations)
        result = o3d.pipelines.registration.registration_icp(
            source_cloud, target_cloud, distance, initial, estimation, criteria)
        return np.asarray(result.transformation)

    def _make_pcd(self, points: np.ndarray):
        import open3d as o3d
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel * 4, max_nn=30))
        return cloud

    def _finish_candidates(self, source_cloud, target_cloud, initial: np.ndarray) -> list[np.ndarray]:
        candidates = []
        entry_forward = self._tight_p2p(source_cloud, target_cloud, initial, self.voxel * 1.2, self.voxel * 0.6)
        entry_backward = self._tight_p2p(target_cloud, source_cloud, invert_transform(initial),
                                         self.voxel * 1.2, self.voxel * 0.6)
        for distance_scale, k_scale in ((0.8, 0.4), (0.4, 0.2)):
            candidates.append(self._tight_p2p(source_cloud, target_cloud, entry_forward,
                                              self.voxel * distance_scale, self.voxel * k_scale))
            candidates.append(invert_transform(
                self._tight_p2p(target_cloud, source_cloud, entry_backward,
                                self.voxel * distance_scale, self.voxel * k_scale)))
        return candidates

    # ---------- local grid lock ----------

    def _local_grid_lock(self, rotation_estimate: np.ndarray) -> list[np.ndarray]:
        """Deterministic rotation grid around an estimate with FFT translation re-solve.

        A residual rotation error of δ shifts the correlation-optimal translation
        by roughly radius·sin(δ), so translation can only be pinned down after
        the rotation is locked to ~1°. The grid guarantees a rotation within
        ~0.7 step of the truth if the estimate is within the span.
        """
        span = self.config.multi_lock_span_deg
        step = self.config.multi_lock_step_deg
        offsets = np.arange(-span, span + 1e-9, step)
        scored: list[tuple[float, np.ndarray]] = []
        for offset_x in offsets:
            for offset_y in offsets:
                for offset_z in offsets:
                    rotation = rotation_estimate
                    for axis, angle in zip(np.eye(3), (offset_x, offset_y, offset_z)):
                        if angle:
                            rotation = axis_angle_matrix(axis, np.radians(angle)) @ rotation
                    scored.append((self.forward.score(rotation), rotation))
        scored.sort(key=lambda item: -item[0])
        if self.fine_forward is None:
            self.fine_forward = SignedCorrelationSolver(self.fast_source, self.fast_target, self.voxel,
                                                        dims=96, normal_radius=self.voxel * 3)
        return [self.fine_forward.best_transform(rotation)
                for _, rotation in scored[: self.config.multi_lock_top]]

    # ---------- candidate metrics and selection ----------

    def _candidate_metrics(self, transform: np.ndarray) -> tuple[float, float, float, float, float]:
        fitness, _ = self.scorer.symmetric_fitness(transform)
        violation = self.scorer.violation(transform)
        fine_violation = self.fine_scorer.violation(transform)
        moved = apply_transform(self.fast_source, transform)
        forward, _ = self.target_tree.query(moved)
        backward, _ = NearestNeighborIndex(moved).query(self.fast_target)
        threshold = self.voxel * 2
        forward_in = forward[forward <= threshold]
        backward_in = backward[backward <= threshold]
        rmse = max(float(np.sqrt(np.mean(forward_in ** 2))) if len(forward_in) else 9e9,
                   float(np.sqrt(np.mean(backward_in ** 2))) if len(backward_in) else 9e9)
        fine_fitness = min(float(np.mean(forward <= self.voxel * 0.8)),
                           float(np.mean(backward <= self.voxel * 0.8)))
        return fitness, violation, fine_violation, rmse, fine_fitness

    def _select(self, candidates: list[np.ndarray]) -> tuple[np.ndarray, float, float, bool]:
        """Gate on free-space consistency, then Borda-vote across four metrics.

        Tangential sliding on low-overlap pairs creates impostor optima a few
        millimetres away that beat any single surface metric; rank aggregation
        across coarse/fine fitness, fine violation and inlier RMSE is markedly
        more stable than any one of them.
        """
        rows = [(transform, *self._candidate_metrics(transform)) for transform in candidates]
        gated = [row for row in rows
                 if row[2] <= self.config.multi_violation_gate and row[1] >= self.config.multi_min_fitness]
        pool = gated if gated else rows
        rank_keys = (lambda row: -(row[1] - 10.0 * row[2]),
                     lambda row: -(row[1] - 10.0 * row[3]),
                     lambda row: row[4],
                     lambda row: -(row[5] - 5.0 * row[3]))
        totals = np.zeros(len(pool))
        for key in rank_keys:
            order = sorted(range(len(pool)), key=lambda i: key(pool[i]))
            for rank, index in enumerate(order):
                totals[index] += rank
        best = pool[int(np.argmin(totals))]
        return best[0], best[1], best[2], bool(gated)

    def run(self) -> tuple[np.ndarray, str]:
        rotations = super_fibonacci_rotations(self.config.multi_grid_rotations)
        scores = np.array([self._biscore(rotation) for rotation in rotations])
        order = np.argsort(-scores)
        quats = Rotation.from_matrix(rotations).as_quat()

        seeds: list[tuple[float, np.ndarray]] = []
        seed_quats: list[np.ndarray] = []
        for index in order:
            if len(seeds) >= self.config.multi_seed_count:
                break
            quat = quats[index]
            if all(2 * np.degrees(np.arccos(min(1.0, abs(float(quat @ other))))) > 15.0
                   for other in seed_quats):
                seeds.append((float(scores[index]), rotations[index]))
                seed_quats.append(quat)

        climbed = [self._hill_climb(rotation, score, self.config.multi_climb_rounds)
                   for score, rotation in seeds]
        climbed.sort(key=lambda item: -item[1])

        refine_pool: list[np.ndarray] = []
        picked_quats: list[np.ndarray] = []
        for rotation, _ in climbed:
            if len(refine_pool) >= self.config.multi_refine_seeds:
                break
            quat = Rotation.from_matrix(rotation).as_quat()
            if all(2 * np.degrees(np.arccos(min(1.0, abs(float(quat @ other))))) > 10.0
                   for other in picked_quats):
                refine_pool.append(rotation)
                picked_quats.append(quat)

        source_cloud = self._make_pcd(self.source)
        target_cloud = self._make_pcd(self.target)
        candidates: list[np.ndarray] = []
        refined_poses: list[np.ndarray] = []
        for rotation in refine_pool:
            refined = self._fitness_refine(rotation)
            refined_poses.append(refined)
            candidates.append(refined)
            candidates.extend(self._finish_candidates(source_cloud, target_cloud, refined))
        stage_one, _, _, _ = self._select(candidates)

        # 网格锁定：围绕第一阶段旋转做确定性细网格 + 细FFT平移，压住切向滑移
        for start in self._local_grid_lock(stage_one[:3, :3]):
            candidates.append(start)
            candidates.extend(self._finish_candidates(source_cloud, target_cloud, start))

        transform, fitness, violation, gate = self._select(candidates)
        gate_state = "pass" if gate else "violated"
        return transform, f"global_search fitness={fitness:.3f} violation={violation:.3f} gate={gate_state}"
