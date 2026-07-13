from __future__ import annotations

from time import perf_counter

import numpy as np

from .models import CoarseCandidate, RegistrationConfig
from .nearest import nearest_neighbors
from .transforms import apply_transform, make_transform


def extract_fpfh(points: np.ndarray, voxel_size: float):
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("FPFH registration requires Open3D") from exc
    if voxel_size <= 0:
        raise ValueError("FPFH requires voxel_size > 0")
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    descriptor = o3d.pipelines.registration.compute_fpfh_feature(
        cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    return cloud, descriptor


def fpfh_correspondences(source_feature, target_feature) -> np.ndarray:
    """Return reciprocal nearest-neighbour FPFH matches as (source, target) indices."""
    from scipy.spatial import cKDTree

    source_desc = np.asarray(source_feature.data).T
    target_desc = np.asarray(target_feature.data).T
    source_to_target = cKDTree(target_desc).query(source_desc, k=1)[1]
    target_to_source = cKDTree(source_desc).query(target_desc, k=1)[1]
    source_indices = np.arange(len(source_desc))
    mutual = target_to_source[source_to_target] == source_indices
    return np.column_stack((source_indices[mutual], source_to_target[mutual])).astype(int)


def relaxed_fpfh_correspondences(source_feature, target_feature, top_k: int = 3,
                                 ratio_threshold: float = 0.9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-k reciprocal FPFH matches filtered by a source-side Lowe ratio test."""
    from scipy.spatial import cKDTree

    source_desc = np.asarray(source_feature.data).T
    target_desc = np.asarray(target_feature.data).T
    if len(source_desc) < 2 or len(target_desc) < 2:
        return np.empty((0, 2), dtype=int), np.empty(0), np.empty(0)
    k_forward = min(max(2, top_k), len(target_desc))
    k_reverse = min(max(1, top_k), len(source_desc))
    forward_distance, forward_index = cKDTree(target_desc).query(source_desc, k=k_forward)
    reverse_index = cKDTree(source_desc).query(target_desc, k=k_reverse)[1]
    if forward_index.ndim == 1:
        forward_index = forward_index[:, None]
        forward_distance = forward_distance[:, None]
    if reverse_index.ndim == 1:
        reverse_index = reverse_index[:, None]
    best_target = forward_index[:, 0]
    ratios = forward_distance[:, 0] / np.maximum(forward_distance[:, 1], 1e-12)
    source_indices = np.arange(len(source_desc))
    reciprocal = np.any(reverse_index[best_target] == source_indices[:, None], axis=1)
    accepted = reciprocal & (ratios <= ratio_threshold)
    matches = np.column_stack((source_indices[accepted], best_target[accepted])).astype(int)
    return matches, forward_distance[accepted, 0], ratios[accepted]


def _fpfh_pair(source: np.ndarray, target: np.ndarray, voxel_size: float):
    return (*extract_fpfh(source, voxel_size), *extract_fpfh(target, voxel_size))


def fpfh_registration(source: np.ndarray, target: np.ndarray, voxel_size: float, seed: int = 42) -> np.ndarray:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("FPFH registration requires Open3D") from exc
    if voxel_size <= 0:
        raise ValueError("FPFH requires voxel_size > 0")
    np.random.seed(seed)
    o3d.utility.random.seed(seed)
    source_cloud, source_feature, target_cloud, target_feature = _fpfh_pair(source, target, voxel_size)
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


def spatial_compatibility(source: np.ndarray, target: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Build first- and second-order rigid-distance compatibility matrices."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("matched source and target must have shape (N, 3)")
    source_dist = np.linalg.norm(source[:, None, :] - source[None, :, :], axis=2)
    target_dist = np.linalg.norm(target[:, None, :] - target[None, :, :], axis=2)
    discrepancy = np.abs(source_dist - target_dist)
    first = np.clip(1.0 - (discrepancy / threshold) ** 2, 0.0, 1.0)
    np.fill_diagonal(first, 0.0)
    second = first * (first @ first)
    scale = second.max()
    if scale > 0:
        second /= scale
    return first, second


def sc2_spectral_scores(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if first.shape != second.shape or first.ndim != 2 or first.shape[0] != first.shape[1]:
        raise ValueError("compatibility matrices must be matching square arrays")
    matrix = 0.5 * (second + second.T)
    if not len(matrix) or not np.any(matrix):
        return np.zeros(len(matrix))
    values, vectors = np.linalg.eigh(matrix)
    scores = np.abs(vectors[:, np.argmax(values)])
    return scores * (first.sum(axis=1) + 1e-12)


def weighted_rigid_svd(source: np.ndarray, target: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if source.shape != target.shape or len(source) < 3 or weights.shape != (len(source),):
        raise ValueError("weighted rigid alignment needs matching points and weights")
    active = weights > 1e-12
    if active.sum() < 3 or weights[active].sum() <= 0:
        raise ValueError("fewer than three weighted correspondences")
    source, target, weights = source[active], target[active], weights[active]
    weights /= weights.sum()
    source_center = np.sum(source * weights[:, None], axis=0)
    target_center = np.sum(target * weights[:, None], axis=0)
    covariance = (source - source_center).T @ ((target - target_center) * weights[:, None])
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    return make_transform(rotation, target_center - rotation @ source_center)


def gnc_tls_registration(source: np.ndarray, target: np.ndarray, noise_bound: float,
                         max_iterations: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Robust rigid alignment with graduated non-convex truncated least squares."""
    if noise_bound <= 0 or len(source) < 3 or source.shape != target.shape:
        raise ValueError("GNC-TLS needs at least three matches and a positive noise bound")
    weights = np.ones(len(source))
    transform = weighted_rigid_svd(source, target, weights)
    residual2 = np.sum((apply_transform(source, transform) - target) ** 2, axis=1)
    c2 = noise_bound ** 2
    maximum = float(residual2.max(initial=0.0))
    mu = max(1e-3, 1.0 / max(2.0 * maximum / c2 - 1.0, 1e-3))
    for _ in range(max_iterations):
        previous = weights.copy()
        upper = (mu + 1.0) / mu * c2
        lower = mu / (mu + 1.0) * c2
        weights = np.ones(len(source))
        weights[residual2 >= upper] = 0.0
        middle = (residual2 > lower) & (residual2 < upper)
        weights[middle] = np.sqrt(c2 * mu * (mu + 1.0) / residual2[middle]) - mu
        if np.count_nonzero(weights > 1e-8) < 3:
            best = np.argsort(residual2)[:3]
            weights[best] = 1.0
        transform = weighted_rigid_svd(source, target, weights)
        residual2 = np.sum((apply_transform(source, transform) - target) ** 2, axis=1)
        if mu >= 1e5 or np.max(np.abs(weights - previous)) < 1e-5:
            break
        mu *= 1.4
    return transform, weights


def sc2_gnc_registration(source: np.ndarray, target: np.ndarray, config: RegistrationConfig) -> np.ndarray:
    if config.voxel_size <= 0:
        raise ValueError("SC2-GNC requires voxel_size > 0")
    _, source_feature, _, target_feature = _fpfh_pair(source, target, config.voxel_size)
    matches = fpfh_correspondences(source_feature, target_feature)
    if len(matches) < config.min_correspondences:
        raise RuntimeError(f"SC2-GNC found only {len(matches)} reciprocal FPFH matches")
    if len(matches) > config.sc2_max_correspondences:
        # Descriptor matches with the smallest feature-space distance are the most useful input.
        source_desc = np.asarray(source_feature.data).T[matches[:, 0]]
        target_desc = np.asarray(target_feature.data).T[matches[:, 1]]
        descriptor_distance = np.linalg.norm(source_desc - target_desc, axis=1)
        matches = matches[np.argsort(descriptor_distance)[:config.sc2_max_correspondences]]
    matched_source, matched_target = source[matches[:, 0]], target[matches[:, 1]]
    threshold = config.voxel_size * config.sc2_distance_threshold_factor
    first, second = spatial_compatibility(matched_source, matched_target, threshold)
    scores = sc2_spectral_scores(first, second)
    keep = max(config.min_correspondences, int(len(matches) * config.sc2_keep_fraction))
    ranked = np.argsort(scores)[::-1]
    selections = [ranked[:min(keep, len(matches))]]
    # Partial and nearly symmetric shapes can contain several strong compatibility
    # clusters. Generate local SC2 hypotheses around the best spectral seeds, then
    # select only by correspondence consensus (never by ground truth).
    for seed in ranked[:min(24, len(ranked))]:
        compatible = np.flatnonzero(first[seed] > 0.5)
        if seed not in compatible:
            compatible = np.append(compatible, seed)
        if len(compatible) < max(6, config.min_correspondences // 2):
            continue
        local_score = scores[compatible] * (first[seed, compatible] + 1e-6)
        selections.append(compatible[np.argsort(local_score)[::-1][:min(keep, len(compatible))]])

    noise_bound = config.voxel_size * config.gnc_noise_bound_factor
    best_transform, best_weights = None, None
    best_quality = (-1, -float("inf"))
    for selected in selections:
        try:
            candidate, weights = gnc_tls_registration(
                matched_source[selected], matched_target[selected], noise_bound, config.gnc_max_iterations)
        except (ValueError, np.linalg.LinAlgError):
            continue
        residual = np.linalg.norm(apply_transform(matched_source, candidate) - matched_target, axis=1)
        consensus = int(np.count_nonzero(residual <= noise_bound * 2.0))
        robust_cost = -float(np.sum(np.minimum(residual, noise_bound * 2.0)))
        quality = (consensus, robust_cost)
        if quality > best_quality:
            best_quality = quality
            best_transform, best_weights = candidate, weights
    if best_transform is None or best_quality[0] < 3 or np.count_nonzero(best_weights > 0.5) < 3:
        raise RuntimeError("SC2-GNC retained fewer than three geometrically consistent matches")
    return best_transform


def _rotation_difference_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3] @ second[:3, :3].T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def deduplicate_candidates(candidates: list[CoarseCandidate], rotation_threshold_deg: float = 0.5,
                           translation_threshold: float = 1e-3) -> list[CoarseCandidate]:
    unique: list[CoarseCandidate] = []
    for candidate in candidates:
        duplicate = any(
            _rotation_difference_deg(candidate.transformation, kept.transformation) <= rotation_threshold_deg
            and np.linalg.norm(candidate.transformation[:3, 3] - kept.transformation[:3, 3]) <= translation_threshold
            for kept in unique)
        if not duplicate:
            unique.append(candidate)
    return unique


def symmetric_full_cloud_score(source: np.ndarray, target: np.ndarray, transform: np.ndarray,
                               max_distance: float) -> tuple[float, float]:
    """Harmonic bidirectional fitness and pooled bidirectional inlier RMSE."""
    moved_source = apply_transform(source, transform)
    forward, _ = nearest_neighbors(moved_source, target)
    moved_target = apply_transform(target, np.linalg.inv(transform))
    reverse, _ = nearest_neighbors(moved_target, source)
    forward_inliers = forward <= max_distance
    reverse_inliers = reverse <= max_distance
    forward_fitness = float(np.mean(forward_inliers))
    reverse_fitness = float(np.mean(reverse_inliers))
    fitness = (2.0 * forward_fitness * reverse_fitness / (forward_fitness + reverse_fitness)
               if forward_fitness + reverse_fitness > 0 else 0.0)
    squared = np.concatenate((forward[forward_inliers] ** 2, reverse[reverse_inliers] ** 2))
    rmse = float(np.sqrt(np.mean(squared))) if len(squared) else float("inf")
    return fitness, rmse


def _short_point_to_plane(source: np.ndarray, target: np.ndarray, initial: np.ndarray,
                          config: RegistrationConfig) -> np.ndarray:
    if config.validation_icp_iterations == 0:
        return initial.copy()
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("candidate validation requires Open3D") from exc
    source_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source))
    target_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
    target_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=max(config.voxel_size * 2, config.max_correspondence_distance), max_nn=30))
    result = o3d.pipelines.registration.registration_icp(
        source_cloud, target_cloud, config.max_correspondence_distance, initial,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=config.validation_icp_iterations))
    return np.asarray(result.transformation)


def verify_coarse_candidates(source: np.ndarray, target: np.ndarray, candidates: list[CoarseCandidate],
                             config: RegistrationConfig) -> tuple[np.ndarray, list[CoarseCandidate]]:
    if not candidates:
        raise RuntimeError("coarse registration produced no candidate transformations")
    candidates = deduplicate_candidates(
        candidates, translation_threshold=max(config.voxel_size * 0.5, 1e-6))[:config.coarse_hypotheses]
    verified: list[CoarseCandidate] = []
    for candidate in candidates:
        started = perf_counter()
        candidate.pre_fitness, candidate.pre_rmse = symmetric_full_cloud_score(
            source, target, candidate.transformation, config.max_correspondence_distance)
        candidate.transformation = _short_point_to_plane(source, target, candidate.transformation, config)
        candidate.verified_fitness, candidate.verified_rmse = symmetric_full_cloud_score(
            source, target, candidate.transformation, config.max_correspondence_distance)
        candidate.validation_ms = (perf_counter() - started) * 1000
        verified.append(candidate)
    verified.sort(key=lambda item: (-item.verified_fitness, item.verified_rmse))
    for rank, candidate in enumerate(verified, 1):
        candidate.validation_rank = rank
        candidate.selected = rank == 1
    return verified[0].transformation.copy(), verified


def _prepare_relaxed_matches(source_feature, target_feature, config: RegistrationConfig):
    matches, distances, ratios = relaxed_fpfh_correspondences(
        source_feature, target_feature, config.feature_match_top_k, config.feature_ratio_threshold)
    if len(matches) < config.min_correspondences:
        raise RuntimeError(f"relaxed FPFH matching found only {len(matches)} correspondences")
    if len(matches) > config.sc2_max_correspondences:
        quality = distances * (1.0 + ratios)
        chosen = np.argsort(quality)[:config.sc2_max_correspondences]
        matches, distances, ratios = matches[chosen], distances[chosen], ratios[chosen]
    return matches, distances, ratios


def fpfh_multi_verified_registration(source: np.ndarray, target: np.ndarray,
                                     config: RegistrationConfig) -> tuple[np.ndarray, list[CoarseCandidate]]:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("multi-RANSAC registration requires Open3D") from exc
    source_cloud, source_feature, target_cloud, target_feature = _fpfh_pair(source, target, config.voxel_size)
    matches, _, _ = _prepare_relaxed_matches(source_feature, target_feature, config)
    correspondence = o3d.utility.Vector2iVector(matches.astype(np.int32))
    candidates: list[CoarseCandidate] = []
    for index in range(config.coarse_hypotheses):
        seed = config.random_seed + index
        np.random.seed(seed)
        o3d.utility.random.seed(seed)
        result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            source_cloud, target_cloud, correspondence, config.voxel_size * 1.5,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(config.voxel_size * 1.5)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        transform = np.asarray(result.transformation)
        residual = np.linalg.norm(apply_transform(source[matches[:, 0]], transform) - target[matches[:, 1]], axis=1)
        inliers = int(np.count_nonzero(residual <= config.voxel_size * 1.5))
        candidates.append(CoarseCandidate(transform.copy(), f"ransac_seed_{seed}", inliers, float(inliers)))
    candidates = deduplicate_candidates(
        candidates, translation_threshold=max(config.voxel_size * 0.5, 1e-6))
    return verify_coarse_candidates(source, target, candidates, config)
