import numpy as np

from pointreg.coarse import (deduplicate_candidates, gnc_tls_registration,
                             relaxed_fpfh_correspondences, sc2_spectral_scores,
                             spatial_compatibility, symmetric_full_cloud_score,
                             verify_coarse_candidates)
from pointreg.icp import _select_correspondences
from pointreg.models import CoarseCandidate, RegistrationConfig
from pointreg.transforms import apply_transform, make_transform, rotation_angle_deg


def rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_sc2_scores_rank_geometrically_consistent_matches_first():
    rng = np.random.default_rng(102)
    inlier_source = rng.normal(size=(24, 3))
    transform = make_transform(rotation_z(0.4), np.array([0.4, -0.2, 0.1]))
    inlier_target = apply_transform(inlier_source, transform)
    outlier_source = rng.normal(size=(72, 3))
    outlier_target = rng.normal(size=(72, 3)) * 2.0
    source = np.vstack((inlier_source, outlier_source))
    target = np.vstack((inlier_target, outlier_target))
    first, second = spatial_compatibility(source, target, threshold=0.08)
    scores = sc2_spectral_scores(first, second)
    top = np.argsort(scores)[-24:]
    assert np.count_nonzero(top < 24) >= 20


def test_gnc_tls_rejects_many_large_residual_correspondences():
    rng = np.random.default_rng(103)
    source = rng.uniform(-1, 1, size=(120, 3))
    expected = make_transform(rotation_z(0.12), np.array([0.08, -0.04, 0.03]))
    target = apply_transform(source, expected)
    target += rng.normal(scale=0.002, size=target.shape)
    outliers = rng.choice(len(source), 36, replace=False)
    target[outliers] += rng.normal(scale=0.6, size=(len(outliers), 3))
    estimated, weights = gnc_tls_registration(source, target, noise_bound=0.015, max_iterations=80)
    error = estimated @ np.linalg.inv(expected)
    assert rotation_angle_deg(error[:3, :3]) < 1.0
    assert np.linalg.norm(error[:3, 3]) < 0.02
    assert np.mean(weights[outliers]) < np.mean(np.delete(weights, outliers))


def test_adaptive_trim_chooses_small_subset_when_residual_tail_is_large():
    distances = np.concatenate((np.linspace(0.001, 0.004, 30), np.linspace(0.03, 0.08, 70)))
    config = RegistrationConfig(adaptive_trim=True, max_correspondence_distance=0.1,
                                trim_fraction_min=0.2, trim_fraction_max=0.95,
                                trim_fraction_step=0.05, min_correspondences=10)
    selected, fraction = _select_correspondences(distances, config)
    assert 0.2 <= fraction <= 0.4
    assert len(selected) == round(fraction * len(distances))


def test_fixed_trim_behavior_is_preserved():
    distances = np.linspace(0.001, 0.05, 100)
    config = RegistrationConfig(adaptive_trim=False, trim_fraction=0.8,
                                max_correspondence_distance=0.1, min_correspondences=10)
    selected, fraction = _select_correspondences(distances, config)
    assert len(selected) == 80
    assert np.isclose(fraction, 0.8)


class FeatureStub:
    def __init__(self, descriptors):
        self.data = np.asarray(descriptors, dtype=float).T


def test_relaxed_fpfh_uses_top_k_reciprocity_and_ratio_test():
    source = FeatureStub([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
    target = FeatureStub([[0.05, 0.0], [0.9, 1.0], [1.1, 1.0], [4.9, 5.0]])
    matches, distances, ratios = relaxed_fpfh_correspondences(source, target, top_k=3, ratio_threshold=.8)
    assert matches.shape[1] == 2
    assert (0, 0) in map(tuple, matches)
    assert (2, 3) in map(tuple, matches)
    assert np.all(ratios <= .8)
    assert np.all(distances >= 0)


def test_candidate_deduplication_uses_pose_distance():
    first = CoarseCandidate(np.eye(4), "first")
    near = CoarseCandidate(make_transform(rotation_z(np.radians(.1)), np.array([1e-4, 0, 0])), "near")
    far = CoarseCandidate(make_transform(rotation_z(np.radians(5)), np.zeros(3)), "far")
    unique = deduplicate_candidates([first, near, far], rotation_threshold_deg=.5, translation_threshold=.001)
    assert [candidate.origin for candidate in unique] == ["first", "far"]


def test_full_cloud_verification_beats_larger_local_consensus():
    rng = np.random.default_rng(104)
    source = rng.uniform(-.4, .4, size=(500, 3))
    expected = make_transform(rotation_z(.3), np.array([.08, -.03, .02]))
    target = apply_transform(source, expected)
    wrong = make_transform(rotation_z(-1.2), np.array([.2, .1, 0]))
    candidates = [CoarseCandidate(wrong, "wrong_consensus", local_inliers=100),
                  CoarseCandidate(expected, "true_full_cloud", local_inliers=20)]
    config = RegistrationConfig(validation_icp_iterations=0, max_correspondence_distance=.02)
    transform, ranked = verify_coarse_candidates(source, target, candidates, config)
    assert ranked[0].origin == "true_full_cloud"
    assert ranked[0].verified_fitness > ranked[1].verified_fitness
    assert np.allclose(transform, expected)
    fitness, rmse = symmetric_full_cloud_score(source, target, expected, .02)
    assert np.isclose(fitness, 1.0) and rmse < 1e-10
