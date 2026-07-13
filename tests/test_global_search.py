import numpy as np

from pointreg.global_search import (
    GlobalRegistrationSearch,
    SignedCorrelationSolver,
    super_fibonacci_rotations,
)
from pointreg.models import RegistrationConfig
from pointreg.scoring import _DepthMap


def test_super_fibonacci_rotations_are_deterministic_and_valid():
    first = super_fibonacci_rotations(128)
    second = super_fibonacci_rotations(128)
    assert np.array_equal(first, second)
    assert np.allclose(first @ np.swapaxes(first, 1, 2), np.eye(3), atol=1e-12)
    assert np.allclose(np.linalg.det(first), 1.0, atol=1e-12)


def test_signed_fft_recovers_synthetic_translation_within_one_voxel():
    rng = np.random.default_rng(4)
    source = rng.uniform(-0.025, 0.025, size=(300, 3))
    translation = np.array([0.010, -0.005, 0.005])
    target = source + translation
    solver = SignedCorrelationSolver(source, target, voxel=0.005, dims=32, free_penalty=0.0)
    estimated = solver.best_transform(np.eye(3))
    assert np.allclose(estimated[:3, :3], np.eye(3))
    assert np.linalg.norm(estimated[:3, 3] - translation) <= 0.0051


def test_depth_map_penalizes_points_inside_observed_free_space():
    xy = np.array([[x, y] for x in (-0.1, 0.0, 0.1) for y in (-0.1, 0.0, 0.1)])
    surface = np.c_[xy, np.ones(len(xy))]
    depth_map = _DepthMap.build(surface, np.zeros(3), bin_size=0.01)
    assert depth_map.violation_ratio(surface, margin=0.01) == 0.0
    assert depth_map.violation_ratio(surface * 0.5, margin=0.01) == 1.0


def test_external_candidate_deduplication_uses_rotation_and_translation_thresholds():
    search = object.__new__(GlobalRegistrationSearch)
    identity = np.eye(4)
    near = np.eye(4)
    near[0, 3] = 0.00005
    far = np.eye(4)
    far[0, 3] = 0.0002
    unique = search._deduplicate([("identity", identity), ("near", near), ("far", far)])
    assert [label for label, _ in unique] == ["identity", "far"]


def test_borda_selection_applies_free_space_gate(monkeypatch):
    search = object.__new__(GlobalRegistrationSearch)
    search.config = RegistrationConfig(multi_violation_gate=0.016, multi_min_fitness=0.08)
    safe = np.eye(4)
    unsafe = np.eye(4)
    unsafe[0, 3] = 0.01

    def metrics(self, transform):
        if transform[0, 3] == 0.0:
            return 0.10, 0.0, 0.0, 0.004, 0.03
        return 0.90, 0.20, 0.20, 0.001, 0.80

    monkeypatch.setattr(GlobalRegistrationSearch, "_candidate_metrics", metrics)
    selected, rows, gate = search._select([("safe", safe), ("unsafe", unsafe)])
    assert gate
    assert selected.label == "safe"
    assert next(row for row in rows if row.label == "safe").gated
    assert not next(row for row in rows if row.label == "unsafe").gated


def test_reduced_full_search_accepts_external_truth_candidate():
    rng = np.random.default_rng(19)
    source = rng.normal(size=(180, 3)) * np.array([0.025, 0.018, 0.012])
    target = source.copy()
    config = RegistrationConfig(
        coarse_method="none",
        voxel_size=0.0025,
        max_correspondence_distance=0.01,
        adaptive_trim=True,
        multi_score_points=120,
        multi_grid_rotations=100,
        multi_seed_count=4,
        multi_climb_rounds=2,
        multi_refine_seeds=2,
        multi_refine_samples=1,
        multi_lock_span_deg=0.0,
        multi_lock_step_deg=1.0,
        multi_lock_top=1,
    )
    result = GlobalRegistrationSearch(source, target, config).run({"truth": np.eye(4)})
    assert result.transform.shape == (4, 4)
    assert np.isfinite(result.transform).all()
    assert result.fitness > 0.9
    assert any(candidate.label.startswith("external:truth") for candidate in result.candidates)
    assert result.timings_ms["total"] > 0
