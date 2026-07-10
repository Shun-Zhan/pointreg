import numpy as np

from pointreg.metrics import pose_errors, symmetric_overlap
from pointreg.models import RegistrationConfig
from pointreg.pipeline import register_pair
from pointreg.transforms import apply_transform, make_transform


def _rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_low_overlap_pair_recovers_from_rough_fpfh_initial(monkeypatch):
    rng = np.random.default_rng(24)
    shared = rng.uniform(-0.12, 0.12, size=(180, 3))
    source_only = rng.uniform([0.4, -0.2, -0.1], [0.65, 0.2, 0.1], size=(320, 3))
    source = np.vstack([shared, source_only])

    expected = make_transform(_rotation_z(0.04), np.array([0.035, -0.025, 0.018]))
    target_shared = apply_transform(shared, expected)
    target_only = rng.uniform([-0.7, -0.25, -0.1], [-0.45, 0.25, 0.1], size=(320, 3))
    target = np.vstack([target_shared, target_only])
    assert symmetric_overlap(source, target, expected, 0.01) < 0.5

    rough_initial = make_transform(_rotation_z(0.04), np.array([0.055, -0.025, 0.018]))

    def rough_fpfh(source_points, target_points, voxel_size, seed):
        return rough_initial

    monkeypatch.setattr("pointreg.pipeline.fpfh_registration", rough_fpfh)
    config = RegistrationConfig(
        coarse_method="fpfh",
        fine_method="custom_icp",
        voxel_size=0.0,
        max_correspondence_distance=0.01,
        trim_fraction=0.8,
        max_iterations=80,
        min_correspondences=20,
        rmse_tolerance=1e-10,
        transform_tolerance=1e-9,
        enable_pose_hint=False,
    )

    result = register_pair(source, target, config, ground_truth=expected)

    rotation_error, translation_error = pose_errors(result.transformation, expected)
    assert result.status != "failed"
    assert result.success
    assert result.metrics["fitness"] < 0.5
    assert result.metrics["correspondences"] >= config.min_correspondences
    assert rotation_error < 1.0
    assert translation_error < 0.005
    assert "multiscale" in result.message
