import numpy as np

from pointreg.icp import custom_icp, solve_rigid_svd
from pointreg.models import RegistrationConfig
from pointreg.transforms import apply_transform, make_transform


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]])


def test_svd_recovers_rigid_transform():
    rng = np.random.default_rng(4)
    source = rng.normal(size=(100, 3))
    expected = make_transform(rotation_z(.25), np.array([.2, -.1, .3]))
    estimated = solve_rigid_svd(source, apply_transform(source, expected))
    assert np.allclose(estimated, expected, atol=1e-10)
    assert np.linalg.det(estimated[:3, :3]) > .999999


def test_icp_converges_on_synthetic_cloud():
    rng = np.random.default_rng(8)
    source = rng.uniform(-.2, .2, size=(500, 3))
    expected = make_transform(rotation_z(.08), np.array([.015, -.01, .006]))
    target = apply_transform(source, expected)
    config = RegistrationConfig(coarse_method="none", max_correspondence_distance=.08, trim_fraction=1,
                                max_iterations=80, min_correspondences=20, rmse_tolerance=1e-10, transform_tolerance=1e-9)
    estimated, history, status, _ = custom_icp(source, target, np.eye(4), config)
    assert status in {"converged", "max_iterations"}
    assert history[-1].rmse < history[0].rmse
    assert np.allclose(estimated, expected, atol=1e-5)


def test_icp_safe_failure_without_correspondences():
    source = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=float)
    target = source + 100
    config = RegistrationConfig(max_correspondence_distance=.1, min_correspondences=3)
    _, history, status, message = custom_icp(source, target, np.eye(4), config)
    assert status == "failed" and not history
    assert "correspondences" in message

