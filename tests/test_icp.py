import numpy as np
import pointreg.icp as icp_module

from pointreg.icp import custom_icp, solve_rigid_svd
from pointreg.models import RegistrationConfig
from pointreg.nearest import NearestNeighborIndex, nearest_neighbors
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


def test_reusable_nearest_index_matches_one_shot_queries():
    rng = np.random.default_rng(17)
    reference = rng.normal(size=(200, 3))
    first_query = rng.normal(size=(30, 3))
    second_query = rng.normal(size=(40, 3))
    index = NearestNeighborIndex(reference)
    tree_identity = id(index._tree)
    for query in (first_query, second_query):
        expected_distances, expected_indices = nearest_neighbors(query, reference)
        distances, indices = index.query(query)
        assert np.allclose(distances, expected_distances)
        assert np.array_equal(indices, expected_indices)
        assert id(index._tree) == tree_identity


def test_icp_builds_target_index_only_once(monkeypatch):
    builds = 0
    queries = 0
    original_index = icp_module.NearestNeighborIndex

    class CountingIndex:
        def __init__(self, reference):
            nonlocal builds
            builds += 1
            self.index = original_index(reference)

        def query(self, points):
            nonlocal queries
            queries += 1
            return self.index.query(points)

    monkeypatch.setattr(icp_module, "NearestNeighborIndex", CountingIndex)
    rng = np.random.default_rng(23)
    source = rng.uniform(-.1, .1, size=(200, 3))
    target = source + np.array([.01, -.005, .002])
    config = RegistrationConfig(coarse_method="none", max_correspondence_distance=.05,
                                trim_fraction=1, max_iterations=8, rmse_tolerance=0,
                                transform_tolerance=0, min_correspondences=20)
    _, history, _, _ = custom_icp(source, target, np.eye(4), config)
    assert builds == 1
    assert queries == len(history) == config.max_iterations
