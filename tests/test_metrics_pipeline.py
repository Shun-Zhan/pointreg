import numpy as np
import pytest

from pointreg.metrics import alignment_metrics, pose_errors, symmetric_alignment_metrics, symmetric_overlap
from pointreg.models import RegistrationConfig
from pointreg.pipeline import register_pair
from pointreg.transforms import make_transform


def test_config_validation():
    with pytest.raises(ValueError):
        RegistrationConfig(trim_fraction=0).validate()


def test_pose_and_overlap_metrics():
    rng = np.random.default_rng(2)
    points = rng.normal(size=(200, 3))
    identity = np.eye(4)
    assert pose_errors(identity, identity) == (0.0, 0.0)
    assert symmetric_overlap(points, points, identity, 1e-9) == 1.0


def test_symmetric_alignment_metrics():
    rng = np.random.default_rng(3)
    points = rng.normal(size=(120, 3))
    identity = np.eye(4)
    metrics = symmetric_alignment_metrics(points, points, identity, 1e-6)
    assert metrics["symmetric_fitness"] == 1.0
    assert metrics["fitness"] == 1.0


def test_pipeline_returns_failure_instead_of_crashing():
    result = register_pair(np.empty((0, 3)), np.ones((3, 3)), RegistrationConfig(coarse_method="none"))
    assert result.status == "failed"
    assert not result.success

