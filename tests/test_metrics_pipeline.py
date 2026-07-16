import numpy as np
import pytest

from pointreg.metrics import pose_errors, symmetric_overlap
from pointreg.models import RegistrationConfig
from pointreg.pipeline import register_pair
from pointreg.transforms import make_transform


def test_config_validation():
    """验证配置校验:非法参数(trim_fraction=0)在 validate() 时应抛 ValueError。"""
    with pytest.raises(ValueError):
        RegistrationConfig(trim_fraction=0).validate()


def test_pose_and_overlap_metrics():
    """验证度量的边界情形:同一位姿误差为 0,点云与自身的重合率为 1。"""
    rng = np.random.default_rng(2)
    points = rng.normal(size=(200, 3))
    identity = np.eye(4)
    assert pose_errors(identity, identity) == (0.0, 0.0)             # 无差异 -> 零误差
    assert symmetric_overlap(points, points, identity, 1e-9) == 1.0  # 完全重合 -> 重合率 1


def test_pipeline_returns_failure_instead_of_crashing():
    """验证退化输入(空源点云)时流水线优雅返回 failed,而不是抛异常崩溃。"""
    result = register_pair(np.empty((0, 3)), np.ones((3, 3)), RegistrationConfig(coarse_method="none"))
    assert result.status == "failed"
    assert not result.success

