import numpy as np

from pointreg.io import parse_bun_conf
from pointreg.transforms import apply_transform, invert_transform, make_transform, quaternion_xyzw_to_matrix, relative_transform


def test_quaternion_identity_and_inverse():
    """验证四元数转旋转矩阵:单位四元数得单位阵,且变换与其逆相乘为单位阵。"""
    rotation = quaternion_xyzw_to_matrix(np.array([0, 0, 0, 1.0]))
    transform = make_transform(rotation, np.array([1, 2, 3.0]))
    assert np.allclose(rotation, np.eye(3))                         # 单位四元数 -> 单位旋转
    assert np.allclose(invert_transform(transform) @ transform, np.eye(4))  # 逆变换正确


def test_relative_transform_direction():
    """验证相对变换的方向约定:把源坐标系原点映射到目标坐标系下的正确位置。"""
    source_world = make_transform(translation=np.array([2, 0, 0]))
    target_world = make_transform(translation=np.array([1, 0, 0]))
    point_target = apply_transform(np.array([[0., 0, 0]]), relative_transform(source_world, target_world))
    assert np.allclose(point_target, [[1, 0, 0]])   # 源原点在目标系下应位于 (1,0,0)


def test_parse_bunny_conf():
    """验证解析斯坦福 bun.conf:含预期扫描名,基准帧为单位阵,旋转矩阵合法(det=1)。"""
    poses = parse_bun_conf("bunny/data/bun.conf")
    assert "bun000" in poses and "bun045" in poses
    assert np.allclose(poses["bun000"], np.eye(4))                       # bun000 是基准帧
    assert np.isclose(np.linalg.det(poses["bun045"][:3, :3]), 1.0)      # 旋转矩阵合法
    # Stanford's legacy quaternion convention makes the y rotation positive
    # after conversion to our active column-vector convention.
    assert poses["bun045"][0, 2] > 0
