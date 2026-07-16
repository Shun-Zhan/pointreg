import numpy as np
import pointreg.icp as icp_module

from pointreg.icp import custom_icp, solve_rigid_svd
from pointreg.models import RegistrationConfig
from pointreg.nearest import NearestNeighborIndex, nearest_neighbors
from pointreg.transforms import apply_transform, make_transform


def rotation_z(angle):
    """构造绕 z 轴旋转的 3x3 旋转矩阵,供各测试造真值用。"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]])


def test_svd_recovers_rigid_transform():
    """验证 SVD 闭式解:已知完美对应时能精确恢复刚体变换,且旋转矩阵行列式=1(无镜像)。"""
    rng = np.random.default_rng(4)
    source = rng.normal(size=(100, 3))
    expected = make_transform(rotation_z(.25), np.array([.2, -.1, .3]))
    estimated = solve_rigid_svd(source, apply_transform(source, expected))
    assert np.allclose(estimated, expected, atol=1e-10)         # 精确复原变换
    assert np.linalg.det(estimated[:3, :3]) > .999999           # 行列式≈1,是纯旋转而非反射


def test_icp_converges_on_synthetic_cloud():
    """验证 ICP 在合成点云上能从单位阵初值收敛到真值,且 RMSE 逐步下降。"""
    rng = np.random.default_rng(8)
    source = rng.uniform(-.2, .2, size=(500, 3))
    expected = make_transform(rotation_z(.08), np.array([.015, -.01, .006]))
    target = apply_transform(source, expected)
    config = RegistrationConfig(coarse_method="none", max_correspondence_distance=.08, trim_fraction=1,
                                max_iterations=80, min_correspondences=20, rmse_tolerance=1e-10, transform_tolerance=1e-9)
    estimated, history, status, _ = custom_icp(source, target, np.eye(4), config)
    assert status in {"converged", "max_iterations"}
    assert history[-1].rmse < history[0].rmse   # RMSE 随迭代下降,说明确实在收敛
    assert np.allclose(estimated, expected, atol=1e-5)


def test_icp_safe_failure_without_correspondences():
    """验证两片点云相距过远、找不到对应时,ICP 能安全失败(返回 failed 而非崩溃)。"""
    source = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=float)
    target = source + 100    # 平移 100,远超对应距离阈值,不可能有对应
    config = RegistrationConfig(max_correspondence_distance=.1, min_correspondences=3)
    _, history, status, message = custom_icp(source, target, np.eye(4), config)
    assert status == "failed" and not history          # 失败且无迭代历史
    assert "correspondences" in message                # 错误信息指明是对应不足


def test_reusable_nearest_index_matches_one_shot_queries():
    """验证可复用的最近邻索引:多次查询结果与一次性查询一致,且底层 KD 树只建一次。"""
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
        assert id(index._tree) == tree_identity     # 树对象始终是同一个,未被重建


def test_icp_builds_target_index_only_once(monkeypatch):
    """验证 ICP 只对目标点云建一次索引:build 计数=1,query 次数=迭代数+1(含末次评估)。"""
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
    assert builds == 1                              # 索引只建一次(性能关键)
    assert len(history) == config.max_iterations
    assert queries == len(history) + 1              # 每轮一次 + 最后再评估一次


def test_history_rmse_matches_updated_iteration_transform():
    """验证历史里记录的 RMSE 与"用更新后变换"重算的值一致(确认记录的是迭代后的状态)。"""
    rng = np.random.default_rng(29)
    source = rng.uniform(-.15, .15, size=(300, 3))
    target = apply_transform(source, make_transform(rotation_z(.06), np.array([.01, -.008, .004])))
    config = RegistrationConfig(coarse_method="none", max_correspondence_distance=.08,
                                trim_fraction=.8, max_iterations=1, min_correspondences=20)
    transform, history, status, _ = custom_icp(source, target, np.eye(4), config)
    distances, _ = nearest_neighbors(apply_transform(source, transform), target)
    valid = np.flatnonzero(distances <= config.max_correspondence_distance)
    keep = max(config.min_correspondences, int(len(valid) * config.trim_fraction))
    valid = valid[np.argsort(distances[valid])[:keep]]
    expected_rmse = float(np.sqrt(np.mean(distances[valid] ** 2)))
    assert status == "max_iterations"
    assert len(history) == 1
    assert np.isclose(history[0].rmse, expected_rmse)      # RMSE 对应更新后的变换
    assert history[0].correspondences == len(valid)       # 记录的对应数与实际保留数一致
