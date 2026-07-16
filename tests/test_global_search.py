import numpy as np

from pointreg.global_search import (
    GlobalRegistrationSearch,
    SignedCorrelationSolver,
    super_fibonacci_rotations,
)
from pointreg.models import RegistrationConfig
from pointreg.scoring import _DepthMap


def test_super_fibonacci_rotations_are_deterministic_and_valid():
    """验证超级斐波那契旋转采样:结果可复现,且每个都是合法旋转矩阵(正交且 det=1)。"""
    first = super_fibonacci_rotations(128)
    second = super_fibonacci_rotations(128)
    assert np.array_equal(first, second)                                       # 确定性:两次结果完全一致
    assert np.allclose(first @ np.swapaxes(first, 1, 2), np.eye(3), atol=1e-12) # R·Rᵀ=I,正交
    assert np.allclose(np.linalg.det(first), 1.0, atol=1e-12)                   # det=1,是旋转而非反射


def test_signed_fft_recovers_synthetic_translation_within_one_voxel():
    """验证基于 FFT 的有符号相关求解器:在给定旋转下能恢复平移,误差在一个体素内。"""
    rng = np.random.default_rng(4)
    source = rng.uniform(-0.025, 0.025, size=(300, 3))
    translation = np.array([0.010, -0.005, 0.005])
    target = source + translation
    solver = SignedCorrelationSolver(source, target, voxel=0.005, dims=32, free_penalty=0.0)
    estimated = solver.best_transform(np.eye(3))
    assert np.allclose(estimated[:3, :3], np.eye(3))               # 旋转应保持为单位阵
    assert np.linalg.norm(estimated[:3, 3] - translation) <= 0.0051  # 平移误差在一个体素(0.005)内


def test_depth_map_penalizes_points_inside_observed_free_space():
    """验证自由空间深度图:落在已观测表面上不违规(0),被"推到"更近的自由空间则全违规(1)。"""
    xy = np.array([[x, y] for x in (-0.1, 0.0, 0.1) for y in (-0.1, 0.0, 0.1)])
    surface = np.c_[xy, np.ones(len(xy))]
    depth_map = _DepthMap.build(surface, np.zeros(3), bin_size=0.01)
    assert depth_map.violation_ratio(surface, margin=0.01) == 0.0        # 点在表面上,不侵入自由空间
    assert depth_map.violation_ratio(surface * 0.5, margin=0.01) == 1.0  # 点被拉近到相机一侧,全部违规


def test_external_candidate_deduplication_uses_rotation_and_translation_thresholds():
    """验证外部候选去重:平移差异极小的近似候选被合并,差异够大的才保留。"""
    search = object.__new__(GlobalRegistrationSearch)  # 绕过 __init__,只测 _deduplicate 方法
    identity = np.eye(4)
    near = np.eye(4)
    near[0, 3] = 0.00005    # 与 identity 几乎重合,应被去重
    far = np.eye(4)
    far[0, 3] = 0.0002      # 差异够大,应保留
    unique = search._deduplicate([("identity", identity), ("near", near), ("far", far)])
    assert [label for label, _ in unique] == ["identity", "far"]  # near 被合并掉


def test_borda_selection_applies_free_space_gate(monkeypatch):
    """验证候选选择的自由空间门控:侵入自由空间过多的"高分"候选会被门掉,让安全候选胜出。"""
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
    assert gate                                              # 门控确实生效
    assert selected.label == "safe"                          # 最终选中安全候选,而非高分但违规的
    assert next(row for row in rows if row.label == "safe").gated       # safe 通过了门控
    assert not next(row for row in rows if row.label == "unsafe").gated # unsafe 未通过门控


def test_reduced_full_search_accepts_external_truth_candidate():
    """验证缩减版全局搜索:接受外部传入的真值候选,并给出高 fitness 的有效对齐结果。"""
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
    assert result.transform.shape == (4, 4)                 # 返回合法的 4x4 齐次矩阵
    assert np.isfinite(result.transform).all()              # 无 NaN/Inf
    assert result.fitness > 0.9                             # 源=目标,几乎完全对齐
    assert any(candidate.label.startswith("external:truth") for candidate in result.candidates)  # 外部候选被纳入
    assert result.timings_ms["total"] > 0                   # 记录了总耗时
