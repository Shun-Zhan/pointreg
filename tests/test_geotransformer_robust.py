import sys
import types

import numpy as np
import pytest

from finetune_kit.evaluate_3dmatch import _evaluation_summary, refine_and_select_candidates
from pointreg.coarse import gcransac_from_correspondences
from pointreg.geotransformer import _decode_3dmatch_output
from pointreg.transforms import apply_transform, make_transform


def rotation_z(angle: float) -> np.ndarray:
    """构造绕 z 轴旋转 angle 弧度的 3x3 旋转矩阵,供各测试造真值用。"""
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def test_decode_3dmatch_output_restores_scale_and_direction():
    """验证解码 3DMatch 网络输出:点坐标除以尺度还原、分数归一化、位姿平移同步缩小。"""
    estimate = make_transform(rotation_z(0.2), np.array([0.3, -0.2, 0.1]))
    decoded = _decode_3dmatch_output(
        {
            "src_corr_points": np.array([[10.0, 20.0, 30.0], [20.0, 30.0, 40.0]]),
            "ref_corr_points": np.array([[40.0, 50.0, 60.0], [50.0, 60.0, 70.0]]),
            "corr_scores": np.array([2.0, 1.0]),
            "estimated_transform": estimate,
        },
        10.0,
    )
    assert np.allclose(decoded.source_points[0], [1.0, 2.0, 3.0])   # 坐标已按 scale=10 还原
    assert np.allclose(decoded.target_points[0], [4.0, 5.0, 6.0])
    assert np.allclose(decoded.scores, [1.0, 0.5])                  # 分数按最大值归一化
    assert np.allclose(decoded.lgr_transform[:3, :3], estimate[:3, :3])   # 旋转不受尺度影响
    assert np.allclose(decoded.lgr_transform[:3, 3], estimate[:3, 3] / 10.0)  # 平移按尺度缩小


def test_gcransac_layout_probability_normalization_and_transpose(monkeypatch):
    """验证喂给 pygcransac 的数据布局正确:对应点拼接、概率归一化、阈值传参、返回矩阵转置。"""
    captured = {}
    row_model = np.arange(16, dtype=float).reshape(4, 4)

    def find_rigid_transform(correspondences, probabilities, **kwargs):
        captured["correspondences"] = correspondences
        captured["probabilities"] = probabilities
        captured["kwargs"] = kwargs
        return row_model, np.ones(len(correspondences), dtype=bool)

    monkeypatch.setitem(sys.modules, "pygcransac", types.SimpleNamespace(findRigidTransform=find_rigid_transform))
    source = np.arange(12, dtype=float).reshape(4, 3)
    target = source + 100.0
    transform, inliers = gcransac_from_correspondences(source, target, np.array([0.0, 1.0, 2.0, 4.0]))
    assert np.array_equal(captured["correspondences"][:, :3], source)   # 前三列是源点
    assert np.array_equal(captured["correspondences"][:, 3:], target)   # 后三列是目标点
    assert np.allclose(captured["probabilities"], [0.0, 0.25, 0.5, 1.0])  # 分数按最大值归一化为概率
    assert captured["kwargs"]["threshold"] == 0.01
    assert captured["kwargs"]["max_iters"] == 10000
    assert np.array_equal(transform, row_model.T)   # pygcransac 返回行主序,需转置回列向量约定
    assert inliers.all()


def test_gcransac_rejects_too_few_or_invalid_correspondences():
    """验证输入非法时及时报错:对应点不足三对、或包含非有限值(NaN)。"""
    with pytest.raises(ValueError, match="at least three"):
        gcransac_from_correspondences(np.zeros((2, 3)), np.zeros((2, 3)))  # 仅 2 对,不足
    with pytest.raises(ValueError, match="finite"):
        source = np.zeros((3, 3))
        source[0, 0] = np.nan       # 含 NaN,应被拒绝
        gcransac_from_correspondences(source, np.zeros((3, 3)))


def test_gcransac_recovers_transform_with_many_low_probability_outliers():
    """验证在大量低概率外点中,GC-RANSAC 仍能恢复出正确的刚体变换并找回内点。"""
    pytest.importorskip("pygcransac")  # 未装 pygcransac 则跳过
    rng = np.random.default_rng(12)
    source_inliers = rng.uniform(-0.2, 0.2, size=(50, 3))
    expected = make_transform(rotation_z(0.35), np.array([0.04, -0.025, 0.012]))
    target_inliers = apply_transform(source_inliers, expected)
    source_outliers = rng.uniform(-0.5, 0.5, size=(200, 3))
    target_outliers = rng.uniform(-0.5, 0.5, size=(200, 3))
    source = np.vstack([source_inliers, source_outliers])
    target = np.vstack([target_inliers, target_outliers])
    probabilities = np.r_[np.ones(50), np.full(200, 0.001)]
    estimated, inliers = gcransac_from_correspondences(
        source,
        target,
        probabilities,
        correspondence_distance=0.003,
        max_iters=20000,
    )
    assert int(inliers.sum()) >= 45              # 50 个真内点里至少找回 45 个
    assert np.allclose(estimated, expected, atol=1e-4)   # 恢复的变换逼近真值


def test_candidate_selection_uses_geometry_without_ground_truth(monkeypatch):
    """验证候选择优只靠几何自洽:在"正确"与"错误"两个候选中,选中几何上真正对齐的那个。"""
    rng = np.random.default_rng(31)
    source = rng.uniform(-0.15, 0.15, size=(300, 3))
    correct = make_transform(rotation_z(0.08), np.array([0.012, -0.006, 0.003]))
    target = apply_transform(source, correct)
    wrong = make_transform(rotation_z(1.2), np.array([0.2, 0.2, -0.1]))

    def no_refinement(source_points, target_points, initial, *, voxel_size):
        return np.asarray(initial).copy(), []   # 打桩:跳过 ICP 精配,直接用初值,隔离出"选择"逻辑

    monkeypatch.setattr("finetune_kit.evaluate_3dmatch._two_stage_icp", no_refinement)
    name, transform, rows = refine_and_select_candidates(
        source,
        target,
        {"wrong": wrong, "correct": correct},
        voxel_size=0.0025,
        score_distance=0.01,
    )
    assert name == "correct"                 # 应选中真正对齐的候选
    assert np.allclose(transform, correct)
    assert len(rows) == 2                    # 两个候选都留有记录


def test_evaluation_summary_tracks_strict_relaxed_and_errors():
    """验证评测汇总:正确统计完成数、错误数、严格/放宽档成功数与成功率、失败清单。"""
    rows = [
        {"pair": "a->b", "status": "ok", "success_2pct": True, "success_3pct": True},
        {"pair": "b->a", "status": "ok", "success_2pct": False, "success_3pct": True},
        {"pair": "a->c", "status": "error", "error": "boom"},
    ]
    summary = _evaluation_summary(rows, requested_pairs=6)
    assert summary["completed_pairs"] == 2       # 两对成功跑完
    assert summary["error_pairs"] == 1           # 一对出错
    assert summary["success_2pct"] == 1          # 2% 档成功 1 对
    assert summary["success_3pct"] == 2          # 3% 档成功 2 对
    assert summary["success_rate_2pct_completed"] == 0.5   # 2% 成功率 = 1/2
    assert summary["failed_pairs"] == ["b->a"]   # 未达 2% 的对被列入失败清单
