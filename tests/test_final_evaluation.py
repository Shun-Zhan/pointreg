import numpy as np

from pointreg.experiments import FINAL_EVALUATION_COLUMNS, _ground_truth_inlier_count, _success_at
from pointreg.models import RegistrationResult


def test_final_evaluation_schema_contains_requested_fields():
    """验证最终评测表的列名包含课程要求的全部中文字段(点对、重合率、各类误差等)。"""
    requested = {
        "点对", "重合率", "Transformer对应数", "真值对应内点数", "GC-RANSAC内点数",
        "入选候选", "粗配旋转误差(°)", "最终旋转误差(°)", "fitness", "violation",
        "自由空间门控", "耗时(ms)", "成功_2%", "成功_3%", "成功_5%",
    }
    assert requested <= set(FINAL_EVALUATION_COLUMNS)  # 要求列是实际列的子集


def test_ground_truth_inliers_and_success_tiers():
    """验证真值内点计数,以及三档成功判据(2%/3%/5%)的边界判定是否正确。"""
    points = np.array([[0., 0., 0.], [1., 0., 0.], [3., 0., 0.]])
    target = np.array([[0., 0., 0.], [1.01, 0., 0.]])
    # 前两点在 0.02 阈值内各有近邻,第三点无,故内点数=2
    assert _ground_truth_inlier_count(points, target, np.eye(4), .02) == 2

    result = RegistrationResult(status="converged")
    result.metrics.update(rotation_error_deg=4.9, translation_error_ratio=.025)
    assert not _success_at(result, .02)  # 平移 2.5% > 2%,不达 2% 档
    assert _success_at(result, .03)      # 达到 3% 档
    assert _success_at(result, .05)      # 达到 5% 档
