"""配准质量评价指标：位姿误差、对齐指标、以及真值重合率。

其中 pose_errors 需要真值位姿，衡量“配得准不准”；
alignment_metrics 不需要真值，衡量“配完后两片贴合得好不好”。
"""

from __future__ import annotations

import numpy as np

from .nearest import nearest_neighbors
from .transforms import apply_transform, invert_transform, rotation_angle_deg


def pose_errors(estimated: np.ndarray, ground_truth: np.ndarray) -> tuple[float, float]:
    """对比估计位姿与真值，返回 (旋转误差(度), 平移误差)。

    做法：算出两者的相对变换 delta = GT⁻¹ · estimated，理想情况下 delta
    应为单位阵；delta 的旋转角即旋转误差，平移部分模长即平移误差。
    """
    delta = invert_transform(ground_truth) @ estimated
    return rotation_angle_deg(delta[:3, :3]), float(np.linalg.norm(delta[:3, 3]))


def alignment_metrics(source: np.ndarray, target: np.ndarray, transform: np.ndarray, threshold: float) -> dict[str, float]:
    """无需真值的对齐质量指标（RMSE / fitness / 对应点数）。

    把源点云按 transform 变换后，找它到目标点云的最近邻距离；
    距离在 threshold 内的算“对应上了”。
    - rmse：这些有效对应的均方根距离，越小越贴合
    - fitness：有效对应点占源点的比例，越大重合越多
    - correspondences：有效对应点的绝对数量
    """
    moved = apply_transform(source, transform)
    distances, _ = nearest_neighbors(moved, target)
    valid = distances <= threshold
    return {
        "rmse": float(np.sqrt(np.mean(distances[valid] ** 2))) if np.any(valid) else float("inf"),
        "fitness": float(np.mean(valid)) if len(valid) else 0.0,
        "correspondences": float(np.count_nonzero(valid)),
    }


def symmetric_overlap(source: np.ndarray, target: np.ndarray, ground_truth: np.ndarray, threshold: float) -> float:
    """用真值位姿计算两帧的对称重合率（仅用于实验展示，不参与求解）。

    单向重合率会因两片点数不同而不对称，这里取“源→目标”和“目标→源”
    两个方向的平均，得到更稳健的重合度量。低重合（<0.3）正是本课题难点。
    """
    forward, _ = nearest_neighbors(apply_transform(source, ground_truth), target)
    backward, _ = nearest_neighbors(target, apply_transform(source, ground_truth))
    a = float(np.mean(forward <= threshold)) if len(forward) else 0.0   # 源落在目标附近的比例
    b = float(np.mean(backward <= threshold)) if len(backward) else 0.0  # 目标落在源附近的比例
    return (a + b) / 2.0

