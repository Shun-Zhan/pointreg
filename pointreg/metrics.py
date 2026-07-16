"""配准评价指标：位姿误差、对齐质量（RMSE/fitness）以及两帧真实重叠率。

这些指标分两类：
1）有真值时的位姿误差（旋转/平移误差），衡量“估计位姿离真值有多远”；
2）无真值也能算的对齐质量（RMSE、fitness、对应点数），衡量“对齐后两片点云贴合得多好”。
"""
from __future__ import annotations

import numpy as np

from .nearest import nearest_neighbors
from .transforms import apply_transform, invert_transform, rotation_angle_deg


def pose_errors(estimated: np.ndarray, ground_truth: np.ndarray) -> tuple[float, float]:
    """比较估计位姿与真值位姿，返回 (旋转误差°, 平移误差)。

    做法是求相对偏差 delta = 真值^-1 · 估计：若两者完全一致则 delta 为单位阵。
    旋转误差取 delta 旋转块的旋转角，平移误差取 delta 平移向量的模长。
    """
    delta = invert_transform(ground_truth) @ estimated
    return rotation_angle_deg(delta[:3, :3]), float(np.linalg.norm(delta[:3, 3]))


def alignment_metrics(source: np.ndarray, target: np.ndarray, transform: np.ndarray, threshold: float) -> dict[str, float]:
    """在给定变换下评估源、目标点云的对齐质量（不依赖真值）。

    参数:
        source/target: 源、目标点云 (N, 3)。
        transform: 待评估的 source->target 变换。
        threshold: 内点距离阈值，小于等于它的对应关系才算“有效对齐”。
    返回:
        字典包含：
        - rmse: 有效对应点的均方根距离（越小越好；无有效点时为 inf）。
        - fitness: 有效对应点占源点总数的比例（越大越好，反映覆盖程度）。
        - correspondences: 有效对应点的绝对数量。
    """
    moved = apply_transform(source, transform)  # 先把源点用待评估变换搬到目标坐标系。
    distances, _ = nearest_neighbors(moved, target)  # 每个源点到目标点云的最近距离。
    valid = distances <= threshold  # 距离在阈值内的视为有效对应（内点）。
    return {
        # RMSE 只统计内点，避免离群的非重叠点把误差拉爆。
        "rmse": float(np.sqrt(np.mean(distances[valid] ** 2))) if np.any(valid) else float("inf"),
        "fitness": float(np.mean(valid)) if len(valid) else 0.0,
        "correspondences": float(np.count_nonzero(valid)),
    }


def symmetric_overlap(source: np.ndarray, target: np.ndarray, ground_truth: np.ndarray, threshold: float) -> float:
    """在真值位姿下估计两帧点云的“对称重叠率”，用于判断这对本身难不难配。

    单向重叠会因两片点数、密度不同而有偏，故正反向各算一次再取平均：
    - forward：源点对齐后有多少比例能在目标中找到近邻；
    - backward：目标点有多少比例能在对齐后的源中找到近邻。
    返回 [0, 1] 之间的重叠率，越大说明两帧共视区域越多、越好配。
    """
    forward, _ = nearest_neighbors(apply_transform(source, ground_truth), target)
    backward, _ = nearest_neighbors(target, apply_transform(source, ground_truth))
    a = float(np.mean(forward <= threshold)) if len(forward) else 0.0
    b = float(np.mean(backward <= threshold)) if len(backward) else 0.0
    return (a + b) / 2.0

