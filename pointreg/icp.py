"""精配准（fine registration）之自研 ICP 模块。

在粗配准给出的初值基础上，用迭代最近点（ICP, Iterative Closest Point）不断
细化位姿：每轮“找最近邻对应 -> 用 SVD 解最优刚体变换 -> 更新位姿”，
直到 RMSE 与位姿增量都足够小为止。为应对低重叠，加入了对应点裁剪（trimming）
和自适应裁剪策略，抑制外点影响。
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from .models import ICPRecord, RegistrationConfig
from .nearest import NearestNeighborIndex
from .transforms import apply_transform, make_transform, rotation_angle_deg


def solve_rigid_svd(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """用 SVD 求两组一一对应点之间的最优刚体变换（Kabsch/Umeyama 算法）。

    在最小二乘意义下，最优旋转由去中心化点集的协方差矩阵的 SVD 给出。
    这是 ICP 每轮迭代求解变换增量的核心闭式解。

    参数：
        source / target: 一一对应的两组点，形状均为 (N, 3)，N>=3。
    返回：
        4x4 齐次变换矩阵，把 source 对齐到 target。
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("source and target must be matching (N, 3) arrays with N >= 3")
    # 分别求质心并去中心化，旋转只与相对位置有关。
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    # 互协方差矩阵 H = 源^T · 目标。
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    # 最优旋转 R = V · U^T。
    rotation = vt.T @ u.T
    # 若行列式为负说明得到的是反射，需翻转 V 最后一列修正为纯旋转。
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    # 由旋转确定平移，使两质心对齐。
    translation = target_center - rotation @ source_center
    return make_transform(rotation, translation)


def _effective_trim_fraction(distances: np.ndarray, config: RegistrationConfig) -> float:
    """自适应地决定本轮保留多少比例的对应点（裁剪比例）。

    当当前位姿的几何支撑不足（内点比例低）时，收紧裁剪、只信任最近的一小部分点，
    避免大量外点把 SVD 解拉偏。内点比例较高时则用配置里的默认裁剪比例。
    """
    if not config.adaptive_trim or not len(distances):
        return config.trim_fraction
    # 内点比例：距离在阈值内的对应点占比。
    inlier_ratio = float(np.mean(distances <= config.max_correspondence_distance))
    if inlier_ratio >= 0.6:
        return config.trim_fraction
    # 支撑不足时，裁剪比例随内点比例线性收紧，并夹在 [min_trim_fraction, trim_fraction]。
    return float(np.clip(1.25 * inlier_ratio, config.min_trim_fraction, config.trim_fraction))


def _select_correspondences(distances: np.ndarray, config: RegistrationConfig) -> np.ndarray:
    """从所有对应中挑出参与本轮 SVD 求解的“可信对应点”下标。

    先按最大对应距离过滤，再按裁剪比例只保留距离最小的那部分（Trimmed ICP 思想），
    从而抑制外点。若有效点太少或无需裁剪则不裁。
    """
    # 先剔除距离超阈值的对应。
    valid_indices = np.flatnonzero(distances <= config.max_correspondence_distance)
    trim_fraction = _effective_trim_fraction(distances, config)
    if len(valid_indices) >= config.min_correspondences and trim_fraction < 1:
        # 保留数量 = max(下限, 有效点数 × 裁剪比例)。
        keep = max(config.min_correspondences, int(len(valid_indices) * trim_fraction))
        # 按距离升序取最近的 keep 个，优先信任距离最小的对应。
        valid_indices = valid_indices[np.argsort(distances[valid_indices])[:keep]]
    return valid_indices


def custom_icp(source: np.ndarray, target: np.ndarray, initial: np.ndarray, config: RegistrationConfig) -> tuple[np.ndarray, list[ICPRecord], str, str]:
    """自研的 point-to-point ICP 精配准主循环。

    在初值 initial 基础上迭代：变换源点 -> 查最近邻 -> 选可信对应 ->
    SVD 求增量 delta -> 更新位姿，直至收敛或达到最大迭代次数。

    参数：
        initial: 粗配准给出的初始 4x4 变换。
        config: 配准配置（含裁剪、阈值、迭代上限、收敛容差等）。
    返回：
        (最终变换, 每轮 ICPRecord 历史, 状态字符串, 说明信息)。
        状态可能为 "converged" / "max_iterations" / "failed"。
    """
    if len(source) < 3 or len(target) < 3:
        return initial.copy(), [], "failed", "point cloud is empty or too small"
    transform = initial.copy()
    history: list[ICPRecord] = []
    previous_rmse = float("inf")
    # 为目标点云建最近邻索引（KD 树），迭代中反复查询，只需建一次。
    target_index = NearestNeighborIndex(target)
    # 先用初值把源点云变换过去，做一次最近邻查询作为循环入口状态。
    moved = apply_transform(source, transform)
    distances, indices = target_index.query(moved)
    valid_indices = _select_correspondences(distances, config)
    for iteration in range(1, config.max_iterations + 1):
        started = perf_counter()
        # 可信对应点太少则无法稳定求解，提前判定失败。
        if len(valid_indices) < config.min_correspondences:
            return transform, history, "failed", f"only {len(valid_indices)} valid correspondences"
        # 用当前可信对应求本轮最优刚体增量 delta。
        delta = solve_rigid_svd(moved[valid_indices], target[indices[valid_indices]])
        # 左乘累积到总变换上（delta 作用在世界坐标系）。
        transform = delta @ transform
        # 用更新后的位姿重新变换源点并重新查最近邻，为下一轮做准备。
        moved = apply_transform(source, transform)
        distances, indices = target_index.query(moved)
        valid_indices = _select_correspondences(distances, config)
        # 本轮配准误差（可信对应上的 RMSE）与位姿增量大小，记入历史用于监控收敛。
        rmse = float(np.sqrt(np.mean(distances[valid_indices] ** 2))) if len(valid_indices) else float("inf")
        rotation_delta = rotation_angle_deg(delta[:3, :3])
        translation_delta = float(np.linalg.norm(delta[:3, 3]))
        history.append(ICPRecord(iteration, rmse, len(valid_indices), rotation_delta, translation_delta, (perf_counter() - started) * 1000))
        # 收敛判定：RMSE 变化足够小，且旋转/平移增量都足够小，即认为收敛。
        rmse_change = abs(previous_rmse - rmse)
        if rmse_change < config.rmse_tolerance and max(np.radians(rotation_delta), translation_delta) < config.transform_tolerance:
            return transform, history, "converged", "convergence tolerances reached"
        previous_rmse = rmse
    # 迭代用尽仍未满足容差，返回当前最好结果。
    return transform, history, "max_iterations", "maximum iterations reached"
