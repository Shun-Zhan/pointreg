"""自研的 Trimmed 点到点 ICP（迭代最近点）精配准。

标准 ICP 每轮做两件事：(1) 为源点找目标点云中的最近邻作为对应；
(2) 用 SVD 闭式求解使对应点对齐的最优刚体变换。为了应对部分重合/离群，
本实现加入了距离阈值 + Trimmed 截断，只用最可信的一部分对应点求解。
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from .models import ICPRecord, RegistrationConfig
from .nearest import NearestNeighborIndex
from .transforms import apply_transform, make_transform, rotation_angle_deg


def solve_rigid_svd(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """用 Kabsch/SVD 算法求解一对已配对点集之间的最优刚体变换。

    经典闭式解：先各自去质心，构造协方差矩阵 H = Σ(sᵢ-s̄)(tᵢ-t̄)ᵀ，
    对 H 做 SVD 得到旋转；再由质心关系推出平移。若得到的旋转行列式为负
    （出现镜像反射），翻转最后一个奇异向量的符号修正为真正的旋转。
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("source and target must be matching (N, 3) arrays with N >= 3")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)  # 3×3 协方差矩阵
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:  # 修正镜像反射，保证是纯旋转
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return make_transform(rotation, translation)


def _select_correspondences(distances: np.ndarray, config: RegistrationConfig) -> np.ndarray:
    """从最近邻结果中筛选参与本轮求解的对应点索引（阈值 + Trimmed 截断）。

    先剔除距离超过阈值的对应（应对部分重合造成的错误配对）；若剩余数量足够
    且 trim_fraction<1，再按距离升序只保留最近的一部分，进一步排除噪声对应，
    这是 Trimmed-ICP 提升鲁棒性的关键。
    """
    valid_indices = np.flatnonzero(distances <= config.max_correspondence_distance)
    if len(valid_indices) >= config.min_correspondences and config.trim_fraction < 1:
        keep = max(config.min_correspondences, int(len(valid_indices) * config.trim_fraction))
        # argsort 取距离最小的前 keep 个，保留最可信的对应
        valid_indices = valid_indices[np.argsort(distances[valid_indices])[:keep]]
    return valid_indices


def custom_icp(source: np.ndarray, target: np.ndarray, initial: np.ndarray, config: RegistrationConfig) -> tuple[np.ndarray, list[ICPRecord], str, str]:
    """从初值 initial 出发迭代优化位姿，返回 (变换, 逐轮历史, 状态, 说明)。

    每轮流程：建立对应 → SVD 求增量 delta → 累积到当前变换 → 重算对应；
    当 RMSE 变化和位姿增量都小于容差时判定收敛，否则跑满最大迭代次数。
    目标点云的 KD-tree 只建一次并全程复用（见 NearestNeighborIndex）。
    """
    if len(source) < 3 or len(target) < 3:
        return initial.copy(), [], "failed", "point cloud is empty or too small"
    transform = initial.copy()
    history: list[ICPRecord] = []
    previous_rmse = float("inf")
    target_index = NearestNeighborIndex(target)  # 目标点云索引建一次、反复查询
    moved = apply_transform(source, transform)    # 用初值先变换一次源点云
    distances, indices = target_index.query(moved)
    valid_indices = _select_correspondences(distances, config)
    for iteration in range(1, config.max_iterations + 1):
        started = perf_counter()
        if len(valid_indices) < config.min_correspondences:
            # 有效对应点太少，无法稳定求解，判失败
            return transform, history, "failed", f"only {len(valid_indices)} valid correspondences"
        # 用当前对应点对求本轮位姿增量，并左乘累积到总变换上
        delta = solve_rigid_svd(moved[valid_indices], target[indices[valid_indices]])
        transform = delta @ transform
        # 用更新后的位姿重新变换源点云并重建对应关系
        moved = apply_transform(source, transform)
        distances, indices = target_index.query(moved)
        valid_indices = _select_correspondences(distances, config)
        rmse = float(np.sqrt(np.mean(distances[valid_indices] ** 2))) if len(valid_indices) else float("inf")
        rotation_delta = rotation_angle_deg(delta[:3, :3])
        translation_delta = float(np.linalg.norm(delta[:3, 3]))
        history.append(ICPRecord(iteration, rmse, len(valid_indices), rotation_delta, translation_delta, (perf_counter() - started) * 1000))
        # 收敛判据：RMSE 几乎不再下降，且本轮位姿增量足够小
        rmse_change = abs(previous_rmse - rmse)
        if rmse_change < config.rmse_tolerance and max(np.radians(rotation_delta), translation_delta) < config.transform_tolerance:
            return transform, history, "converged", "convergence tolerances reached"
        previous_rmse = rmse
    return transform, history, "max_iterations", "maximum iterations reached"
