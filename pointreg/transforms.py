"""位姿变换工具集：4x4 齐次变换矩阵的构造、应用、求逆与误差度量。

本项目所有位姿都用 4x4 齐次变换矩阵 T 表示，左上 3x3 是旋转 R、右上 3x1 是平移 t：
    T = [[R, t], [0, 1]]
对点 p 施加变换即 p' = R·p + t。这里集中提供相关的小工具函数。
"""
from __future__ import annotations

import numpy as np


def quaternion_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    """把 xyzw 顺序的四元数转换成 3x3 旋转矩阵。

    参数:
        q: 长度为 4 的数组，顺序为 (x, y, z, w)。函数内部会先做归一化，
           因此不要求输入是单位四元数。
    返回:
        对应的 3x3 旋转矩阵。
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        raise ValueError("quaternion must contain x, y, z, w")
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        # 接近零的四元数无法表示有效旋转（归一化会数值爆炸），直接报错。
        raise ValueError("zero quaternion is invalid")
    x, y, z, w = q / norm  # 归一化，保证得到的是正交旋转矩阵。
    # 标准的“四元数 -> 旋转矩阵”闭式公式。
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def make_transform(rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> np.ndarray:
    """由旋转和/或平移组装出一个 4x4 齐次变换矩阵。

    未提供的部分保持单位阵/零向量，缺省即返回单位变换（不改变点）。
    """
    transform = np.eye(4)
    if rotation is not None:
        transform[:3, :3] = np.asarray(rotation, dtype=float)  # 填入旋转块 R。
    if translation is not None:
        transform[:3, 3] = np.asarray(translation, dtype=float)  # 填入平移列 t。
    return transform


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """把 4x4 变换施加到一批点上，返回变换后的点集。

    参数:
        points: 形状 (N, 3) 的点集，每行一个 xyz 坐标。
        transform: 4x4 齐次变换矩阵。
    返回:
        形状 (N, 3) 的变换后点集，等价于逐点计算 R·p + t。
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    # 点按行排列，故用 points @ R.T 实现批量旋转，再加上平移（广播到每一行）。
    return points @ transform[:3, :3].T + transform[:3, 3]


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """求齐次变换的逆。

    对刚体变换而言，无需通用矩阵求逆：旋转的逆就是转置 R.T，
    对应的平移则是 -R.T·t，这样更快也更稳定。
    """
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return make_transform(rotation.T, -rotation.T @ translation)


def relative_transform(source_to_world: np.ndarray, target_to_world: np.ndarray) -> np.ndarray:
    """由两帧各自的“到世界”位姿，计算把 source 对齐到 target 的相对变换。

    已知 source->world 与 target->world，则 source->target = (target->world)^-1 · (source->world)。
    在本项目里用于从 bun.conf 真值位姿推导出两帧之间的真实相对位姿。
    """
    return invert_transform(target_to_world) @ source_to_world


def rotation_angle_deg(rotation: np.ndarray) -> float:
    """由旋转矩阵求其对应的旋转角（单位：度）。

    利用 trace(R) = 1 + 2·cos(θ) 反解出夹角 θ；clip 到 [-1, 1] 是为了
    抵消浮点误差导致 arccos 参数略微越界的问题。常用于衡量旋转误差大小。
    """
    value = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))
