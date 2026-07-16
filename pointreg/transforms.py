"""刚体变换工具：四元数/旋转矩阵/4×4 齐次变换之间的基础运算。

本模块只依赖 NumPy，提供整个配准流程共用的几何原语：
构造/应用/求逆变换、复合相对位姿、以及从旋转矩阵反算旋转角。
"""

from __future__ import annotations

import numpy as np


def quaternion_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    """把 (x, y, z, w) 顺序的四元数转换成 3×3 旋转矩阵。

    输入会先做归一化以消除数值误差；Stanford 的 bun.conf 就采用这种
    xyzw 排列，因此解析真值位姿时会用到本函数。
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        raise ValueError("quaternion must contain x, y, z, w")
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("zero quaternion is invalid")
    x, y, z, w = q / norm  # 归一化后拆出四个分量
    # 标准的四元数→旋转矩阵公式（右手系、主动旋转）
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def make_transform(rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> np.ndarray:
    """由旋转矩阵和平移向量拼装出 4×4 齐次变换矩阵，缺省部分取单位/零。"""
    transform = np.eye(4)
    if rotation is not None:
        transform[:3, :3] = np.asarray(rotation, dtype=float)  # 左上 3×3 放旋转
    if translation is not None:
        transform[:3, 3] = np.asarray(translation, dtype=float)  # 右上 3×1 放平移
    return transform


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """把 4×4 变换作用到 (N, 3) 点集上，返回变换后的坐标。

    这里用 `points @ R.T + t` 的写法对整批点做向量化运算，比逐点乘法快很多。
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    return points @ transform[:3, :3].T + transform[:3, 3]


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """求刚体变换的逆。

    对旋转 R、平移 t，其逆为 (Rᵀ, -Rᵀt)；利用旋转矩阵正交（R⁻¹=Rᵀ）
    避免直接做数值求逆，更快也更稳定。
    """
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return make_transform(rotation.T, -rotation.T @ translation)


def relative_transform(source_to_world: np.ndarray, target_to_world: np.ndarray) -> np.ndarray:
    """给定两帧各自到世界坐标系的位姿，算出“源→目标”的相对变换（真值）。"""
    return invert_transform(target_to_world) @ source_to_world


def rotation_angle_deg(rotation: np.ndarray) -> float:
    """由旋转矩阵反算旋转角（度）。

    利用 trace(R) = 1 + 2cosθ 的关系；clip 到 [-1, 1] 防止浮点误差
    让 arccos 拿到域外输入。用于衡量估计位姿与真值的旋转误差。
    """
    value = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))
