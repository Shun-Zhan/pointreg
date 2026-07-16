"""点云预处理：体素下采样、离群点剔除，以及包围盒对角线长度计算。

下采样既降低点数以提速，也让点密度均匀化、利于稳定的对应关系；
包围盒对角线则作为平移误差的归一化尺度（成功判据里用到）。
"""

from __future__ import annotations

import numpy as np


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """体素栅格下采样：把落在同一体素内的点用其质心代替。

    纯 NumPy 实现，不依赖 Open3D：先按体素尺寸把坐标离散成整数栅格键，
    再对相同键的点求坐标均值。voxel_size<=0 或空点云时原样返回。
    """
    points = np.asarray(points, dtype=float)
    if voxel_size <= 0 or not len(points):
        return points.copy()
    keys = np.floor(points / voxel_size).astype(np.int64)  # 每个点所属体素的整数坐标
    # inverse 给出每个点对应的“唯一体素”编号
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = np.reshape(inverse, -1)  # 兼容 NumPy 2.x：unique 的 inverse 形状可能带多余维度
    counts = np.bincount(inverse)      # 每个体素内的点数
    # 对每个坐标轴按体素分组求和再除以点数，得到质心
    result = np.column_stack([
        np.bincount(inverse, weights=points[:, axis]) / counts for axis in range(3)
    ])
    return result


def preprocess_points(points: np.ndarray, voxel_size: float, remove_outliers: bool = False) -> np.ndarray:
    """完整预处理：先体素下采样，可选再做统计离群点剔除。"""
    down = voxel_downsample(points, voxel_size)
    # 点太少时剔除离群点意义不大，且可能误删有效点，直接返回
    if not remove_outliers or len(down) < 30:
        return down
    try:
        import open3d as o3d
    except ImportError:
        return down  # 没装 Open3D 就跳过离群点剔除
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(down))
    # 基于邻域距离的统计滤波：距离超过均值 2 倍标准差的点视为离群点
    filtered, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return np.asarray(filtered.points)


def bounding_box_diagonal(*clouds: np.ndarray) -> float:
    """计算一个或多个点云合并后的轴对齐包围盒（AABB）对角线长度。

    用作平移误差的归一化基准：平移误差 / 对角线 = 相对平移误差。
    """
    valid = [np.asarray(c) for c in clouds if len(c)]
    if not valid:
        return 0.0
    joined = np.vstack(valid)
    # 对角线 = 各轴最大坐标与最小坐标之差构成向量的模长
    return float(np.linalg.norm(joined.max(axis=0) - joined.min(axis=0)))
