"""点云预处理：体素下采样、离群点剔除、法向量估计、包围盒对角线。

原始扫描点云动辄几万点，直接配准既慢又易受噪声干扰。预处理的目的是在保留几何
结构的前提下减少点数、去掉噪点，并为需要法向量的算法（如点到面 ICP）准备法向。
"""
from __future__ import annotations

import numpy as np


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """体素下采样：把空间划分成边长为 voxel_size 的立方体格子，每格用其内点的质心代替。

    这样能在近似均匀密度的同时大幅减少点数。voxel_size<=0 或点云为空时原样返回。
    """
    points = np.asarray(points, dtype=float)
    if voxel_size <= 0 or not len(points):
        return points.copy()
    # 把坐标除以体素边长再向下取整，得到每个点所属体素的整数网格坐标（当作分组键）。
    keys = np.floor(points / voxel_size).astype(np.int64)
    # 对体素键去重：inverse 给出每个原始点属于哪个唯一体素（分组标签）。
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = np.reshape(inverse, -1)
    counts = np.bincount(inverse)  # 每个体素落入的点数，用作求质心的分母。
    # 按体素分组分别对 x/y/z 求和再除以点数，即得到每个体素的质心坐标。
    result = np.column_stack([
        np.bincount(inverse, weights=points[:, axis]) / counts for axis in range(3)
    ])
    return result


def preprocess_points(points: np.ndarray, voxel_size: float, remove_outliers: bool = False) -> np.ndarray:
    """标准预处理流程：先体素下采样，可选再做统计离群点剔除。

    参数:
        voxel_size: 体素边长，控制下采样粒度。
        remove_outliers: 是否额外剔除离群点（依赖 Open3D，缺库或点太少时自动跳过）。
    返回:
        预处理后的点云 (N, 3)。
    """
    down = voxel_downsample(points, voxel_size)
    # 未开启离群点剔除，或点数太少（统计不可靠）时，直接返回下采样结果。
    if not remove_outliers or len(down) < 30:
        return down
    try:
        import open3d as o3d
    except ImportError:
        return down  # 没有 Open3D 就跳过剔除，保证流程可继续。
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(down))
    # 统计式离群点剔除：对每点看其 20 个近邻的平均距离，超过均值 2 倍标准差的判为离群。
    filtered, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return np.asarray(filtered.points)


def estimate_outward_normals(points: np.ndarray, radius: float) -> np.ndarray | None:
    """为部分扫描点云估计法向量，并统一朝向“背离质心”的外侧方向。

    法向量本身有正负两个方向的歧义，点到面 ICP 等算法需要一致的定向。这里的做法是：
    先估计法向，再统一指向质心，最后取负号翻转成朝外，从而得到一致朝外的法向。
    缺少 Open3D 或点数不足 3 时返回 None。
    """
    try:
        import open3d as o3d
    except ImportError:
        return None
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return None  # 少于 3 个点无法定义平面/法向。
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    # 混合搜索：同时受半径 radius 和最大近邻数 30 限制，兼顾稀疏与稠密区域。
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    # 先把所有法向统一指向质心（“相机”设在质心处）……
    cloud.orient_normals_towards_camera_location(points.mean(axis=0))
    return -np.asarray(cloud.normals)  # ……再整体取负，使法向一致朝外。


def bounding_box_diagonal(*clouds: np.ndarray) -> float:
    """计算若干点云合并后的轴对齐包围盒（AABB）对角线长度。

    这个长度代表整体尺度，常用来把平移误差归一化成无量纲比例，
    使不同尺度的数据集之间可比。全部为空时返回 0。
    """
    valid = [np.asarray(c) for c in clouds if len(c)]
    if not valid:
        return 0.0
    joined = np.vstack(valid)  # 多片点云拼在一起，取整体的最小/最大角点。
    return float(np.linalg.norm(joined.max(axis=0) - joined.min(axis=0)))
