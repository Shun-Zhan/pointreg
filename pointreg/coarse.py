"""粗配准：为精配准 ICP 提供一个足够好的初始位姿。

ICP 对初值敏感、容易陷入局部最优，因此先用粗配准把两片点云大致对上。
提供两种方法：
- pca_registration：基于主轴（PCA）对齐，纯 NumPy，无需 Open3D；
- fpfh_registration：FPFH 特征 + RANSAC，鲁棒性更好，但依赖 Open3D。
"""

from __future__ import annotations

from itertools import permutations, product

import numpy as np

from .nearest import nearest_neighbors
from .transforms import apply_transform, make_transform


def pca_registration(source: np.ndarray, target: np.ndarray, sample_limit: int = 5000) -> np.ndarray:
    """主轴（PCA）粗配准：对齐两片点云的主惯性轴。

    分别对源/目标做 PCA 得到三条主轴，理论上把源主轴旋到目标主轴即可对齐。
    但主轴方向和顺序都有歧义（每轴正负两向、三轴可任意排列），因此枚举全部
    6×8=48 种“轴排列×符号”组合，对每种候选用最近邻中位距离打分，取最优。
    """
    if len(source) < 3 or len(target) < 3:
        raise ValueError("PCA registration needs at least three points per cloud")
    cs, ct = source.mean(axis=0), target.mean(axis=0)  # 两片点云的质心
    # 对去质心后的协方差矩阵求特征向量即主轴；eigh 返回升序，[:, ::-1] 转成降序
    _, es = np.linalg.eigh(np.cov((source - cs).T))
    _, et = np.linalg.eigh(np.cov((target - ct).T))
    es, et = es[:, ::-1], et[:, ::-1]
    sample = source[::max(1, len(source) // sample_limit)]  # 抽样以加速逐候选打分
    best_score, best = float("inf"), np.eye(4)
    for perm in permutations(range(3)):        # 枚举目标三条主轴的排列
        target_axes = et[:, perm]
        for signs in product((-1.0, 1.0), repeat=3):  # 枚举每条轴的正负方向
            candidate_axes = target_axes @ np.diag(signs)
            rotation = candidate_axes @ es.T
            if np.linalg.det(rotation) < 0:    # 跳过镜像（行列式为负不是纯旋转）
                continue
            candidate = make_transform(rotation, ct - rotation @ cs)
            distances, _ = nearest_neighbors(apply_transform(sample, candidate), target)
            score = float(np.median(distances))  # 用中位距离作分数，抗离群
            if score < best_score:
                best_score, best = score, candidate
    return best


def fpfh_registration(source: np.ndarray, target: np.ndarray, voxel_size: float, seed: int = 42) -> np.ndarray:
    """FPFH 特征 + RANSAC 粗配准（依赖 Open3D）。

    流程：先估计法向量，再计算每点的 FPFH（快速点特征直方图）描述子，然后
    基于特征匹配用 RANSAC 稳健地估计变换。相比 PCA 更能应对部分重合，是本
    项目默认的粗配准方法。固定随机种子以保证结果可复现。
    """
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("FPFH registration requires Open3D") from exc
    if voxel_size <= 0:
        raise ValueError("FPFH requires voxel_size > 0")
    np.random.seed(seed)
    o3d.utility.random.seed(seed)

    def features(points: np.ndarray):
        """为单片点云估计法向并计算 FPFH 描述子，返回 (点云, 特征)。"""
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        # 法向估计半径取 2 倍体素，FPFH 计算半径取 5 倍体素（经验值）
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        descriptor = o3d.pipelines.registration.compute_fpfh_feature(
            cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
        return cloud, descriptor

    source_cloud, source_feature = features(source)
    target_cloud, target_feature = features(target)
    # 基于特征匹配的 RANSAC：每次采 3 个对应点估计变换，用两重几何一致性检查过滤
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_cloud, target_cloud, source_feature, target_feature, True,
        voxel_size * 1.5,  # 判定内点的最大对应距离
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,                 # 每次 RANSAC 采样的对应点数
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),   # 边长一致性
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)],  # 距离一致性
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),  # 最大迭代/置信度
    )
    return np.asarray(result.transformation)
