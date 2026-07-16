"""粗配准（coarse registration）模块。

本模块提供多种“把源点云大致对齐到目标点云”的初值估计方法，供后续 ICP 精配准使用：
- pca_registration：基于主成分分析（PCA）的轴对齐法，无需第三方库；
- fpfh_registration / multiscale_fpfh_registration：基于 FPFH 特征 + RANSAC（依赖 Open3D）；
- gcransac_fpfh_registration：基于 FPFH 匹配 + GC-RANSAC（依赖 pygcransac / scipy）。
低重叠场景下，粗配准的好坏直接决定 ICP 能否收敛到正确解。
"""

from __future__ import annotations

from itertools import permutations, product

import numpy as np

from .nearest import nearest_neighbors
from .transforms import apply_transform, make_transform


def pca_registration(source: np.ndarray, target: np.ndarray, sample_limit: int = 5000) -> np.ndarray:
    """用主成分分析（PCA）估计一个粗配准位姿。

    思路：分别求两个点云的主轴方向，再把源点云的主轴旋转到目标点云的主轴上。
    由于特征向量存在方向（正负号）和轴顺序的歧义，这里枚举所有轴顺序（6 种排列）
    和轴朝向（每轴 ±1，共 8 种）组合，逐个打分挑最好的一个。

    参数：
        source: 源点云，形状 (N, 3)。
        target: 目标点云，形状 (M, 3)。
        sample_limit: 打分时对源点云抽样的上限，避免每个候选都算全量最近邻太慢。
    返回：
        4x4 齐次变换矩阵（源到目标），即得分最优的候选位姿。
    """
    if len(source) < 3 or len(target) < 3:
        raise ValueError("PCA registration needs at least three points per cloud")
    # 各自的质心（中心点），后面用来去中心化并平移对齐。
    cs, ct = source.mean(axis=0), target.mean(axis=0)
    # 对去中心化后的协方差矩阵做特征分解，得到各自的主轴方向 es / et。
    _, es = np.linalg.eigh(np.cov((source - cs).T))
    _, et = np.linalg.eigh(np.cov((target - ct).T))
    # eigh 返回的特征值升序，这里反转成“主轴（方差最大）在前”的顺序。
    es, et = es[:, ::-1], et[:, ::-1]
    # 对源点云等间隔抽样，用于对每个候选变换快速打分。
    sample = source[::max(1, len(source) // sample_limit)]
    best_score, best = float("inf"), np.eye(4)
    for perm in permutations(range(3)):          # 枚举目标主轴的排列顺序（解决轴对应歧义）
        target_axes = et[:, perm]
        for signs in product((-1.0, 1.0), repeat=3):   # 枚举每根轴的正负朝向（解决方向歧义）
            candidate_axes = target_axes @ np.diag(signs)
            # 旋转矩阵：把源主轴映射到目标主轴。
            rotation = candidate_axes @ es.T
            # 行列式为负说明是镜像（反射）而非纯旋转，丢弃这类非法候选。
            if np.linalg.det(rotation) < 0:
                continue
            # 由旋转和质心差构造完整齐次变换。
            candidate = make_transform(rotation, ct - rotation @ cs)
            # 用最近邻距离的中位数打分：中位数对离群点更鲁棒，越小越好。
            distances, _ = nearest_neighbors(apply_transform(sample, candidate), target)
            score = float(np.median(distances))
            if score < best_score:
                best_score, best = score, candidate
    return best


def fpfh_registration(source: np.ndarray, target: np.ndarray, voxel_size: float, seed: int = 42) -> np.ndarray:
    """基于 FPFH 特征 + RANSAC 的粗配准（调用 Open3D 实现）。

    流程：对每个点云先估计法向量，再计算 FPFH（快速点特征直方图）描述子，
    然后用 RANSAC 在两组特征之间反复采样、假设、验证，找到一致的刚体变换。

    参数：
        voxel_size: 尺度参数，用于设定法向量/特征的搜索半径和匹配距离阈值。
        seed: 随机种子，固定它可复现 RANSAC 结果。
    返回：
        4x4 源到目标的变换矩阵。
    """
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("FPFH registration requires Open3D") from exc
    if voxel_size <= 0:
        raise ValueError("FPFH requires voxel_size > 0")
    # 同时固定 numpy 与 Open3D 的随机数，保证 RANSAC 结果可复现。
    np.random.seed(seed)
    o3d.utility.random.seed(seed)
    def features(points: np.ndarray):
        # 构建 Open3D 点云对象。
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        # 法向量搜索半径取 2 倍体素；FPFH 需要法向量作为输入。
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        # FPFH 描述子用更大的半径（5 倍体素）以刻画更大邻域的几何结构。
        descriptor = o3d.pipelines.registration.compute_fpfh_feature(
            cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
        return cloud, descriptor
    source_cloud, source_feature = features(source)
    target_cloud, target_feature = features(target)
    # RANSAC 主流程：基于特征匹配假设变换，并用多重几何检查器过滤错误对应。
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_cloud, target_cloud, source_feature, target_feature, True,
        voxel_size * 1.5,   # 内点判定的最大对应距离
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,                  # 每次假设采样 3 对点（估计刚体变换的最少点数）
        # 两个几何一致性检查器：边长比例约束 + 对应点距离约束，快速剔除无效假设。
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)],
        # 收敛准则：最多 10 万次迭代，或达到 0.999 置信度即停止。
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return np.asarray(result.transformation)


def _fpfh_cloud_and_features(points: np.ndarray, voxel_size: float):
    """对点云做体素下采样并计算 FPFH 特征，返回（下采样点云, 特征矩阵）。

    与 fpfh_registration 里的 features 不同，这里额外做了 voxel_down_sample 下采样，
    降低点数、加速后续匹配。返回的特征矩阵形状为 (点数, 33)（FPFH 维度为 33）。
    """
    import open3d as o3d

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud = cloud.voxel_down_sample(voxel_size)   # 体素下采样，均匀化并减少点数
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    features = o3d.pipelines.registration.compute_fpfh_feature(
        cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )
    # Open3D 的特征 data 是 (33, N)，转置成 (N, 33) 方便按点索引。
    return cloud, np.asarray(features.data).T


def _geometric_candidate_score(source: np.ndarray, target: np.ndarray, transform: np.ndarray, distance: float) -> tuple[int, int, float]:
    """仅用“互为最近邻”的几何对应关系给一个候选变换打分。

    低重叠场景下不能只看单向最近邻，否则大量点会被错误匹配。这里要求双向一致
    （reciprocal / 互最近邻）且双向距离都在阈值内，才算作可靠内点（inlier）。

    返回三元组：
        count:     可靠互最近邻内点数（越多越好，主排序依据）；
        all_count: 单向距离达标的粗略计数（次要排序依据）；
        error:     内点距离中位数（越小越好，用于最后打破平局）。
    """
    moved = apply_transform(source, transform)
    # 正向：每个源点在目标里的最近邻。
    source_distances, source_indices = nearest_neighbors(moved, target)
    # 反向：每个目标点在源里的最近邻。
    target_distances, target_indices = nearest_neighbors(target, moved)
    # 互最近邻检验：源点 i 的最近邻的最近邻仍是 i，才算互相匹配。
    reciprocal = target_indices[source_indices] == np.arange(len(source))
    # 内点还需满足双向距离都不超过阈值。
    inliers = reciprocal & (source_distances <= distance) & (target_distances[source_indices] <= distance)
    count = int(inliers.sum())
    all_count = int((source_distances <= distance).sum() + (target_distances <= distance).sum())
    error = float(np.median(source_distances[inliers])) if count else float("inf")
    return count, all_count, error


def multiscale_fpfh_registration(
    source: np.ndarray,
    target: np.ndarray,
    voxel_size: float,
    seed: int = 42,
    correspondence_distance: float = 0.01,
    trials_per_scale: int = 12,
) -> np.ndarray:
    """多尺度 FPFH-RANSAC：跑多个假设，用几何共识挑出最佳者。

    单次 FPFH-RANSAC 在低重叠时不稳定，容易给出错误位姿。这里在多个体素尺度、
    多个随机种子下反复生成候选，再用 _geometric_candidate_score 统一打分，
    保留内点最多、残差最小的那个，从而提高鲁棒性。

    参数：
        correspondence_distance: 判定内点的距离阈值。
        trials_per_scale: 每个尺度尝试的随机种子次数。
    返回：
        几何共识最优的 4x4 变换矩阵。
    """
    if voxel_size <= 0:
        raise ValueError("multi-scale FPFH requires voxel_size > 0")
    # 三个由小到大的尺度：小尺度看细节、大尺度更稳健，兼顾两者。
    scales = (voxel_size, voxel_size * 1.6, voxel_size * 2.4)
    best_transform: np.ndarray | None = None
    # 打分是三元组 (内点数, 粗略计数, -残差)，初值设成最差以便任何候选都能胜出。
    best_score = (-1, -1, float("-inf"))
    for scale_index, scale in enumerate(scales):
        for trial in range(trials_per_scale):
            # 每次用不同种子，保证同一尺度下 RANSAC 探索不同假设。
            candidate = fpfh_registration(source, target, scale, seed + scale_index * trials_per_scale + trial)
            reciprocal, total, error = _geometric_candidate_score(source, target, candidate, correspondence_distance)
            # 排序优先级：互最近邻内点越多越好；并列时残差越小越好（故取 -error）。
            score = (reciprocal, total, -error)
            if score > best_score:
                best_transform, best_score = candidate, score
    if best_transform is None:
        raise RuntimeError("multi-scale FPFH did not produce a registration hypothesis")
    return best_transform


def gcransac_from_correspondences(
    source_points: np.ndarray,
    target_points: np.ndarray,
    probabilities: np.ndarray | None = None,
    *,
    correspondence_distance: float = 0.01,
    max_iters: int = 10000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """从外部给定的一组对应点估计刚体变换（GC-RANSAC）。

    与普通 RANSAC 不同，GC-RANSAC（Graph-Cut RANSAC）用图割在局部优化内点集合，
    通常更准更稳。本函数只负责“给定匹配对 -> 估计变换”，匹配的产生在别处完成。

    参数：
        source_points / target_points: 一一对应的两组点，形状均为 (N, 3)。
        probabilities: 每对匹配的可信度权重（可选），越可信应越大；内部会归一化。
        correspondence_distance: 内点距离阈值。
        max_iters: RANSAC 最大迭代次数。
    返回：
        (4x4 变换矩阵, 内点布尔掩码)。
    """
    try:
        import pygcransac
    except ImportError as exc:
        raise RuntimeError("GC-RANSAC requires pygcransac") from exc
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    # 一系列输入合法性校验：形状、点数、阈值、有限性，尽早暴露错误。
    if source_points.ndim != 2 or source_points.shape[1] != 3 or target_points.shape != source_points.shape:
        raise ValueError("source_points and target_points must be matching (N, 3) arrays")
    if len(source_points) < 3:
        raise ValueError("GC-RANSAC needs at least three correspondences")
    if correspondence_distance <= 0 or max_iters < 1:
        raise ValueError("correspondence_distance and max_iters must be positive")
    if not (np.isfinite(source_points).all() and np.isfinite(target_points).all()):
        raise ValueError("GC-RANSAC correspondences must be finite")
    if probabilities is None:
        # 未给权重则视为等权。
        weights = np.ones(len(source_points), dtype=np.float64)
    else:
        weights = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        if len(weights) != len(source_points) or not np.isfinite(weights).all():
            raise ValueError("probabilities must be a finite vector matching the correspondence count")
        # 权重截到非负，并按最大值归一化到 [0, 1]（全零则退化为等权）。
        weights = np.maximum(weights, 0.0)
        maximum = float(weights.max())
        weights = weights / maximum if maximum > 0 else np.ones(len(weights), dtype=np.float64)
    # pygcransac 期望把源、目标点拼成一行 6 维 (x,y,z, x',y',z') 的对应关系。
    correspondences = np.concatenate([source_points, target_points], axis=1)
    np.random.seed(seed)
    model, inliers = pygcransac.findRigidTransform(
        correspondences,
        weights,
        threshold=correspondence_distance,
        conf=0.999,
        max_iters=max_iters,
        neighborhood=0,
        use_space_partitioning=True,
    )
    if model is None:
        raise RuntimeError("GC-RANSAC could not estimate a transform")
    inlier_mask = np.asarray(inliers, dtype=bool).reshape(-1)
    # 内点数不足 3 无法确定刚体变换，视为失败。
    if len(inlier_mask) != len(source_points) or int(inlier_mask.sum()) < 3:
        raise RuntimeError("GC-RANSAC returned fewer than three inliers")
    # pygcransac 使用行向量齐次坐标约定，而 PointReg 用列向量（主动式）变换，
    # 因此要对它返回的 4x4 矩阵做转置以匹配本项目约定。
    return np.asarray(model, dtype=float).T, inlier_mask


def gcransac_fpfh_registration(
    source: np.ndarray,
    target: np.ndarray,
    voxel_size: float,
    correspondence_distance: float = 0.01,
) -> np.ndarray:
    """完整流程：计算 FPFH -> 建立互匹配 -> 用 GC-RANSAC 估计位姿。

    先在特征空间里建立源、目标点的双向最近邻，只保留互为最近邻的可靠匹配，
    再按特征距离给每对匹配赋权（越像权重越高），最后交给 GC-RANSAC 求变换。
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("GC-RANSAC requires scipy and pygcransac") from exc
    source_cloud, source_features = _fpfh_cloud_and_features(source, voxel_size)
    target_cloud, target_features = _fpfh_cloud_and_features(target, voxel_size)
    source_points = np.asarray(source_cloud.points)
    target_points = np.asarray(target_cloud.points)
    # 在特征空间用 KD 树做双向最近邻查询。
    target_indices = cKDTree(target_features).query(source_features)[1]   # 每个源特征 -> 最近的目标特征
    source_indices = cKDTree(source_features).query(target_features)[1]   # 每个目标特征 -> 最近的源特征
    # 互最近邻筛选：源点 i 的匹配再映射回去仍是 i，才保留（剔除单向的假匹配）。
    matches = np.flatnonzero(source_indices[target_indices] == np.arange(len(source_points)))
    if len(matches) < 3:
        raise RuntimeError(f"GC-RANSAC found only {len(matches)} mutual FPFH correspondences")
    # 用特征距离衡量匹配质量，转成权重：距离越小（越相似）权重越接近 1。
    feature_distance = np.linalg.norm(source_features[matches] - target_features[target_indices[matches]], axis=1)
    probabilities = np.exp(-feature_distance / max(float(np.median(feature_distance)), 1e-6))
    # 把筛选后的对应点和权重交给 GC-RANSAC 求刚体变换。
    transform, _ = gcransac_from_correspondences(
        source_points[matches],
        target_points[target_indices[matches]],
        probabilities,
        correspondence_distance=correspondence_distance,
    )
    return transform
