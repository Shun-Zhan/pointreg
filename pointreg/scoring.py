from __future__ import annotations

# ============================================================================
# 本模块：假设打分 / 自由空间穿透验证（低重叠配准的关键判别器）
# ----------------------------------------------------------------------------
# 核心思想（物理约束）：
#   激光/结构光扫描仪从某个视点(viewpoint)去观测物体表面，从视点到每个测量
#   点之间的连线（射线）必然穿过“空气”，即这段空间是被观测证实为“空”的自由空间。
#   如果一个候选位姿把另一帧点云的点塞进了这段本应为空的自由空间里，那么这个
#   对齐在物理上是不可能成立的——不管它的最近邻拟合度(fitness)看起来多高。
#   低重叠场景下，正确位姿往往拟合度不高，却几乎不产生穿透；而“错位滑移”的
#   假位姿可能拟合度很高，却大量穿透自由空间。利用这一点即可把二者区分开。
# ============================================================================

from dataclasses import dataclass, field

import numpy as np

from .nearest import NearestNeighborIndex
from .preprocessing import estimate_outward_normals
from .transforms import apply_transform, invert_transform


def _estimate_viewpoint(points: np.ndarray, normals: np.ndarray | None) -> np.ndarray:
    """根据平均外向法向量估计扫描仪的大致位置（视点）。

    原理：一次距离扫描只记录从某一侧可见的表面，所以所有表面法向量的平均值
    大致指回传感器所在方向。据此把视点放在点云中心沿平均法向外推若干倍包围盒
    对角线长度处，就得到一个合理的扫描仪位置估计。整个过程只用到输入点云本身，
    不涉及任何真值位姿(ground-truth)。

    参数：
        points: (N,3) 点云坐标。
        normals: (N,3) 每个点的外向法向量；若为 None 或空则退化为默认视点。
    返回：
        (3,) 估计的视点坐标。
    """
    center = points.mean(axis=0)
    # 包围盒对角线长度：作为“物体尺度”，用于确定视点外推的距离
    diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    # 没有法向信息时的退化处理：把视点放在中心正上方(Z方向)一个远处位置
    if normals is None or not len(normals):
        return center + np.array([0.0, 0.0, diagonal * 8.0])
    # 平均外向法向：近似指向传感器所在的方向
    direction = normals.mean(axis=0)
    norm = np.linalg.norm(direction)
    # 平均法向几乎抵消为零（表面朝向太均匀）时同样退化为默认视点
    if norm < 1e-9:
        return center + np.array([0.0, 0.0, diagonal * 8.0])
    # 沿单位化后的平均法向，从中心外推 8 倍对角线的距离作为视点
    return center + direction / norm * diagonal * 8.0


@dataclass(slots=True)
class _DepthMap:
    """从某个视点看向点云所构建的“深度图”，用于自由空间穿透判定。

    可以把它理解成：站在 viewpoint 处，用一张以视点为原点的“方向-深度”查找表
    记录“沿每个视线方向上，被观测到的最近表面点距离视点有多远”。
    有了这张表，就能判断任意一个查询点是落在已知表面之前（自由空间，穿透）
    还是之后（被表面遮挡，看不见）。

    字段：
        viewpoint: (3,) 扫描仪视点坐标。
        axis_u, axis_v: 与视线主方向正交的两个基向量，把三维单位视线方向投影成
                        二维 (u, v)，从而对方向做“像素格”离散化。
        bin_size: 方向离散化的分辨率（角度方向上的格子大小）。
        min_depth: {方向格键 -> 该方向上最近表面深度} 的字典，即深度图本体。
    """
    viewpoint: np.ndarray
    axis_u: np.ndarray
    axis_v: np.ndarray
    bin_size: float
    min_depth: dict[int, float] = field(default_factory=dict)

    def _keys(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """把一批点转换成 (方向格键, 深度) 两个数组。

        对每个点：
          1. 计算从视点指向该点的射线 rays，其长度就是深度 depths；
          2. 把射线单位化得到视线方向，再投影到 (axis_u, axis_v) 平面得到 (u, v)；
          3. 按 bin_size 把 (u, v) 量化取整，并用 u*200000 + v 编码成一个整数键，
             使方向相近的点落进同一个“方向格”，便于查同一视线上的最近表面深度。
        """
        rays = points - self.viewpoint
        depths = np.linalg.norm(rays, axis=1)          # 各点到视点的距离即“深度”
        directions = rays / depths[:, None]            # 单位化：只保留视线方向
        u = directions @ self.axis_u                   # 方向在 u 轴上的分量
        v = directions @ self.axis_v                   # 方向在 v 轴上的分量
        # 把 (u,v) 量化成整数格，再用大乘子把二维格坐标压成一维唯一整数键
        keys = np.round(u / self.bin_size).astype(np.int64) * 200000 + np.round(v / self.bin_size).astype(np.int64)
        return keys, depths

    @classmethod
    def build(cls, points: np.ndarray, viewpoint: np.ndarray, bin_size: float) -> "_DepthMap":
        """从给定视点构建深度图：记录每个方向格上被观测到的最近表面深度。

        步骤：
          1. 由所有点的平均视线方向确定“正前方” forward，作为深度图的主光轴；
          2. 用叉乘构造与 forward 正交的两个基 axis_u / axis_v（并处理 forward 与
             上向量近平行的退化情况），从而建立方向投影坐标系；
          3. 把所有点转成 (方向格键, 深度)，对每个方向格取最小深度，得到 min_depth。
             同一方向上最近的那个表面点，就代表这条视线上的可见表面。
        """
        rays = points - viewpoint
        depths = np.linalg.norm(rays, axis=1)
        directions = rays / depths[:, None]
        # 平均视线方向作为主光轴（正前方）
        forward = directions.mean(axis=0)
        forward /= np.linalg.norm(forward)
        # 用 forward 与世界 Y 轴叉乘得到一个水平基向量 axis_u
        axis_u = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        # 若 forward 与 Y 轴近乎平行导致叉乘退化，则改用 X 轴重新叉乘
        if np.linalg.norm(axis_u) < 1e-6:
            axis_u = np.cross(forward, np.array([1.0, 0.0, 0.0]))
        axis_u /= np.linalg.norm(axis_u)
        # axis_v 与 forward、axis_u 都正交，三者构成视点坐标系
        axis_v = np.cross(forward, axis_u)
        depth_map = cls(viewpoint=viewpoint, axis_u=axis_u, axis_v=axis_v, bin_size=bin_size)
        keys, depths = depth_map._keys(points)
        # 按方向格键排序，使同一方向格的点连续排列，便于分段求最小深度
        order = np.argsort(keys)
        sorted_keys, sorted_depths = keys[order], depths[order]
        # unique_keys 是去重后的方向格键，starts 是每段（同一键）的起始下标
        unique_keys, starts = np.unique(sorted_keys, return_index=True)
        # reduceat 按段规约：对每个方向格取该段内的最小深度 = 该方向的可见表面深度
        minima = np.minimum.reduceat(sorted_depths, starts)
        depth_map.min_depth = dict(zip(unique_keys.tolist(), minima.tolist()))
        return depth_map

    def violation_ratio(self, points: np.ndarray, margin: float) -> float:
        """返回“落在扫描仪已观测自由空间内”的点所占比例（即穿透率）。

        判据：一个点若沿其视线方向的深度，明显小于该方向上已知的最近表面深度
        （depth < min_depth - margin），说明它跑到了本应为空的自由空间里，即发生
        穿透，是几何上不合理的。margin 是容差，避免把贴着表面的正常点误判为穿透。
        分母 covered 只统计“落在有观测覆盖的方向格里”的点，未被覆盖的方向不计。

        参数：
            points: 待检验的点（通常是被候选位姿变换后的另一帧点云）。
            margin: 深度容差，越大越宽松。
        返回：
            穿透点数 / 被覆盖点数，取值 [0,1]，越大表示位姿越不合理。
        """
        keys, depths = self._keys(points)
        bad = 0        # 穿透（跑进自由空间）的点数
        covered = 0    # 落在有观测覆盖方向上的点数
        for key, depth in zip(keys.tolist(), depths.tolist()):
            min_depth = self.min_depth.get(key)
            if min_depth is None:
                continue                    # 该方向没有观测，无法判定，跳过
            covered += 1
            if depth < min_depth - margin:  # 比可见表面还近 => 穿透进自由空间
                bad += 1
        return bad / max(1, covered)


class HypothesisScorer:
    """候选位姿打分器：同时衡量“对称拟合度”和“自由空间一致性(不穿透)”。

    物理约束：扫描仪到每个测量点的射线都穿过空气(自由空间)。一个把另一帧点云
    塞进这段自由空间的候选对齐，在几何上是不可能成立的——无论它的最近邻拟合度
    多高。正因如此，本打分器能让“低重叠但正确”的位姿（拟合度低、几乎零穿透）
    击败“拟合度高但实为错位滑移”的假位姿（拟合度高、大量穿透）。

    构造时同时为 source 和 target 各建一张深度图，从而支持双向(对称)的穿透检验。
    """

    def __init__(self, source: np.ndarray, target: np.ndarray, voxel_size: float,
                 fitness_threshold: float, bin_size: float = 0.004) -> None:
        self.source = source
        self.target = target
        self.fitness_threshold = fitness_threshold   # 判定“内点”的最近邻距离阈值
        self.margin = voxel_size * 2                 # 穿透判定的深度容差（越大越宽松）
        self.target_index = NearestNeighborIndex(target)  # target 的最近邻索引(算拟合度用)
        # 估计各自的外向法向，进而估计各自扫描仪视点，最后各建一张深度图
        source_normals = estimate_outward_normals(source, voxel_size * 3)
        target_normals = estimate_outward_normals(target, voxel_size * 3)
        self.source_depth = _DepthMap.build(source, _estimate_viewpoint(source, source_normals), bin_size)
        self.target_depth = _DepthMap.build(target, _estimate_viewpoint(target, target_normals), bin_size)

    def symmetric_fitness(self, transform: np.ndarray) -> tuple[float, float]:
        """对称拟合度：正向和反向内点比例取较小值，附带内点 RMSE。

        单向拟合度容易被“把小片点云藏进大片点云内部”这类假位姿骗过，所以这里
        同时算 source->target(forward) 和 target->source(backward) 两个方向的内点
        比例，取较小者作为最终拟合度，抑制单向虚高。RMSE 由正向内点距离算出。
        返回：(对称拟合度, 内点RMSE)。
        """
        moved = apply_transform(self.source, transform)      # 用候选位姿变换 source
        forward, _ = self.target_index.query(moved)          # 变换后各点到 target 的最近距离
        backward, _ = NearestNeighborIndex(moved).query(self.target)  # target 各点到变换后 source 的最近距离
        # 内点比例 = 最近距离在阈值内的点所占比例
        fitness_forward = float(np.mean(forward <= self.fitness_threshold)) if len(forward) else 0.0
        fitness_backward = float(np.mean(backward <= self.fitness_threshold)) if len(backward) else 0.0
        inliers = forward[forward <= self.fitness_threshold]
        rmse = float(np.sqrt(np.mean(inliers ** 2))) if len(inliers) else float("inf")
        return min(fitness_forward, fitness_backward), rmse  # 取双向较小值，更保守可靠

    def violation(self, transform: np.ndarray) -> float:
        """双向自由空间穿透率：取正反两个方向穿透率的最大值。

        forward：把 source 用候选位姿变换后，检验它是否穿进 target 的自由空间；
        backward：把 target 用逆变换搬到 source 坐标系，检验它是否穿进 source 的
        自由空间。任一方向穿透严重都说明位姿不合理，故取 max（最严格判定）。
        """
        forward = self.target_depth.violation_ratio(apply_transform(self.source, transform), self.margin)
        backward = self.source_depth.violation_ratio(apply_transform(self.target, invert_transform(transform)), self.margin)
        return max(forward, backward)

    def score(self, transform: np.ndarray) -> tuple[float, float, float]:
        """返回 (综合分, 拟合度, 穿透率)；综合分越高越好。

        综合分 = 拟合度 − 3×穿透率：在鼓励高拟合度的同时对穿透施加较重惩罚，
        从而把物理上不合理的高拟合假位姿压下去。
        """
        fitness, _ = self.symmetric_fitness(transform)
        violation = self.violation(transform)
        return fitness - 3.0 * violation, fitness, violation
