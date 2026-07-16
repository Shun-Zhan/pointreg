from __future__ import annotations

# ============================================================================
# 本模块：手工全局搜索配准（低重叠两帧点云）
# ----------------------------------------------------------------------------
# 目标：在没有真值、只有两帧点云的情况下，自动找出把 source 对齐到 target 的
#       刚体变换(4x4位姿)。低重叠意味着两帧只有一小块公共区域，纯靠拟合度极易
#       被“错位滑移”的假位姿误导，因此本模块引入“自由空间穿透”作为物理判据。
#
# 整体流程（GlobalRegistrationSearch.run）：
#   1) 旋转粗搜：用 super-Fibonacci 在 SO(3) 上均匀撒一批旋转，用“带符号体素栅格
#      + FFT 互相关”对每个旋转打分（双向取小），挑出多样化的种子旋转。
#   2) 爬山：对每个种子做随机扰动爬山，进一步优化旋转分数。
#   3) 位姿级精修：以拟合度−穿透惩罚为目标，用 FFT 重解平移 + 短 trimmed-ICP 内抛光。
#   4) 紧致收尾：双向、两档紧度的 point-to-plane ICP 精配，产生大量候选位姿。
#   5) 门控 + Borda 投票选择：候选必须满足穿透率低于阈值且拟合度非平凡，再用四个
#      指标做排名聚合(Borda)选出最终位姿，比任何单一指标都稳。
#   期间还有“局部网格锁定”阶段专门压制毫米级切向滑移。
# 全程只使用两帧输入点云，不使用任何真值。
# ============================================================================

from dataclasses import asdict, dataclass, replace
from time import perf_counter
from typing import Mapping

import numpy as np
import scipy.fft as scipy_fft
from scipy.spatial.transform import Rotation

from .icp import custom_icp
from .models import RegistrationConfig
from .nearest import NearestNeighborIndex
from .preprocessing import estimate_outward_normals
from .scoring import HypothesisScorer, _DepthMap, _estimate_viewpoint
from .transforms import apply_transform, invert_transform, make_transform, rotation_angle_deg


@dataclass(slots=True)
class GlobalCandidateMetrics:
    """单个候选位姿的评估指标集合（供排序、投票和输出诊断用）。

    字段含义：
        label: 候选来源标签（如 global:0:refined、local_lock:2 等），便于追踪。
        transform: 该候选的 4x4 位姿矩阵。
        fitness: 对称拟合度（越大越好）。
        violation: 粗尺度自由空间穿透率（越小越好）。
        fine_violation: 细尺度穿透率（更敏感，捕捉轻微滑移）。
        rmse: 内点均方根误差（越小越好）。
        fine_fitness: 细阈值下的拟合度。
        gated: 是否通过门控（穿透率与拟合度双重约束）。
        borda_score: Borda 排名聚合总分（越小越靠前）。
    """
    label: str
    transform: np.ndarray
    fitness: float
    violation: float
    fine_violation: float
    rmse: float
    fine_fitness: float
    gated: bool = False
    borda_score: float = float("inf")

    def to_dict(self) -> dict[str, object]:
        # 转成可 JSON 序列化的字典：numpy 矩阵需转成嵌套列表
        data = asdict(self)
        data["transform"] = self.transform.tolist()
        return data


@dataclass(slots=True)
class GlobalSearchResult:
    """全局搜索的最终结果打包（最终位姿 + 粗位姿 + 各阶段计时 + 全部候选诊断）。"""
    transform: np.ndarray
    selected_label: str
    coarse_transform: np.ndarray
    coarse_label: str
    fitness: float
    violation: float
    fine_violation: float
    gate_passed: bool
    message: str
    timings_ms: dict[str, float]
    candidates: list[GlobalCandidateMetrics]

    def to_dict(self) -> dict[str, object]:
        return {
            "transform": self.transform.tolist(),
            "selected_label": self.selected_label,
            "coarse_transform": self.coarse_transform.tolist(),
            "coarse_label": self.coarse_label,
            "fitness": self.fitness,
            "violation": self.violation,
            "fine_violation": self.fine_violation,
            "gate_passed": self.gate_passed,
            "message": self.message,
            "timings_ms": self.timings_ms,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def super_fibonacci_rotations(count: int) -> np.ndarray:
    """用 super-Fibonacci 螺旋在 SO(3) 上做近似均匀采样(Alexa 2022)。

    旋转搜索需要在“所有可能朝向”里均匀撒点。直接对欧拉角均匀采样会在两极堆积，
    不均匀。super-Fibonacci 用两个无理数(phi、psi)构造两条螺旋相位，在四维单位球
    (即单位四元数，一一对应三维旋转)上生成分布极均匀的一批四元数，再转成旋转矩阵。
    参数 count 是采样数量；返回 (count,3,3) 旋转矩阵数组。
    """
    phi = np.sqrt(2.0)
    psi = 1.533751168755204288118041     # 论文给定的第二个无理常数
    s = np.arange(count) + 0.5           # 半整数偏移，避免端点退化
    t = s / count                        # 归一化到 (0,1)，控制两组分量的能量分配
    d = 2 * np.pi * s
    # 四元数按 (r·三角, R·三角) 拆成两组，r²+R²=1 保证落在单位球面上
    r = np.sqrt(t)
    big_r = np.sqrt(1 - t)
    alpha = d / phi                      # 第一条螺旋相位
    beta = d / psi                       # 第二条螺旋相位
    quats = np.stack([r * np.sin(alpha), r * np.cos(alpha),
                      big_r * np.sin(beta), big_r * np.cos(beta)], axis=1)
    return Rotation.from_quat(quats).as_matrix()


def axis_angle_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """由“旋转轴 + 旋转角”构造旋转矩阵（罗德里格斯公式）。

    axis 会被单位化，cross 是其反对称叉乘矩阵；
    R = I + sin(θ)·[axis]× + (1−cos(θ))·[axis]×²。
    在爬山/网格锁定阶段用它对现有旋转施加小角度扰动。
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)   # 单位化旋转轴
    # 叉乘矩阵 [axis]×：满足 [axis]× v = axis × v
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle_rad) * cross + (1 - np.cos(angle_rad)) * (cross @ cross)


def fibonacci_axes(count: int) -> np.ndarray:
    """用斐波那契球面采样在单位球面上均匀撒 count 个方向向量。

    黄金角螺旋能让点在球面上近似等面积分布（避免两极堆积），常用于生成均匀的
    候选旋转轴/视线方向。返回 (count,3) 的单位方向数组。
    """
    index = np.arange(count)
    golden = (1 + 5 ** 0.5) / 2          # 黄金比，用于黄金角螺旋
    z = 1 - (2 * index + 1) / count      # 沿 z 轴等间距分层（等面积）
    radius = np.sqrt(1 - z * z)          # 每层圆环半径
    theta = 2 * np.pi * index / golden   # 黄金角决定的方位角
    return np.stack([radius * np.cos(theta), radius * np.sin(theta), z], axis=1)


def _occupancy_grid(points: np.ndarray, origin: np.ndarray, voxel: float, dims: np.ndarray) -> np.ndarray:
    """把点云体素化成“占据栅格”：有点的体素置 1，其余为 0。

    origin 是栅格左下角(最小角)的世界坐标，voxel 是体素边长，dims 是三维格数。
    先把每个点减去 origin 除以 voxel 取整得到体素下标，过滤掉越界的点，再置 1。
    这是后续 FFT 互相关的输入之一（源点云的“形状指纹”）。
    """
    grid = np.zeros(tuple(dims), dtype=np.float32)
    index = np.floor((points - origin) / voxel).astype(int)   # 各点所在体素下标
    valid = ((index >= 0) & (index < dims)).all(axis=1)       # 只保留落在栅格范围内的点
    grid[index[valid, 0], index[valid, 1], index[valid, 2]] = 1.0
    return grid


def _signed_grid(points: np.ndarray, origin: np.ndarray, voxel: float, dims: np.ndarray,
                 free_penalty: float, normal_radius: float) -> np.ndarray:
    """带符号占据栅格：表面体素为 +1，扫描仪已观测的自由空间体素为负惩罚值。

    普通占据栅格只区分“有点/没点”，无法区分“没点是因为空气”还是“没点是因为
    被遮挡看不见”。这里把已被观测证实为空的自由空间体素赋成 -free_penalty：
    这样在 FFT 互相关里，如果一个候选位姿把源点云推穿过目标表面、落进自由空间，
    就会命中负值、被扣分；从而相关性偏好“物理上合理的表面接触”，而非“壳穿壳”的
    假位姿。normal_radius 用于估计法向进而估计视点。
    """
    grid = np.zeros(tuple(dims), dtype=np.float32)
    # 估计法向 -> 估计视点 -> 构建深度图，用于判断哪些体素处于自由空间
    normals = estimate_outward_normals(points, normal_radius)
    depth_map = _DepthMap.build(points, _estimate_viewpoint(points, normals), bin_size=0.004)
    # 枚举所有体素中心的世界坐标（+0.5 取体素中心而非角点）
    axes = [np.arange(d) for d in dims]
    mesh = np.meshgrid(*axes, indexing="ij")
    centers = origin + (np.stack(mesh, axis=-1).reshape(-1, 3) + 0.5) * voxel
    # 查每个体素中心在深度图里对应方向的可见表面深度
    keys, depths = depth_map._keys(centers)
    min_depths = np.array([depth_map.min_depth.get(key, np.inf) for key in keys.tolist()])
    # 体素比该方向的可见表面还近(留 1.5 体素余量) => 处于自由空间 => 标记为负惩罚
    free = depths < (min_depths - voxel * 1.5)
    grid.ravel()[free] = -free_penalty
    # 再把真正有表面点的体素覆盖为 +1（表面证据优先于自由空间惩罚）
    index = np.floor((points - origin) / voxel).astype(int)
    valid = ((index >= 0) & (index < dims)).all(axis=1)
    grid[index[valid, 0], index[valid, 1], index[valid, 2]] = 1.0
    return grid


class SignedCorrelationSolver:
    """基于带符号体素栅格的 FFT 互相关：给定旋转，一次性求出最优平移并打分。

    核心技巧（为什么用 FFT）：给定一个固定旋转后，我们还要找最优平移。若暴力枚举
    所有平移量，代价极高。但“把源栅格平移 t 后与目标栅格的重叠得分”正是两个栅格的
    互相关(cross-correlation)在 t 处的取值，而互相关可由 FFT 高效计算：
        corr = IFFT( conj(FFT(source)) · FFT(target) )
    一次 FFT 就同时算出了所有平移量的得分，corr 的峰值位置即最优平移、峰值大小即得分
    （表面接触数 减去 自由空间穿透罚分）。全程只用两帧输入点云。
    注意这是循环(circular)相关，故用足够大的栅格并把大于半程的位移解释成负向位移。
    """

    def __init__(self, source: np.ndarray, target: np.ndarray, voxel: float,
                 dims: int = 64, free_penalty: float = 2.0, normal_radius: float = 0.0) -> None:
        self.voxel = voxel
        self.dims = np.array([dims, dims, dims])
        extent = self.dims * voxel               # 栅格覆盖的物理边长
        self.source = source
        self.source_center = source.mean(axis=0) # 源点云质心：旋转都绕质心进行
        # 目标栅格以目标质心为中心；源栅格以原点为中心。两 origin 之差用于平移换算
        self.target_origin = target.mean(axis=0) - extent / 2
        self.source_origin = -extent / 2
        radius = normal_radius if normal_radius > 0 else voxel * 1.5
        # 目标端只需算一次：构建带符号栅格并预先做 FFT，缓存其频谱以复用
        self.target_spectrum = scipy_fft.rfftn(
            _signed_grid(target, self.target_origin, voxel, self.dims, free_penalty, radius), workers=4)
        self.count = len(source)                 # 源点数，用于把得分归一化

    def _correlation(self, rotation: np.ndarray) -> np.ndarray:
        """对给定旋转，返回所有平移量上的互相关得分体（3D 数组）。"""
        rotated = (self.source - self.source_center) @ rotation.T  # 绕质心旋转源点云
        grid = _occupancy_grid(rotated, self.source_origin, self.voxel, self.dims)
        spectrum = scipy_fft.rfftn(grid, workers=4)                # 源栅格频谱
        # 互相关定理：conj(源频谱)·目标频谱，逆变换回空间域即得所有平移的得分
        return scipy_fft.irfftn(spectrum.conj() * self.target_spectrum, s=tuple(self.dims), workers=4)

    def score(self, rotation: np.ndarray) -> float:
        """该旋转下的最优得分 = 互相关峰值 / 源点数（归一化，便于跨旋转比较）。"""
        return float(self._correlation(rotation).max()) / self.count

    def best_transform(self, rotation: np.ndarray) -> np.ndarray:
        """由互相关峰值位置反解出最优平移，组装成完整 4x4 位姿。"""
        correlation = self._correlation(rotation)
        # 峰值所在体素下标 = 最优的整数平移量（以体素为单位）
        cell = np.array(np.unravel_index(int(np.argmax(correlation)), tuple(self.dims)), dtype=float)
        # 循环相关：超过栅格一半的位移应解释为反方向的负位移（把 [0,dims) 折到 [-dims/2,dims/2)）
        shift = np.where(cell > self.dims / 2, cell - self.dims, cell)
        # 把体素位移换算成世界坐标平移，再补上两栅格 origin 的偏移
        translation = (self.target_origin - self.source_origin) + shift * self.voxel
        # 因为旋转是绕源质心做的，最终平移需扣掉旋转带来的质心位移
        return make_transform(rotation, translation - rotation @ self.source_center)


class GlobalRegistrationSearch:
    """带自由空间感知目标的两帧全局位姿搜索（本模块的主控类）。

    阶段：(1) SO(3) 均匀旋转网格，用双向带符号相关打分；(2) 从多样化的高分种子
    出发做随机爬山；(3) 以“拟合度−穿透惩罚”为目标做局部精修，内嵌短 trimmed-ICP
    抛光；(4) 双向、两档紧度的 point-to-plane 紧致收尾；(5) 门控选择：候选须保持
    穿透率低于标定阈值且对称拟合度非平凡，再按 fitness − 10×violation 等指标做
    Borda 排名聚合。全程只用两帧输入点云。
    """

    def __init__(self, source: np.ndarray, target: np.ndarray, config: RegistrationConfig) -> None:
        self.config = config
        self.voxel = config.voxel_size
        self.source = source
        self.target = target
        # 抽稀出“快速版”点云用于打分/相关，控制计算量（精配阶段仍用原始稠密点云）
        self.fast_source = self._sample(source, config.multi_score_points)
        self.fast_target = self._sample(target, config.multi_score_points)
        self.rng = np.random.default_rng(config.random_seed)   # 固定随机种子，结果可复现
        # 正/反两个方向的相关求解器：正向 source->target，反向 target->source
        self.forward = SignedCorrelationSolver(self.fast_source, self.fast_target, self.voxel * 2,
                                               normal_radius=self.voxel * 3)
        self.backward = SignedCorrelationSolver(self.fast_target, self.fast_source, self.voxel * 2,
                                                normal_radius=self.voxel * 3)
        # 粗尺度打分器：拟合阈值=voxel，穿透 margin=voxel*2
        self.scorer = HypothesisScorer(self.fast_source, self.fast_target, self.voxel, self.voxel * 2)
        # margin=voxel*0.8 的细违规打分器：捕捉毫米级切向滑移带来的轻微穿透
        self.fine_scorer = HypothesisScorer(self.fast_source, self.fast_target, self.voxel * 0.4,
                                            self.voxel * 2)
        self.fine_forward: SignedCorrelationSolver | None = None  # 高分辨率相关器，延迟到网格锁定时才建
        self.target_tree = NearestNeighborIndex(self.fast_target) # 目标快速点云的最近邻索引

    @staticmethod
    def _sample(points: np.ndarray, limit: int) -> np.ndarray:
        """均匀抽稀点云到不超过 limit 个点（等间隔切片，保持空间分布）。"""
        if len(points) <= limit:
            return points
        return points[:: max(1, len(points) // limit)]

    # ---------- 旋转层目标（rotation-level objective） ----------

    def _biscore(self, rotation: np.ndarray) -> float:
        """双向相关得分取较小值：正向用 rotation，反向用其转置(逆旋转)。

        取双向最小值可抑制“单向看着好、反向其实很差”的旋转，得到更可靠的旋转分。
        """
        return min(self.forward.score(rotation), self.backward.score(rotation.T))

    def _hill_climb(self, rotation: np.ndarray, score: float, rounds: int) -> tuple[np.ndarray, float]:
        """随机扰动爬山：在旋转分数上做带自适应步长的局部搜索。

        每轮对当前旋转施加一个随机轴、随机小角度的扰动，若新旋转分更高则接受；
        若连续 4 次没有改进(stall)，就把扰动半径 radius 缩小(×0.6，最低 1.5°)，
        实现“先大范围探索、后小步精修”的退火式搜索。返回最终旋转及其分数。
        """
        radius, stall = 9.0, 0                # 初始扰动半径(度)与停滞计数
        for _ in range(rounds):
            # 随机轴 + [0.3,1.0]×radius 的随机角，左乘施加到当前旋转上
            candidate = axis_angle_matrix(self.rng.normal(size=3),
                                          np.radians(radius) * self.rng.uniform(0.3, 1.0)) @ rotation
            candidate_score = self._biscore(candidate)
            if candidate_score > score:
                rotation, score, stall = candidate, candidate_score, 0  # 接受更优解，重置停滞
            else:
                stall += 1
                if stall >= 4:
                    radius, stall = max(1.5, radius * 0.6), 0           # 停滞则收缩步长
        return rotation, score

    # ---------- 位姿层精修（pose-level refinement） ----------

    def _short_icp(self, transform: np.ndarray) -> np.ndarray:
        """由粗到精的短程 trimmed-ICP 抛光：从大对应距离逐步收紧到小距离。

        分三档 (距离倍率, 迭代数) 逐级把对应距离收紧，每档都做带截断(trim)的 ICP。
        trim_fraction=0.5 + adaptive_trim 会剔除较差的一半对应，抗离群/抗低重叠。
        某档若失败(status=failed)则保留上一档结果，保证不倒退。
        """
        result = transform
        for distance_scale, iterations in ((1.5, 10), (1.0, 15), (0.7, 15)):
            stage = replace(self.config,
                            max_correspondence_distance=self.config.max_correspondence_distance * distance_scale,
                            max_iterations=iterations, trim_fraction=0.5, adaptive_trim=True,
                            min_correspondences=10)
            polished, _, status, _ = custom_icp(self.fast_source, self.fast_target, result, stage)
            if status != "failed":
                result = polished
        return result

    def _fine_fitness(self, transform: np.ndarray) -> float:
        """细阈值(1.2 体素)下的对称拟合度：正反两向内点比例取小值。"""
        moved = apply_transform(self.fast_source, transform)
        forward, _ = self.target_tree.query(moved)                     # 变换后 source 到 target 的最近距离
        backward, _ = NearestNeighborIndex(moved).query(self.fast_target)  # target 到变换后 source 的最近距离
        threshold = self.voxel * 1.2
        return min(float(np.mean(forward <= threshold)), float(np.mean(backward <= threshold)))

    def _refine_objective(self, transform: np.ndarray) -> float:
        """精修目标函数 = 细拟合度 − 3×穿透率，并对“逃逸到零重叠”设下限保护。

        为什么需要穿透惩罚：漂移(drift)的位姿常常能靠把点堆到边缘刷高拟合度，但这些
        点其实落进了已观测的自由空间；减去穿透惩罚能让爬山“诚实”，不被假优解带偏。
        下限保护：拟合度 < 0.03 说明已经滑到几乎零重叠的虚无处，直接返回 -inf 拒绝。
        """
        fitness = self._fine_fitness(transform)
        if fitness < 0.03:
            return float("-inf")
        return fitness - 3.0 * self.scorer.violation(transform)

    def _fitness_refine(self, rotation: np.ndarray) -> np.ndarray:
        """在旋转上爬升精修目标；每次评估都用 FFT 重解平移 + 短 trimmed-ICP 抛光。

        流程：先由相关器求出该旋转的最优平移并 ICP 抛光作为起点，然后做三轮
        (半径逐轮减半) 的随机旋转扰动爬山；每个候选旋转都重新 FFT 定平移再抛光再打分，
        接受更优者。这样旋转与平移交替优化，收敛到局部最优位姿。
        """
        current = self._short_icp(self.forward.best_transform(rotation))
        current_score = self._refine_objective(current)
        radius = 8.0
        for _ in range(3):
            for _ in range(self.config.multi_refine_samples):
                # 对当前旋转施加随机小扰动
                candidate_rotation = axis_angle_matrix(
                    self.rng.normal(size=3), np.radians(radius) * self.rng.uniform(0.1, 1.0)) @ current[:3, :3]
                # 新旋转 -> FFT 重解平移 -> 短 ICP 抛光 -> 打分
                candidate = self._short_icp(self.forward.best_transform(candidate_rotation))
                score = self._refine_objective(candidate)
                if score > current_score:
                    current, current_score = candidate, score
            radius = max(1.5, radius * 0.5)   # 每轮收缩扰动半径，先探索后精修
        return current

    # ---------- 紧致收尾（tight finishing） ----------

    def _tight_p2p(self, source_cloud, target_cloud, initial: np.ndarray,
                   distance: float, tukey_k: float, iterations: int = 60) -> np.ndarray:
        """调用 Open3D 的 point-to-plane ICP 做紧致精配，配合 Tukey 鲁棒损失。

        point-to-plane（点到面）比点到点收敛更快、精度更高，适合最后阶段贴合表面。
        TukeyLoss(k) 是鲁棒核，抑制大残差对应(离群)的影响；distance 是最大对应距离。
        返回精配后的 4x4 位姿。
        """
        import open3d as o3d
        loss = o3d.pipelines.registration.TukeyLoss(k=tukey_k)
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations)
        result = o3d.pipelines.registration.registration_icp(
            source_cloud, target_cloud, distance, initial, estimation, criteria)
        return np.asarray(result.transformation)

    def _make_pcd(self, points: np.ndarray):
        """把 numpy 点数组封装成 Open3D 点云并估计法向（point-to-plane ICP 需要法向）。"""
        import open3d as o3d
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel * 4, max_nn=30))
        return cloud

    def _finish_candidates(self, source_cloud, target_cloud, initial: np.ndarray,
                           label: str) -> list[tuple[str, np.ndarray]]:
        """围绕一个初始位姿，生成一组双向、两档紧度的精配候选。

        先用较宽(1.2 体素)的 point-to-plane 各做一次正向/反向 ICP 得到“入口位姿”，
        再从入口出发以两档更紧的 (距离, Tukey-k) 各做正反向精配。反向结果需取逆变换
        换回 source->target 方向。这样同一起点会派生出多个略有差异的候选，交给后续
        门控+投票挑选，比只信一个方向/一档紧度更稳健。
        """
        candidates: list[tuple[str, np.ndarray]] = []
        # 入口位姿：较宽对应距离先粗贴一次，正反各一次
        entry_forward = self._tight_p2p(source_cloud, target_cloud, initial, self.voxel * 1.2, self.voxel * 0.6)
        entry_backward = self._tight_p2p(target_cloud, source_cloud, invert_transform(initial),
                                         self.voxel * 1.2, self.voxel * 0.6)
        # 两档逐步收紧的紧致精配
        for distance_scale, k_scale in ((0.8, 0.4), (0.4, 0.2)):
            suffix = f"d{distance_scale:.1f}"
            candidates.append((
                f"{label}:forward:{suffix}",
                self._tight_p2p(source_cloud, target_cloud, entry_forward,
                                self.voxel * distance_scale, self.voxel * k_scale),
            ))
            candidates.append((
                f"{label}:backward:{suffix}",
                # 反向精配的结果是 target->source，取逆换回 source->target
                invert_transform(self._tight_p2p(
                    target_cloud, source_cloud, entry_backward,
                    self.voxel * distance_scale, self.voxel * k_scale,
                )),
            ))
        return candidates

    # ---------- 局部网格锁定（local grid lock） ----------

    def _local_grid_lock(self, rotation_estimate: np.ndarray) -> list[np.ndarray]:
        """在一个旋转估计附近做确定性细网格搜索，并用高分辨率 FFT 重解平移。

        为什么需要：残余旋转误差 δ 会让相关最优的平移偏移约 radius·sin(δ)，也就是说
        旋转没锁到 ~1° 以内，平移就无法真正钉死。这里以 rotation_estimate 为中心，
        在三个坐标轴上以 step 为步长、±span 为范围做确定性三重网格枚举，只要真值旋转
        落在跨度内，网格就能保证找到与真值相差不超过约 0.7 步的旋转。选出相关分最高的
        若干个旋转，再用更细(dims=96、体素更小)的相关器精确重解平移，输出候选位姿。
        主要用于压制低重叠常见的“切向滑移”。
        """
        span = self.config.multi_lock_span_deg   # 网格半跨度(度)
        step = self.config.multi_lock_step_deg   # 网格步长(度)
        offsets = np.arange(-span, span + 1e-9, step)
        scored: list[tuple[float, np.ndarray]] = []
        # 三重循环：分别绕 X/Y/Z 轴叠加小角度偏移，构成局部旋转网格
        for offset_x in offsets:
            for offset_y in offsets:
                for offset_z in offsets:
                    rotation = rotation_estimate
                    # 依次绕三个坐标轴施加偏移（角度为 0 则跳过，省算力）
                    for axis, angle in zip(np.eye(3), (offset_x, offset_y, offset_z)):
                        if angle:
                            rotation = axis_angle_matrix(axis, np.radians(angle)) @ rotation
                    scored.append((self.forward.score(rotation), rotation))
        scored.sort(key=lambda item: -item[0])   # 按相关分降序
        # 延迟构建高分辨率相关器（体素更小、dims=96），用于更精确的平移求解
        if self.fine_forward is None:
            self.fine_forward = SignedCorrelationSolver(self.fast_source, self.fast_target, self.voxel,
                                                        dims=96, normal_radius=self.voxel * 3)
        # 取分数最高的前 multi_lock_top 个旋转，各自精解平移，得到候选位姿列表
        return [self.fine_forward.best_transform(rotation)
                for _, rotation in scored[: self.config.multi_lock_top]]

    # ---------- 候选指标与选择（candidate metrics and selection） ----------

    def _candidate_metrics(self, transform: np.ndarray) -> tuple[float, float, float, float, float]:
        """为一个候选位姿计算五项评估指标，供门控与 Borda 投票使用。

        返回 (对称拟合度, 粗穿透率, 细穿透率, 双向内点RMSE, 细拟合度)。
        RMSE 取正反两向的较大者(更保守)；某方向无内点则记为极大值 9e9 表示很差。
        细拟合度用更严的 0.8 体素阈值，能区分“真正贴合”与“勉强靠近”。
        """
        fitness, _ = self.scorer.symmetric_fitness(transform)
        violation = self.scorer.violation(transform)            # 粗尺度穿透率
        fine_violation = self.fine_scorer.violation(transform)  # 细尺度穿透率(更敏感)
        moved = apply_transform(self.fast_source, transform)
        forward, _ = self.target_tree.query(moved)
        backward, _ = NearestNeighborIndex(moved).query(self.fast_target)
        threshold = self.voxel * 2
        forward_in = forward[forward <= threshold]              # 正向内点
        backward_in = backward[backward <= threshold]           # 反向内点
        # RMSE 取双向较大者：任一方向对得差都会被暴露出来
        rmse = max(float(np.sqrt(np.mean(forward_in ** 2))) if len(forward_in) else 9e9,
                   float(np.sqrt(np.mean(backward_in ** 2))) if len(backward_in) else 9e9)
        # 细拟合度：0.8 体素严阈值下的双向内点比例取小值
        fine_fitness = min(float(np.mean(forward <= self.voxel * 0.8)),
                           float(np.mean(backward <= self.voxel * 0.8)))
        return fitness, violation, fine_violation, rmse, fine_fitness

    @staticmethod
    def _same_pose(first: np.ndarray, second: np.ndarray) -> bool:
        """判断两个位姿是否几乎相同（旋转差<0.1°且平移差<0.1mm），用于去重。"""
        rotation_delta = rotation_angle_deg(first[:3, :3] @ second[:3, :3].T)  # 相对旋转角
        translation_delta = float(np.linalg.norm(first[:3, 3] - second[:3, 3]))
        return rotation_delta < 0.1 and translation_delta < 0.0001

    def _deduplicate(self, candidates: list[tuple[str, np.ndarray]]) -> list[tuple[str, np.ndarray]]:
        """清洗候选列表：剔除形状非法/含非有限值的位姿，并去除重复位姿。"""
        unique: list[tuple[str, np.ndarray]] = []
        for label, transform in candidates:
            transform = np.asarray(transform, dtype=float)
            # 丢弃非 4x4 或含 NaN/Inf 的坏位姿
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                continue
            # 与已收集的位姿都不重复才加入
            if not any(self._same_pose(transform, other) for _, other in unique):
                unique.append((label, transform))
        return unique

    def _select(self, candidates: list[tuple[str, np.ndarray]]) -> tuple[GlobalCandidateMetrics, list[GlobalCandidateMetrics], bool]:
        """先按自由空间一致性做门控，再对四个指标做 Borda 排名聚合选出最优位姿。

        为什么用排名聚合而非单一指标：低重叠对的切向滑移会在几毫米外制造出“假最优”，
        它能在某个单一表面指标上胜过真值。综合“粗/细拟合度、细穿透率、内点RMSE”四个
        视角做 Borda 投票(各指标各自排名求和，总和越小越好)，比任何单一指标都稳。

        门控：穿透率≤阈值且拟合度≥下限的候选进入 gated 池；若无一通过则退回全体 rows。
        返回 (最优候选, 全部候选指标行, 是否有候选通过门控)。
        """
        # 对去重后的每个候选算全套指标，打包成 GlobalCandidateMetrics 行
        rows = [GlobalCandidateMetrics(label, transform, *self._candidate_metrics(transform))
                for label, transform in self._deduplicate(candidates)]
        if not rows:
            raise RuntimeError("global search produced no valid candidates")
        # 门控：同时满足“穿透率足够低”和“拟合度足够高”
        gated = [row for row in rows
                 if row.violation <= self.config.multi_violation_gate and row.fitness >= self.config.multi_min_fitness]
        for row in gated:
            row.gated = True
        pool = gated if gated else rows   # 优先在通过门控的候选里选；没有则放宽到全体
        # 四个排名键（都设计成“值越小越靠前”）：
        #   1) 拟合度−10×粗穿透  2) 拟合度−10×细穿透  3) 内点RMSE  4) 细拟合度−5×细穿透
        rank_keys = (lambda row: -(row.fitness - 10.0 * row.violation),
                     lambda row: -(row.fitness - 10.0 * row.fine_violation),
                     lambda row: row.rmse,
                     lambda row: -(row.fine_fitness - 5.0 * row.fine_violation))
        totals = np.zeros(len(pool))
        # Borda 计分：每个指标各自排序，名次(0,1,2,...)累加到各候选的总分上
        for key in rank_keys:
            order = sorted(range(len(pool)), key=lambda i: key(pool[i]))
            for rank, index in enumerate(order):
                totals[index] += rank
        for index, total in enumerate(totals):
            pool[index].borda_score = float(total)
        best = pool[int(np.argmin(totals))]   # 名次总和最小者胜出
        return best, rows, bool(gated)

    def run(self, extra_candidates: Mapping[str, np.ndarray] | None = None) -> GlobalSearchResult:
        """主流程：执行完整的五阶段全局搜索，返回最终位姿及诊断信息。

        extra_candidates 是外部提供的额外初始位姿(如其他算法给的先验)，会一并纳入
        精配与选择流程。各阶段耗时记录在 timings 里(毫秒)。
        """
        total_started = perf_counter()
        timings: dict[str, float] = {}

        # ===== 阶段一：SO(3) 均匀旋转网格粗搜 =====
        started = perf_counter()
        rotations = super_fibonacci_rotations(self.config.multi_grid_rotations)  # 均匀撒一批旋转
        scores = np.array([self._biscore(rotation) for rotation in rotations])   # 每个旋转的双向相关分
        timings["rotation_grid"] = (perf_counter() - started) * 1000
        order = np.argsort(-scores)                          # 按分数从高到低排序
        quats = Rotation.from_matrix(rotations).as_quat()    # 转四元数，便于计算旋转间夹角

        # 选取多样化的高分种子：贪心地从高分往下挑，且新种子与已选种子的旋转夹角需>15°，
        # 避免种子扎堆在同一个局部最优附近
        seeds: list[tuple[float, np.ndarray]] = []
        seed_quats: list[np.ndarray] = []
        for index in order:
            if len(seeds) >= self.config.multi_seed_count:
                break
            quat = quats[index]
            # 四元数点积->旋转夹角(度)：2·arccos(|q1·q2|)；对所有已选种子都>15°才接受
            if all(2 * np.degrees(np.arccos(min(1.0, abs(float(quat @ other))))) > 15.0
                   for other in seed_quats):
                seeds.append((float(scores[index]), rotations[index]))
                seed_quats.append(quat)

        # ===== 阶段二：对每个种子做随机爬山，优化旋转分 =====
        started = perf_counter()
        climbed = [self._hill_climb(rotation, score, self.config.multi_climb_rounds)
                   for score, rotation in seeds]
        climbed.sort(key=lambda item: -item[1])   # 按爬山后分数降序
        timings["hill_climb"] = (perf_counter() - started) * 1000

        # 从爬山结果里再挑多样化的一批进入精修池（旋转夹角需>10°去重）
        refine_pool: list[np.ndarray] = []
        picked_quats: list[np.ndarray] = []
        for rotation, _ in climbed:
            if len(refine_pool) >= self.config.multi_refine_seeds:
                break
            quat = Rotation.from_matrix(rotation).as_quat()
            if all(2 * np.degrees(np.arccos(min(1.0, abs(float(quat @ other))))) > 10.0
                   for other in picked_quats):
                refine_pool.append(rotation)
                picked_quats.append(quat)

        # 预备 Open3D 稠密点云(带法向)，供紧致 point-to-plane 收尾使用
        source_cloud = self._make_pcd(self.source)
        target_cloud = self._make_pcd(self.target)
        candidates: list[tuple[str, np.ndarray]] = []

        # ===== 阶段三+四：对每个精修种子做目标爬升精修，再紧致收尾，产出候选 =====
        started = perf_counter()
        for index, rotation in enumerate(refine_pool):
            refined = self._fitness_refine(rotation)             # 拟合度-穿透目标精修
            label = f"global:{index}:refined"
            candidates.append((label, refined))
            candidates.extend(self._finish_candidates(source_cloud, target_cloud, refined, label))

        # 把外部提供的额外初始位姿也纳入：原始位姿、其派生的紧致候选，以及用其旋转经 FFT
        # 重解平移+ICP 得到的位姿及其派生候选
        for label, transform in (extra_candidates or {}).items():
            external_label = f"external:{label}"
            candidates.append((f"{external_label}:raw", transform))
            candidates.extend(self._finish_candidates(source_cloud, target_cloud, transform, external_label))
            fft_pose = self._short_icp(self.forward.best_transform(np.asarray(transform)[:3, :3]))
            fft_label = f"{external_label}:fft"
            candidates.append((fft_label, fft_pose))
            candidates.extend(self._finish_candidates(source_cloud, target_cloud, fft_pose, fft_label))
        timings["pose_refinement"] = (perf_counter() - started) * 1000
        # 第一阶段选择：先选出一个当前最优，用它的旋转作为局部网格锁定的中心
        stage_one, _, _ = self._select(candidates)

        # ===== 阶段五（前置）：局部网格锁定，专门压制切向滑移 =====
        # 网格锁定：围绕第一阶段旋转做确定性细网格 + 细FFT平移，压住切向滑移
        started = perf_counter()
        for index, start in enumerate(self._local_grid_lock(stage_one.transform[:3, :3])):
            label = f"local_lock:{index}"
            candidates.append((label, start))
            candidates.extend(self._finish_candidates(source_cloud, target_cloud, start, label))
        timings["local_grid_lock"] = (perf_counter() - started) * 1000

        # ===== 阶段五：在全部候选(含网格锁定新增)上做最终门控+Borda 选择 =====
        started = perf_counter()
        selected, rows, gate = self._select(candidates)
        timings["final_selection"] = (perf_counter() - started) * 1000
        timings["total"] = (perf_counter() - total_started) * 1000
        gate_state = "pass" if gate else "violated"
        # 汇总一条人类可读的诊断信息(拟合度/穿透率/门控状态/胜出候选标签)
        message = (
            f"global_search fitness={selected.fitness:.3f} violation={selected.violation:.3f} "
            f"gate={gate_state} candidate={selected.label}"
        )
        # 打包返回：最终位姿、粗位姿(stage_one)、各项指标、计时与全部候选诊断行
        return GlobalSearchResult(
            selected.transform,
            selected.label,
            stage_one.transform,
            stage_one.label,
            selected.fitness,
            selected.violation,
            selected.fine_violation,
            gate,
            message,
            timings,
            rows,
        )
