"""数据模型模块：定义配准的配置、单轮迭代记录和最终结果。

这里用 dataclass 集中管理所有可调参数（RegistrationConfig）、ICP 每轮的过程量
（ICPRecord）以及一次配准的完整输出（RegistrationResult），让 pipeline 各环节
通过统一的数据结构传参，便于复现实验和记录报告。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

# 可选的粗配准方法与精配准方法名（用 Literal 约束取值，防止拼错）。
CoarseMethod = Literal["none", "pca", "fpfh", "fpfh_multiscale", "gcransac", "geotransformer"]
FineMethod = Literal["custom_icp", "point_to_plane"]


@dataclass(slots=True)
class RegistrationConfig:
    """配准全流程的配置项集合（含粗配准、ICP、全局搜索等所有可调参数）。"""

    coarse_method: CoarseMethod = "fpfh"          # 粗配准方法
    fine_method: FineMethod = "custom_icp"        # 精配准方法
    voxel_size: float = 0.0025                    # 体素下采样尺度，也是特征/匹配半径的基准
    max_correspondence_distance: float = 0.01     # ICP 最大对应距离（超过则不算内点）
    trim_fraction: float = 0.8                    # Trimmed ICP 保留比例（保留最近的 80%）
    max_iterations: int = 60                      # ICP 最大迭代次数
    rmse_tolerance: float = 1e-7                  # RMSE 变化收敛容差
    transform_tolerance: float = 1e-7             # 位姿增量收敛容差
    min_correspondences: int = 20                 # 求解所需的最少对应点数
    remove_outliers: bool = False                 # 预处理是否做统计离群点剔除
    random_seed: int = 42                         # 全局随机种子（复现实验）
    geotransformer_checkpoint: str | None = None  # GeoTransformer 权重路径
    geotransformer_num_points: int = 717          # GeoTransformer 输入采样点数
    success_rotation_deg: float = 5.0             # 判定成功的旋转误差上限（度）
    success_translation_ratio: float = 0.02       # 判定成功的平移误差比例上限（相对包围盒对角线）
    adaptive_trim: bool = False                   # 是否启用自适应裁剪
    min_trim_fraction: float = 0.35               # 自适应裁剪时的最小保留比例
    # 以下为全局搜索（multi-*）相关参数，用于在解空间做旋转网格搜索 + 爬山 + 精修。
    multi_score_points: int = 1200                # 全局搜索打分时采样点数
    multi_grid_rotations: int = 4000              # 旋转网格候选数量
    multi_seed_count: int = 20                    # 保留进入爬山的种子数
    multi_climb_rounds: int = 25                  # 爬山迭代轮数
    multi_refine_seeds: int = 6                   # 进入精修的种子数
    multi_refine_samples: int = 20                # 精修阶段的采样数
    multi_violation_gate: float = 0.016           # 违背（穿模等）约束的门限
    multi_min_fitness: float = 0.08               # 可接受候选的最小 fitness
    multi_lock_span_deg: float = 4.0              # 锁定精修的角度搜索范围（度）
    multi_lock_step_deg: float = 1.3              # 锁定精修的角度步长（度）
    multi_lock_top: int = 6                       # 锁定精修保留的候选数

    def validate(self) -> None:
        """校验各参数取值合法，非法时抛 ValueError 尽早暴露配置错误。"""
        if self.voxel_size < 0 or self.max_correspondence_distance <= 0:
            raise ValueError("voxel_size must be >= 0 and correspondence distance must be > 0")
        if not 0 < self.trim_fraction <= 1:
            raise ValueError("trim_fraction must be in (0, 1]")
        if self.max_iterations < 1 or self.min_correspondences < 3:
            raise ValueError("max_iterations must be >= 1 and min_correspondences >= 3")
        if self.geotransformer_num_points < 3:
            raise ValueError("geotransformer_num_points must be >= 3")
        if self.adaptive_trim and not 0 < self.min_trim_fraction <= self.trim_fraction:
            raise ValueError("min_trim_fraction must be in (0, trim_fraction]")
        if self.multi_grid_rotations < 100 or self.multi_refine_seeds < 1 or self.multi_score_points < 100:
            raise ValueError("invalid global-search settings")


@dataclass(slots=True)
class ICPRecord:
    """ICP 单轮迭代的过程记录，用于收敛曲线绘制和答辩讲解。"""

    iteration: int              # 迭代序号（从 1 开始）
    rmse: float                 # 本轮可信对应上的均方根误差
    correspondences: int        # 本轮参与求解的对应点数
    rotation_delta_deg: float   # 本轮位姿旋转增量（度）
    translation_delta: float    # 本轮位姿平移增量
    elapsed_ms: float           # 本轮耗时（毫秒）
    stage: str = "direct"       # 所处阶段标记（如 direct / refine）


@dataclass(slots=True)
class RegistrationResult:
    """一次完整配准的输出：最终位姿、成功标志、各类指标与耗时、迭代历史。"""

    transformation: np.ndarray = field(default_factory=lambda: np.eye(4))  # 最终 4x4 变换
    success: bool = False                                    # 是否达到成功判据
    status: str = "not_started"                              # 状态：converged/failed/...
    message: str = ""                                        # 说明信息
    metrics: dict[str, float] = field(default_factory=dict)  # 各项指标（fitness、误差等）
    timings_ms: dict[str, float] = field(default_factory=dict)  # 各阶段耗时（毫秒）
    history: list[ICPRecord] = field(default_factory=list)   # ICP 迭代历史
    source_points: int = 0                                   # 源点云原始点数
    target_points: int = 0                                   # 目标点云原始点数

    def to_dict(self) -> dict[str, Any]:
        """转成普通字典（便于 JSON 序列化）；numpy 矩阵转为嵌套列表。"""
        data = asdict(self)
        data["transformation"] = self.transformation.tolist()
        return data
