"""配准流程共用的数据模型：配置、逐轮 ICP 记录、以及最终结果。

用 dataclass 集中定义参数与输出，让 pipeline / dataset / experiments / app
各处传递统一的对象，避免到处传散乱的关键字参数。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

# 粗配准可选方法：none=不做粗配 / pca=主轴对齐 / fpfh=FPFH 特征+RANSAC
CoarseMethod = Literal["none", "pca", "fpfh"]
# 精配准可选方法：自研点到点 ICP / Open3D 点到面 ICP
FineMethod = Literal["custom_icp", "point_to_plane"]


@dataclass(slots=True)
class RegistrationConfig:
    """一次配准运行的全部可调参数（默认值对应课程设计的推荐设置）。"""

    coarse_method: CoarseMethod = "fpfh"          # 粗配准方法
    fine_method: FineMethod = "custom_icp"        # 精配准方法
    voxel_size: float = 0.0025                    # 体素下采样尺寸（米）
    max_correspondence_distance: float = 0.01     # 判定对应点的最大距离阈值
    trim_fraction: float = 0.8                    # Trimmed-ICP 保留的对应点比例
    max_iterations: int = 60                      # ICP 最大迭代次数
    rmse_tolerance: float = 1e-7                  # RMSE 变化收敛阈值
    transform_tolerance: float = 1e-7             # 位姿增量收敛阈值
    min_correspondences: int = 20                 # 有效对应点数下限，低于则判失败
    remove_outliers: bool = False                 # 是否做统计离群点剔除
    random_seed: int = 42                         # RANSAC 等随机过程的种子
    success_rotation_deg: float = 5.0             # 成功判据：旋转误差上限（度）
    success_translation_ratio: float = 0.02       # 成功判据：平移误差占包围盒对角线比例上限

    def validate(self) -> None:
        """在正式运行前校验参数合法性，尽早暴露配置错误。"""
        if self.voxel_size < 0 or self.max_correspondence_distance <= 0:
            raise ValueError("voxel_size must be >= 0 and correspondence distance must be > 0")
        if not 0 < self.trim_fraction <= 1:
            raise ValueError("trim_fraction must be in (0, 1]")
        if self.max_iterations < 1 or self.min_correspondences < 3:
            raise ValueError("max_iterations must be >= 1 and min_correspondences >= 3")


@dataclass(slots=True)
class ICPRecord:
    """ICP 单轮迭代的日志，用于绘制收敛曲线和分析每步位姿增量。"""

    iteration: int              # 迭代序号（从 1 开始）
    rmse: float                 # 本轮对应点的均方根误差
    correspondences: int        # 本轮参与求解的有效对应点数
    rotation_delta_deg: float   # 本轮位姿增量中的旋转角（度）
    translation_delta: float    # 本轮位姿增量中的平移量
    elapsed_ms: float           # 本轮耗时（毫秒）
    stage: str = "direct"       # 所属阶段；桥接法中记录当前是哪条边


@dataclass(slots=True)
class RegistrationResult:
    """一次配准的完整输出：位姿、成功与否、各类指标、耗时与迭代历史。"""

    transformation: np.ndarray = field(default_factory=lambda: np.eye(4))  # 估计的源→目标 4×4 变换
    success: bool = False                                    # 是否满足成功判据
    status: str = "not_started"                              # 运行状态：converged/failed/max_iterations 等
    message: str = ""                                        # 人类可读的说明信息
    metrics: dict[str, float] = field(default_factory=dict)  # 各类量化指标（rmse/fitness/误差等）
    timings_ms: dict[str, float] = field(default_factory=dict)  # 各阶段耗时（毫秒）
    history: list[ICPRecord] = field(default_factory=list)   # 逐轮 ICP 记录
    source_points: int = 0                                   # 源点云原始点数
    target_points: int = 0                                   # 目标点云原始点数

    def to_dict(self) -> dict[str, Any]:
        """转成可 JSON 序列化的字典（把 ndarray 变换矩阵转为嵌套列表）。"""
        data = asdict(self)
        data["transformation"] = self.transformation.tolist()
        return data
