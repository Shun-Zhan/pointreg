"""配准主流程（pipeline）模块。

register_pair 把整条配准流水线串起来：读点云 -> 预处理（下采样/去噪）->
粗配准估初值 -> 精配准（ICP）细化 -> 计算指标与误差 -> 判定成功并记录各阶段耗时。
这是整个项目的总入口，各具体算法分散在 coarse / icp / geotransformer 等模块中。
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

from .coarse import fpfh_registration, gcransac_fpfh_registration, multiscale_fpfh_registration, pca_registration
from .geotransformer import geotransformer_registration
from .icp import custom_icp
from .io import read_points
from .metrics import alignment_metrics, pose_errors
from .models import RegistrationConfig, RegistrationResult
from .preprocessing import bounding_box_diagonal, preprocess_points
from .runtime import preload_open3d


def _point_to_plane(source: np.ndarray, target: np.ndarray, initial: np.ndarray, config: RegistrationConfig) -> tuple[np.ndarray, str, str]:
    """point-to-plane ICP 精配准（调用 Open3D 实现），作为自研 ICP 的备选。

    与点到点 ICP 相比，点到面 ICP 把误差投影到目标表面法向上，通常收敛更快、
    对平面结构更稳。需要目标点云的法向量，故先估计法向量再调用。

    返回：(最终变换, 状态, 说明信息)。
    """
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("point-to-plane ICP requires Open3D") from exc
    source_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source))
    target_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
    # 点到面 ICP 需要目标法向量；搜索半径取体素与对应距离的较大者。
    target_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max(config.voxel_size * 2, config.max_correspondence_distance), max_nn=30))
    result = o3d.pipelines.registration.registration_icp(
        source_cloud, target_cloud, config.max_correspondence_distance, initial,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=config.max_iterations),
    )
    # fitness > 0 说明存在有效对应，视为收敛，否则判失败。
    status = "converged" if result.fitness > 0 else "failed"
    return np.asarray(result.transformation), status, f"Open3D fitness={result.fitness:.4f}"


def register_pair(source: str | Path | np.ndarray, target: str | Path | np.ndarray, config: RegistrationConfig | None = None, *, ground_truth: np.ndarray | None = None, initial: np.ndarray | None = None) -> RegistrationResult:
    """配准两帧点云的总入口：读入 -> 预处理 -> 粗配准 -> 精配准 -> 评估。

    参数：
        source / target: 源、目标点云，可以是文件路径，也可以直接是 (N, 3) 数组。
        config: 配准配置；为空则用默认配置。
        ground_truth: 真值变换（可选）。给了它就用位姿误差判定成功，否则用 fitness。
        initial: 初始位姿（可选）。给了它就跳过粗配准，直接用它作为 ICP 初值。
    返回：
        RegistrationResult，含最终位姿、成功标志、各项指标与分阶段耗时。
    整个过程包在 try 里，任何异常都会被捕获并记为 failed，保证接口始终返回结果对象。
    """
    config = config or RegistrationConfig()
    config.validate()   # 先校验配置合法
    result = RegistrationResult()
    # 预热 Open3D（首次导入较慢），单独计时避免污染后续阶段耗时。
    result.timings_ms["runtime_warmup"] = preload_open3d()
    total_start = perf_counter()
    try:
        # ---- 阶段 1：加载点云 ----
        started = perf_counter()
        # 是路径就读文件，否则视作已有数组直接转 float。
        source_raw = read_points(source) if isinstance(source, (str, Path)) else np.asarray(source, dtype=float)
        target_raw = read_points(target) if isinstance(target, (str, Path)) else np.asarray(target, dtype=float)
        result.timings_ms["load"] = (perf_counter() - started) * 1000
        result.source_points, result.target_points = len(source_raw), len(target_raw)
        if len(source_raw) < 3 or len(target_raw) < 3:
            raise ValueError("both point clouds must contain at least three points")

        # ---- 阶段 2：预处理（体素下采样，可选去离群点）----
        started = perf_counter()
        source_points = preprocess_points(source_raw, config.voxel_size, config.remove_outliers)
        target_points = preprocess_points(target_raw, config.voxel_size, config.remove_outliers)
        result.timings_ms["preprocess"] = (perf_counter() - started) * 1000

        # ---- 阶段 3：粗配准（估计 ICP 初值）----
        # 若外部已给 initial 则用它，否则按配置选择粗配准方法。
        transform = np.eye(4) if initial is None else np.asarray(initial, dtype=float).copy()
        started = perf_counter()
        if initial is None and config.coarse_method == "pca":
            transform = pca_registration(source_points, target_points)
        elif initial is None and config.coarse_method == "fpfh":
            transform = fpfh_registration(source_points, target_points, config.voxel_size, config.random_seed)
        elif initial is None and config.coarse_method == "fpfh_multiscale":
            transform = multiscale_fpfh_registration(
                source_points, target_points, config.voxel_size, config.random_seed, config.max_correspondence_distance
            )
        elif initial is None and config.coarse_method == "gcransac":
            transform = gcransac_fpfh_registration(
                source_points, target_points, config.voxel_size, config.max_correspondence_distance
            )
        elif initial is None and config.coarse_method == "geotransformer":
            transform = geotransformer_registration(
                source_points,
                target_points,
                checkpoint=config.geotransformer_checkpoint,
                num_points=config.geotransformer_num_points,
                seed=config.random_seed,
            )
        # 注：coarse_method == "none" 时不进入任何分支，直接沿用单位/给定初值。
        result.timings_ms["coarse"] = (perf_counter() - started) * 1000

        # ---- 阶段 4：精配准（ICP 细化）----
        started = perf_counter()
        if config.fine_method == "custom_icp":
            transform, history, status, message = custom_icp(source_points, target_points, transform, config)
            result.history = history
        else:
            transform, status, message = _point_to_plane(source_points, target_points, transform, config)
        result.timings_ms["fine"] = (perf_counter() - started) * 1000
        result.transformation, result.status, result.message = transform, status, message

        # ---- 阶段 5：评估指标与成功判定 ----
        # 对齐质量指标（fitness、内点 RMSE 等）。
        result.metrics.update(alignment_metrics(source_points, target_points, transform, config.max_correspondence_distance))
        # 包围盒对角线：作为平移误差归一化的尺度基准。
        diagonal = bounding_box_diagonal(source_raw, target_raw)
        result.metrics["bbox_diagonal"] = diagonal
        if ground_truth is not None:
            # 有真值：用旋转误差 + 归一化平移误差判定是否成功。
            rotation_error, translation_error = pose_errors(transform, ground_truth)
            result.metrics.update(rotation_error_deg=rotation_error, translation_error=translation_error,
                                  translation_error_ratio=translation_error / diagonal if diagonal else float("inf"))
            result.success = status != "failed" and rotation_error < config.success_rotation_deg and result.metrics["translation_error_ratio"] < config.success_translation_ratio
        else:
            # 无真值：只能退而用 fitness > 0 作为“看起来对齐了”的弱判据。
            result.success = status != "failed" and result.metrics["fitness"] > 0
    except Exception as exc:
        # 任何环节出错都归为失败，并把异常信息记入 message，不向外抛。
        result.status, result.message, result.success = "failed", str(exc), False
    result.timings_ms["total"] = (perf_counter() - total_start) * 1000
    return result
