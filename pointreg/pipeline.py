"""单组两帧配准主流程：串起 加载→预处理→粗配准→精配准→评估 的完整链路。

register_pair 是整个工具包最核心的入口，app / cli / experiments 都通过它
（或桥接版 register_dataset_pair）来完成一次配准，并统一记录各阶段耗时与指标。
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

from .coarse import fpfh_registration, pca_registration
from .icp import custom_icp
from .io import read_points
from .metrics import alignment_metrics, pose_errors
from .models import RegistrationConfig, RegistrationResult
from .preprocessing import bounding_box_diagonal, preprocess_points
from .runtime import preload_open3d


def _point_to_plane(source: np.ndarray, target: np.ndarray, initial: np.ndarray, config: RegistrationConfig) -> tuple[np.ndarray, str, str]:
    """调用 Open3D 的点到面 ICP 作为精配准备选方案（需要 Open3D 与法向量）。

    点到面 ICP 用点到目标切平面的距离作误差，收敛通常比点到点更快更稳，
    但依赖法向量估计。返回 (变换, 状态, 说明)，与自研 ICP 的接口保持一致。
    """
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("point-to-plane ICP requires Open3D") from exc
    source_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source))
    target_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
    # 点到面需要目标法向量；估计半径取体素 2 倍与对应距离阈值中的较大者
    target_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max(config.voxel_size * 2, config.max_correspondence_distance), max_nn=30))
    result = o3d.pipelines.registration.registration_icp(
        source_cloud, target_cloud, config.max_correspondence_distance, initial,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=config.max_iterations),
    )
    status = "converged" if result.fitness > 0 else "failed"
    return np.asarray(result.transformation), status, f"Open3D fitness={result.fitness:.4f}"


def register_pair(source: str | Path | np.ndarray, target: str | Path | np.ndarray, config: RegistrationConfig | None = None, *, ground_truth: np.ndarray | None = None, initial: np.ndarray | None = None) -> RegistrationResult:
    """对两帧点云执行一次完整配准，返回包含位姿、指标与耗时的结果对象。

    参数 source/target 既可以是文件路径也可以是已加载的 (N,3) 数组；
    ground_truth 若给出则会计算旋转/平移误差并据此判定成功；
    initial 若给出则跳过粗配准、直接以它为 ICP 初值（桥接法/扰动实验会用到）。
    全程用 try/except 包裹：任何异常都转成 status="failed" 的结果而非抛出，
    方便批量实验中单个点对失败时不中断整体流程。
    """
    config = config or RegistrationConfig()
    config.validate()
    result = RegistrationResult()
    # 预热 Open3D（首次导入较慢），把冷启动耗时单独计入，不污染各阶段计时
    result.timings_ms["runtime_warmup"] = preload_open3d()
    total_start = perf_counter()
    try:
        # —— 阶段 1：加载点云 ——
        started = perf_counter()
        source_raw = read_points(source) if isinstance(source, (str, Path)) else np.asarray(source, dtype=float)
        target_raw = read_points(target) if isinstance(target, (str, Path)) else np.asarray(target, dtype=float)
        result.timings_ms["load"] = (perf_counter() - started) * 1000
        result.source_points, result.target_points = len(source_raw), len(target_raw)
        if len(source_raw) < 3 or len(target_raw) < 3:
            raise ValueError("both point clouds must contain at least three points")
        # —— 阶段 2：预处理（体素下采样，可选离群点剔除）——
        started = perf_counter()
        source_points = preprocess_points(source_raw, config.voxel_size, config.remove_outliers)
        target_points = preprocess_points(target_raw, config.voxel_size, config.remove_outliers)
        result.timings_ms["preprocess"] = (perf_counter() - started) * 1000

        # —— 阶段 3：粗配准（给出 initial 时跳过，直接用外部初值）——
        transform = np.eye(4) if initial is None else np.asarray(initial, dtype=float).copy()
        started = perf_counter()
        if initial is None and config.coarse_method == "pca":
            transform = pca_registration(source_points, target_points)
        elif initial is None and config.coarse_method == "fpfh":
            transform = fpfh_registration(source_points, target_points, config.voxel_size, config.random_seed)
        result.timings_ms["coarse"] = (perf_counter() - started) * 1000

        # 记录“粗配准后”的对齐指标，供批量实验对比粗/精两阶段效果。
        # 注意：这是几何层面的对齐度量，不是 RANSAC 内部的内点集合。
        coarse_metrics = alignment_metrics(
            source_points, target_points, transform, config.max_correspondence_distance
        )
        result.metrics.update({f"coarse_{key}": value for key, value in coarse_metrics.items()})
        if ground_truth is not None:
            # 有真值时额外记录粗配准阶段的旋转/平移误差
            coarse_rotation_error, coarse_translation_error = pose_errors(transform, ground_truth)
            coarse_diagonal = bounding_box_diagonal(source_raw, target_raw)
            result.metrics.update(
                coarse_rotation_error_deg=coarse_rotation_error,
                coarse_translation_error=coarse_translation_error,
                coarse_translation_error_ratio=(
                    coarse_translation_error / coarse_diagonal if coarse_diagonal else float("inf")
                ),
            )

        # —— 阶段 4：精配准（自研 ICP 或 Open3D 点到面）——
        started = perf_counter()
        if config.fine_method == "custom_icp":
            transform, history, status, message = custom_icp(source_points, target_points, transform, config)
            result.history = history  # 自研 ICP 会返回逐轮历史用于可视化
        else:
            transform, status, message = _point_to_plane(source_points, target_points, transform, config)
        result.timings_ms["fine"] = (perf_counter() - started) * 1000
        result.transformation, result.status, result.message = transform, status, message
        # —— 阶段 5：最终评估 ——
        result.metrics.update(alignment_metrics(source_points, target_points, transform, config.max_correspondence_distance))
        diagonal = bounding_box_diagonal(source_raw, target_raw)
        result.metrics["bbox_diagonal"] = diagonal
        if ground_truth is not None:
            # 有真值：按“旋转误差<阈值 且 相对平移误差<阈值”判定是否成功
            rotation_error, translation_error = pose_errors(transform, ground_truth)
            result.metrics.update(rotation_error_deg=rotation_error, translation_error=translation_error,
                                  translation_error_ratio=translation_error / diagonal if diagonal else float("inf"))
            result.success = status != "failed" and rotation_error < config.success_rotation_deg and result.metrics["translation_error_ratio"] < config.success_translation_ratio
        else:
            # 无真值：只能退而用 fitness>0（有重合）作为粗略的成功标志
            result.success = status != "failed" and result.metrics["fitness"] > 0
    except Exception as exc:
        # 任何异常都转为失败结果返回，保证批量实验不因单点对报错而中断
        result.status, result.message, result.success = "failed", str(exc), False
    result.timings_ms["total"] = (perf_counter() - total_start) * 1000
    return result
