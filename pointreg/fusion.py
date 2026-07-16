# 中文说明（课程设计讲解）：
# 本模块是整个低重叠配准方案的“融合总控”。它把三条互补的线索组合起来：
#   1) GeoTransformer 深度模型给出的稠密对应点（语义/几何理解强）；
#   2) 手工全局搜索（4000 个旋转的 FFT 搜索 + 自由空间选优，覆盖大角度差）；
#   3) 鲁棒后端（GC-RANSAC / RANSAC + FPFH）从对应点里稳健拟合位姿。
# 先用 GeoTransformer 的对应点生成一批“种子位姿”候选，再交给全局搜索择优，
# 从而在两帧重叠很小、初始位姿差很大的困难情况下也能配准成功。
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np

from .coarse import fpfh_registration, gcransac_from_correspondences
from .geotransformer import (
    DEFAULT_3DMATCH_CHECKPOINT,
    GeoTransformerCorrespondences,
    geotransformer_3dmatch_correspondences,
)
from .global_search import GlobalRegistrationSearch, GlobalSearchResult
from .metrics import alignment_metrics, pose_errors
from .models import RegistrationConfig, RegistrationResult
from .preprocessing import bounding_box_diagonal, preprocess_points
from .runtime import preload_open3d


# 进度回调类型：接收一个中文进度提示字符串，用于向前端/日志汇报当前阶段。
ProgressCallback = Callable[[str], None]


def build_fusion_seed_candidates(
    output: GeoTransformerCorrespondences,
    source: np.ndarray,
    target: np.ndarray,
    voxel_size: float,
) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, str]]:
    """Build LGR, multi-run GC-RANSAC and FPFH seeds for global fusion.

    中文说明：从 GeoTransformer 的输出与原始点云出发，生成一批“种子位姿候选”，
    交给后续全局搜索择优。候选来自三类来源：
      1) LGR：GeoTransformer 自带的位姿估计，直接作为一个候选；
      2) GC-RANSAC：对稠密对应点做多组鲁棒拟合（扫描不同距离阈值 + 不同随机种子），
         提高在外点多、重叠低时找到正确位姿的概率；
      3) FPFH：传统手工特征配准，作为与深度模型互补的一个候选。

    参数：
      output:     GeoTransformer 稠密对应点结果；
      source/target: 预处理后的源/目标点云；
      voxel_size: 体素大小（供 FPFH 使用）。
    返回：(候选位姿字典 candidates, 各候选内点数 inlier_counts, 失败原因 errors)。
    """
    # 先把 LGR 位姿放入候选（键名 "lgr"）。
    candidates: dict[str, np.ndarray] = {"lgr": output.lgr_transform}
    inlier_counts: dict[str, int] = {}
    errors: dict[str, str] = {}
    # 网格式扫描：3 个对应点距离阈值 × 3 个随机种子 = 9 组 GC-RANSAC，增强鲁棒性。
    for threshold in (0.008, 0.010, 0.012):
        for seed in (0, 1, 42):
            label = f"geotransformer_gcransac_t{threshold:.3f}_s{seed}"  # 候选的唯一标签
            # 用固定种子打乱对应点顺序：GC-RANSAC 对输入顺序敏感，打乱可获得更多样的解。
            permutation = np.random.default_rng(seed).permutation(len(output.source_points))
            try:
                # 用打乱后的对应点（及其得分作为权重）鲁棒拟合位姿，返回位姿与内点掩码。
                transform, inliers = gcransac_from_correspondences(
                    output.source_points[permutation],
                    output.target_points[permutation],
                    output.scores[permutation],
                    correspondence_distance=threshold,
                    max_iters=10000,
                    seed=seed,
                )
                candidates[label] = transform
                inlier_counts[label] = int(inliers.sum())  # 记录内点数，供分析/比较
            except Exception as exc:
                # 单组失败不影响整体，记下原因继续尝试其它组合。
                errors[label] = str(exc)
    # 再补一个传统 FPFH 配准候选；失败也不致命，记录错误即可。
    try:
        candidates["fpfh"] = fpfh_registration(source, target, voxel_size, seed=42)
    except Exception as exc:
        errors["fpfh"] = str(exc)
    return candidates, inlier_counts, errors


def register_low_overlap_pair(
    source: np.ndarray,
    target: np.ndarray,
    config: RegistrationConfig,
    *,
    checkpoint: str | Path | None = None,
    ground_truth: np.ndarray | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[RegistrationResult, dict[str, object]]:
    """Register a difficult pair with GeoTransformer-seeded global fusion.

    中文说明：本函数是端到端配准主流程，按顺序完成：预处理 → 提取深度对应 →
    生成种子候选 → 全局搜索择优 → 计算评估指标（含与真值的位姿误差）。
    全程用 timings_ms 记录各阶段耗时，用 progress 回调汇报进度，
    并用 try/except 兜底：任何环节异常都会把结果标记为 failed 而不崩溃。

    参数：
      source, target: 原始源/目标点云；
      config:         配准配置（体素大小、成功判定阈值等）；
      checkpoint:     GeoTransformer 权重路径（默认 3DMatch）；
      ground_truth:   可选的真值位姿，用于计算旋转/平移误差并判定是否成功；
      progress:       可选进度回调。
    返回：(RegistrationResult 配准结果, details 诊断信息字典)。
    """
    config.validate()  # 先校验配置合法
    progress = progress or (lambda _: None)  # 未提供回调时用空操作，简化后续调用
    result = RegistrationResult(source_points=len(source), target_points=len(target))
    total_started = perf_counter()  # 记录总起始时间
    details: dict[str, object] = {}
    try:
        # 预热 Open3D（首次导入较慢），并记录预热耗时。
        result.timings_ms["runtime_warmup"] = preload_open3d()
        source = np.asarray(source, dtype=float)
        target = np.asarray(target, dtype=float)
        # 预处理：体素下采样 + 可选离群点剔除，得到用于配准的干净点云。
        started = perf_counter()
        source_points = preprocess_points(source, config.voxel_size, config.remove_outliers)
        target_points = preprocess_points(target, config.voxel_size, config.remove_outliers)
        result.timings_ms["preprocess"] = (perf_counter() - started) * 1000

        # 阶段一：用 3DMatch GeoTransformer 提取稠密对应点。
        progress("正在提取 GeoTransformer 稠密对应…")
        started = perf_counter()
        output = geotransformer_3dmatch_correspondences(
            source_points,
            target_points,
            checkpoint=checkpoint or DEFAULT_3DMATCH_CHECKPOINT,
            voxel_size=config.voxel_size,
        )
        result.timings_ms["geotransformer"] = (perf_counter() - started) * 1000

        # 阶段二：由对应点生成 LGR / GC-RANSAC / FPFH 种子位姿候选。
        progress("正在生成 LGR、GC-RANSAC 与 FPFH 候选…")
        started = perf_counter()
        candidates, seed_inliers, seed_errors = build_fusion_seed_candidates(
            output, source_points, target_points, config.voxel_size
        )
        result.timings_ms["seed_candidates"] = (perf_counter() - started) * 1000

        # 阶段三：全局搜索。以上述候选为起点，做 4000 旋转的 FFT 搜索 + 自由空间选优，
        # 在所有候选中挑出对齐最好的最终位姿。
        progress("正在执行 4000 旋转 FFT 全局搜索与自由空间选优…")
        started = perf_counter()
        search = GlobalRegistrationSearch(source_points, target_points, config)
        search_result: GlobalSearchResult = search.run(candidates)
        result.timings_ms["global_fusion"] = (perf_counter() - started) * 1000

        # 采纳全局搜索选出的位姿，标记为已收敛。
        transform = search_result.transform
        result.transformation = transform
        result.status = "converged"
        result.message = search_result.message
        # 计算对齐质量指标（fitness、RMSE 等）。
        result.metrics.update(
            alignment_metrics(source_points, target_points, transform, config.max_correspondence_distance)
        )
        # 包围盒对角线长度：作为平移误差的归一化尺度（相对误差 = 平移误差 / 对角线）。
        diagonal = bounding_box_diagonal(source, target)
        result.metrics["bbox_diagonal"] = diagonal
        if ground_truth is not None:
            # 有真值：计算旋转误差(度)与平移误差，并换算成相对比例。
            rotation_error, translation_error = pose_errors(transform, ground_truth)
            ratio = translation_error / diagonal if diagonal else float("inf")
            result.metrics.update(
                rotation_error_deg=rotation_error,
                translation_error=translation_error,
                translation_error_ratio=ratio,
            )
            # 旋转和平移相对误差同时低于阈值才算成功。
            result.success = (
                rotation_error < config.success_rotation_deg
                and ratio < config.success_translation_ratio
            )
        else:
            # 无真值：退而用 fitness>0 作为“有有效对齐”的粗略成功判据。
            result.success = result.metrics.get("fitness", 0.0) > 0
        # 汇总诊断信息，便于答辩/调试时分析各候选表现。
        details = {
            "correspondence_count": len(output.source_points),
            "seed_inliers": seed_inliers,
            "seed_errors": seed_errors,
            "search": search_result.to_dict(),
        }
        progress("低覆盖 GeoTransformer 全局配准完成。")
    except Exception as exc:
        # 任一阶段出错：标记失败并记录错误信息，保证函数总能返回而不抛出。
        result.status = "failed"
        result.message = str(exc)
        result.success = False
        details = {"error": str(exc)}
    result.timings_ms["total"] = (perf_counter() - total_started) * 1000  # 记录总耗时
    return result, details
