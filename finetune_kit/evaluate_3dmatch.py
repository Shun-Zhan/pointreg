"""Evaluate the 3DMatch GeoTransformer checkpoint on low-overlap Bunny pairs.

The ``lgr`` backend preserves the original upstream pose path.  The ``robust``
backend re-solves the dense GeoTransformer correspondences with GC-RANSAC and
selects between LGR, GC-RANSAC and FPFH using geometry only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from itertools import permutations
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pointreg.coarse import (
    _geometric_candidate_score,
    fpfh_registration,
    gcransac_from_correspondences,
)
from pointreg.geotransformer import (
    GeoTransformerCorrespondences,
    geotransformer_3dmatch_correspondences,
)
from pointreg.fusion import build_fusion_seed_candidates
from pointreg.global_search import GlobalRegistrationSearch
from pointreg.icp import custom_icp
from pointreg.io import parse_bun_conf, read_points
from pointreg.metrics import alignment_metrics, pose_errors, symmetric_overlap
from pointreg.models import RegistrationConfig
from pointreg.preprocessing import bounding_box_diagonal, preprocess_points
from pointreg.transforms import apply_transform, relative_transform


# 课程设计默认要评测的四对(含从易到难的低重合样例)
DEFAULT_PAIRS = [
    ("bun000", "bun045"),
    ("bun000", "bun180"),
    ("chin", "top2"),
    ("bun000", "ear_back"),
]


def _two_stage_icp(
    source: np.ndarray,
    target: np.ndarray,
    initial: np.ndarray,
    *,
    voxel_size: float,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """两段式 ICP 精配:先用大对应距离粗调,再用小距离收紧,从粗到精逐步逼近。"""
    transform = np.asarray(initial, dtype=float).copy()
    stages: list[dict[str, object]] = []
    # 基础配置:不做粗配(coarse_method=none,初值来自外部候选),只做 custom_icp 精配
    base = RegistrationConfig(
        coarse_method="none",
        fine_method="custom_icp",
        voxel_size=voxel_size,
        max_iterations=30,
        trim_fraction=0.8,       # 每轮只保留 80% 最近的对应,抗离群
    )
    # 由粗到精:第一段允许 1.5cm 内匹配,第二段收紧到 0.8cm
    for distance in (0.015, 0.008):
        config = replace(base, max_correspondence_distance=distance)
        # 上一段的输出 transform 作为下一段的初值,逐级精化
        transform, history, status, message = custom_icp(source, target, transform, config)
        stages.append(
            {
                "distance": distance,
                "status": status,
                "message": message,
                "iterations": len(history),
                "last_rmse": float(history[-1].rmse) if history else None,
            }
        )
    return transform, stages


def refine_and_select_candidates(
    source: np.ndarray,
    target: np.ndarray,
    candidates: dict[str, np.ndarray],
    *,
    voxel_size: float,
    score_distance: float,
) -> tuple[str, np.ndarray, list[dict[str, object]]]:
    """Refine and rank candidate poses without consulting ground truth.

    对每个候选位姿做精配,并只用几何指标(内点数、误差)排序打分,
    全程不看真值——这是评测公平性的关键:选择哪个候选靠的是几何自洽,而非答案。
    """
    if not candidates:
        raise ValueError("at least one registration candidate is required")
    rows: list[dict[str, object]] = []
    best_key: tuple[int, int, float] | None = None
    best_name = ""
    best_transform: np.ndarray | None = None
    for name, coarse in candidates.items():
        # 每个候选先精配,再算几何打分:互近邻内点数、双向内点数、中位误差
        refined, stages = _two_stage_icp(source, target, coarse, voxel_size=voxel_size)
        reciprocal, total, error = _geometric_candidate_score(source, target, refined, score_distance)
        # 排序键:互近邻内点越多越好,双向内点越多越好,误差越小越好(取负号)
        key = (reciprocal, total, -error)
        rows.append(
            {
                "name": name,
                "coarse_transform": np.asarray(coarse, dtype=float).tolist(),
                "refined_transform": refined.tolist(),
                "reciprocal_inliers": reciprocal,
                "bidirectional_inliers": total,
                "median_error": error if np.isfinite(error) else None,
                "icp_stages": stages,
            }
        )
        # 字典比较逐元素进行:优于当前最优就更新
        if best_key is None or key > best_key:
            best_key, best_name, best_transform = key, name, refined
    assert best_transform is not None
    return best_name, best_transform, rows


def _correspondence_diagnostics(
    output: GeoTransformerCorrespondences,
    ground_truth: np.ndarray,
    source_cloud: np.ndarray,
    target_cloud: np.ndarray,
) -> dict[str, object]:
    """诊断 GeoTransformer 给出的对应点质量:数量、分数、尺度、包围盒范围,以及真值内点数。

    这些信息用于事后分析"为什么某对配准失败"——例如对应太少、或对应只集中在
    局部区域(extent_ratio 偏小),说明网络没在整体上找到可靠匹配。
    """
    # 用真值把源对应点变换过去,与目标对应点比残差:残差小说明该对应是真内点
    residuals = np.linalg.norm(
        apply_transform(output.source_points, ground_truth) - output.target_points,
        axis=1,
    )
    return {
        "count": len(output.source_points),
        "score_min": float(output.scores.min()) if len(output.scores) else None,
        "score_max": float(output.scores.max()) if len(output.scores) else None,
        "scale": output.scale,
        "source_min": output.source_points.min(axis=0).tolist(),
        "source_max": output.source_points.max(axis=0).tolist(),
        "target_min": output.target_points.min(axis=0).tolist(),
        "target_max": output.target_points.max(axis=0).tolist(),
        "network_source_min": (output.source_points * output.scale).min(axis=0).tolist(),
        "network_source_max": (output.source_points * output.scale).max(axis=0).tolist(),
        "network_target_min": (output.target_points * output.scale).min(axis=0).tolist(),
        "network_target_max": (output.target_points * output.scale).max(axis=0).tolist(),
        "source_cloud_min": source_cloud.min(axis=0).tolist(),
        "source_cloud_max": source_cloud.max(axis=0).tolist(),
        "target_cloud_min": target_cloud.min(axis=0).tolist(),
        "target_cloud_max": target_cloud.max(axis=0).tolist(),
        "source_corr_extent_ratio": float(
            np.linalg.norm(np.ptp(output.source_points, axis=0))
            / max(np.linalg.norm(np.ptp(source_cloud, axis=0)), 1e-12)
        ),
        "target_corr_extent_ratio": float(
            np.linalg.norm(np.ptp(output.target_points, axis=0))
            / max(np.linalg.norm(np.ptp(target_cloud, axis=0)), 1e-12)
        ),
        # 真值内点数:残差在 5mm / 10mm 内的对应数,直接反映网络匹配的可用程度
        "gt_inliers_0.005": int((residuals <= 0.005).sum()),
        "gt_inliers_0.010": int((residuals <= 0.010).sum()),
    }


def _stage_a_decision(rows: list[dict[str, object]]) -> dict[str, object]:
    """针对最难的 bun000->bun180 对,做"阶段A"诊断并给出下一步动作建议。

    逻辑:若已成功就停止;若接近成功(旋转达标、平移比在 2%~3% 边缘)则建议
    启用阶段C的 Fusion 融合;否则进一步定位失败根因(对应不足/GC-RANSAC 失败等)。
    """
    hard = next(row for row in rows if row["pair"] == "bun000->bun180")
    rotation = float(hard["final_rot"])
    translation = float(hard["final_tr_ratio"])
    # 触发 Fusion 的条件:旋转已达标,但平移比卡在 2%~3% 的"差一口气"区间
    trigger_fusion = rotation < 5.0 and 0.02 <= translation <= 0.03
    # 根据是否成功 / 是否触发条件,决定下一步动作
    if bool(hard["success_2pct"]):
        action = "stage_a_succeeded_stop"
    elif trigger_fusion:
        action = "activate_stage_c_fusion"
    else:
        action = "diagnose_only_do_not_activate_stage_c"
    diagnostics = hard["correspondences"]
    # 找出 GC-RANSAC 候选,用于下面逐级排查失败原因
    gc_candidate = next(
        (candidate for candidate in hard["candidates"] if candidate["name"] == "geotransformer_gcransac"),
        None,
    )
    # 从"最上游"到"最下游"逐层归因:对应太少 -> 求解失败 -> 结构化外点压倒真值 ->
    # ICP 跑出正确收敛域 -> 仅几何的候选选择被"冒充者"骗了
    if int(diagnostics["gt_inliers_0.010"]) < 3:
        cause = "insufficient_geotransformer_correspondences"
    elif gc_candidate is None:
        cause = "gcransac_failed_to_return_a_pose"
    elif float(gc_candidate["coarse_pose"]["rot"]) >= 5.0:
        cause = "structured_outliers_overpowered_true_correspondence_consensus"
    elif float(gc_candidate["refined_pose"]["rot"]) >= 5.0:
        cause = "icp_left_the_correct_convergence_basin"
    else:
        cause = "geometry_only_candidate_selection_preferred_an_impostor"
    return {
        "stage": "A",
        "pair": hard["pair"],
        "trigger_fusion": trigger_fusion,
        "action": action,
        "diagnosed_cause": cause,
        "final_rot": rotation,
        "final_tr_ratio": translation,
        "selected_candidate": hard["selected_candidate"],
        "gcransac_inliers": hard["gcransac_inliers"],
        "correspondences": diagnostics,
        "candidate_errors": hard["candidate_errors"],
        "candidates": hard["candidates"],
    }


def _pose_metrics(transform: np.ndarray, ground_truth: np.ndarray, diagonal: float) -> dict[str, float]:
    """算位姿误差:旋转误差(度)和平移误差比(平移误差/包围盒对角线,做尺度归一化)。"""
    rotation, translation = pose_errors(transform, ground_truth)
    return {
        "rot": round(float(rotation), 3),
        "tr_ratio": round(float(translation / diagonal), 5),
    }


def evaluate_pair(
    source_name: str,
    target_name: str,
    *,
    data_dir: Path,
    checkpoint: Path,
    poses: dict[str, np.ndarray],
    voxel_size: float,
    distance: float,
    backend: str,
) -> dict[str, object]:
    """评测单对扫描的完整流程:算真值 -> 读点云 -> 预处理 -> GeoTransformer 出对应 -> 按 backend 求位姿 -> 精配 -> 打分。"""
    # 真值相对变换(仅用于最后算误差,不参与求解)
    ground_truth = relative_transform(poses[source_name], poses[target_name])
    raw_source = read_points(data_dir / f"{source_name}.ply")
    raw_target = read_points(data_dir / f"{target_name}.ply")
    # 用真值和距离阈值算这对的对称重合率,衡量任务难度
    overlap = symmetric_overlap(raw_source, raw_target, ground_truth, distance)
    # 体素下采样,统一到工作分辨率
    source = preprocess_points(raw_source, voxel_size)
    target = preprocess_points(raw_target, voxel_size)
    diagonal = bounding_box_diagonal(source, target)  # 包围盒对角线,用于平移误差归一化
    # 用 3DMatch 权重跑 GeoTransformer,得到稠密对应点及内置的 LGR 位姿
    output = geotransformer_3dmatch_correspondences(
        source,
        target,
        checkpoint=checkpoint,
        voxel_size=voxel_size,
    )
    diagnostics = _correspondence_diagnostics(output, ground_truth, source, target)
    errors: dict[str, str] = {}
    gc_inliers = 0
    global_search: dict[str, object] | None = None

    # ---- 三种后端:fusion(融合全局搜索)/ lgr(直接用官方 LGR 位姿)/ robust(多候选择优) ----
    if backend == "fusion":
        # 由 GeoTransformer 对应构造多个种子候选位姿,再交给全局搜索择优
        candidates, fusion_inliers, errors = build_fusion_seed_candidates(
            output, source, target, voxel_size
        )
        config = RegistrationConfig(
            coarse_method="none",
            voxel_size=voxel_size,
            max_correspondence_distance=distance,
            adaptive_trim=True,   # 自适应裁剪,适应低重合
        )
        started = perf_counter()
        search = GlobalRegistrationSearch(source, target, config)
        initialization_ms = (perf_counter() - started) * 1000  # 记录初始化耗时
        result = search.run(candidates)  # 在候选 + 全局旋转网格上搜索最优位姿
        global_search = result.to_dict()
        global_search["timings_ms"]["initialization"] = initialization_ms
        global_search["seed_inliers"] = fusion_inliers
        # 给每个候选补上相对真值的位姿误差,便于事后分析
        for candidate in global_search["candidates"]:
            candidate["pose"] = _pose_metrics(
                np.asarray(candidate["transform"]), ground_truth, diagonal
            )
        coarse = result.coarse_transform
        final = result.transform
        selected = result.selected_label
        candidate_rows = global_search["candidates"]
    elif backend == "lgr":
        # 最简单的路径:直接采用 GeoTransformer 官方的 LGR 位姿,再做一次 ICP 精配
        coarse = output.lgr_transform
        config = RegistrationConfig(
            coarse_method="none",
            voxel_size=voxel_size,
            max_correspondence_distance=distance,
        )
        final, history, status, message = custom_icp(source, target, coarse, config)
        selected = "lgr"
        candidate_rows = [
            {
                "name": "lgr",
                "coarse_transform": coarse.tolist(),
                "refined_transform": final.tolist(),
                "icp_stages": [
                    {
                        "distance": distance,
                        "status": status,
                        "message": message,
                        "iterations": len(history),
                        "last_rmse": float(history[-1].rmse) if history else None,
                    }
                ],
            }
        ]
    else:
        # robust 后端:同时准备 LGR / GC-RANSAC / FPFH 三种候选,最后仅凭几何择优
        candidates: dict[str, np.ndarray] = {"lgr": output.lgr_transform}
        try:
            # 用 GC-RANSAC 在 GeoTransformer 对应上重新鲁棒求解位姿(以对应分数为先验概率)
            gc_transform, inlier_mask = gcransac_from_correspondences(
                output.source_points,
                output.target_points,
                output.scores,
                correspondence_distance=distance,
                max_iters=10000,
                seed=42,
            )
            candidates["geotransformer_gcransac"] = gc_transform
            gc_inliers = int(inlier_mask.sum())
        except Exception as exc:
            errors["geotransformer_gcransac"] = str(exc)  # 求解失败则记录错误,不影响其他候选
        try:
            # 传统 FPFH + RANSAC 作为兜底候选
            candidates["fpfh"] = fpfh_registration(source, target, voxel_size, seed=42)
        except Exception as exc:
            errors["fpfh"] = str(exc)
        # 对所有候选精配并按几何指标择优(不看真值)
        selected, final, candidate_rows = refine_and_select_candidates(
            source,
            target,
            candidates,
            voxel_size=voxel_size,
            score_distance=distance,
        )
        coarse = candidates[selected]

    # 非 fusion 后端:为每个候选补上粗配/精配相对真值的位姿误差(仅作诊断)
    if backend != "fusion":
        for candidate in candidate_rows:
            candidate["coarse_pose"] = _pose_metrics(
                np.asarray(candidate["coarse_transform"]), ground_truth, diagonal
            )
            candidate["refined_pose"] = _pose_metrics(
                np.asarray(candidate["refined_transform"]), ground_truth, diagonal
            )
    # ---- 汇总最终指标 ----
    coarse_rotation, coarse_translation = pose_errors(coarse, ground_truth)   # 粗配误差
    final_rotation, final_translation = pose_errors(final, ground_truth)      # 精配后误差
    fitness = alignment_metrics(source, target, final, distance)["fitness"]   # 配准贴合度
    final_ratio = float(final_translation / diagonal)                        # 平移误差比
    return {
        "pair": f"{source_name}->{target_name}",
        "backend": backend,
        "overlap": round(float(overlap), 3),
        "correspondences": diagnostics,
        "gcransac_inliers": gc_inliers,
        "candidate_errors": errors,
        "candidates": candidate_rows,
        "global_search": global_search,
        "selected_candidate": selected,
        "coarse_rot": round(float(coarse_rotation), 2),
        "coarse_tr_ratio": round(float(coarse_translation / diagonal), 4),
        "final_rot": round(float(final_rotation), 2),
        "final_tr_ratio": round(final_ratio, 4),
        "fitness": round(float(fitness), 3),
        # 三档成功判据:旋转均需 <5°,平移误差比分别 <2% / <3% / <5%(由严到宽)
        "success_2pct": bool(final_rotation < 5.0 and final_ratio < 0.02),
        "success_3pct": bool(final_rotation < 5.0 and final_ratio < 0.03),
        "success_practical_5pct": bool(final_rotation < 5.0 and final_ratio < 0.05),
    }


def _write_json_atomic(path: Path, value: object) -> None:
    """原子写 JSON:先写 .tmp 再 replace,避免中途崩溃留下半截文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)  # 同目录 replace 是原子操作


CSV_FIELDS = [
    "pair", "source", "target", "backend", "status", "overlap",
    "correspondence_count", "gt_inliers_0.005", "gt_inliers_0.010",
    "gcransac_inliers", "selected_candidate", "coarse_rot_deg",
    "coarse_tr_ratio", "final_rot_deg", "final_tr_ratio", "fitness",
    "global_fitness", "violation", "fine_violation", "gate_passed",
    "runtime_seconds", "success_2pct", "success_3pct",
    "success_practical_5pct", "error",
]


def _write_csv_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    """把评测结果扁平化写成 CSV(同样原子写);utf-8-sig 便于 Excel 正确识别中文/BOM。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            pair = str(row.get("pair", ""))
            source, separator, target = pair.partition("->")
            diagnostics = row.get("correspondences") or {}
            search = row.get("global_search") or {}
            timings = search.get("timings_ms") or {}
            writer.writerow({
                "pair": pair,
                "source": source,
                "target": target if separator else "",
                "backend": row.get("backend", ""),
                "status": row.get("status", ""),
                "overlap": row.get("overlap", ""),
                "correspondence_count": diagnostics.get("count", ""),
                "gt_inliers_0.005": diagnostics.get("gt_inliers_0.005", ""),
                "gt_inliers_0.010": diagnostics.get("gt_inliers_0.010", ""),
                "gcransac_inliers": row.get("gcransac_inliers", ""),
                "selected_candidate": row.get("selected_candidate", ""),
                "coarse_rot_deg": row.get("coarse_rot", ""),
                "coarse_tr_ratio": row.get("coarse_tr_ratio", ""),
                "final_rot_deg": row.get("final_rot", ""),
                "final_tr_ratio": row.get("final_tr_ratio", ""),
                "fitness": row.get("fitness", ""),
                "global_fitness": search.get("fitness", ""),
                "violation": search.get("violation", ""),
                "fine_violation": search.get("fine_violation", ""),
                "gate_passed": search.get("gate_passed", ""),
                "runtime_seconds": (
                    round(float(timings["total"]) / 1000.0, 3)
                    if "total" in timings else ""
                ),
                "success_2pct": row.get("success_2pct", ""),
                "success_3pct": row.get("success_3pct", ""),
                "success_practical_5pct": row.get("success_practical_5pct", ""),
                "error": row.get("error", ""),
            })
    temporary.replace(path)


def _evaluation_summary(rows: list[dict[str, object]], requested_pairs: int) -> dict[str, object]:
    """汇总所有点对结果:分别统计三档成功数,并算出"占完成数"和"占请求数"两种成功率。"""
    completed = [row for row in rows if row.get("status") == "ok"]     # 成功跑完的对
    errors = [row for row in rows if row.get("status") == "error"]     # 出错的对
    strict = sum(bool(row.get("success_2pct")) for row in completed)   # 2% 严格档
    relaxed = sum(bool(row.get("success_3pct")) for row in completed)  # 3% 放宽档
    practical = sum(bool(row.get("success_practical_5pct")) for row in completed)  # 5% 实用档
    return {
        "requested_pairs": requested_pairs,
        "completed_pairs": len(completed),
        "error_pairs": len(errors),
        "success_2pct": strict,
        "success_3pct": relaxed,
        "success_practical_5pct": practical,
        "success_rate_2pct_completed": strict / len(completed) if completed else 0.0,
        "success_rate_3pct_completed": relaxed / len(completed) if completed else 0.0,
        "success_rate_practical_5pct_completed": practical / len(completed) if completed else 0.0,
        "success_rate_2pct_requested": strict / requested_pairs if requested_pairs else 0.0,
        "success_rate_3pct_requested": relaxed / requested_pairs if requested_pairs else 0.0,
        "success_rate_practical_5pct_requested": practical / requested_pairs if requested_pairs else 0.0,
        "failed_pairs": [row["pair"] for row in completed if not row.get("success_2pct")],
        "errors": [{"pair": row["pair"], "error": row.get("error", "")} for row in errors],
    }


def main() -> None:
    # ---- 命令行参数 ----
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)                 # 要评测的权重(必填)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "bunny" / "data")
    parser.add_argument("--voxel", type=float, default=0.0025)                    # 体素尺寸
    parser.add_argument("--distance", type=float, default=0.01)                   # 对应/内点距离阈值
    parser.add_argument("--backend", choices=["lgr", "robust", "fusion"], default="lgr")  # 求解后端
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "eval_3dmatch.json")
    parser.add_argument("--all-pairs", action="store_true", help="evaluate all directed scan pairs")
    parser.add_argument("--overlap-max", type=float, default=None,
                        help="only evaluate pairs whose measured overlap is below this value")
    parser.add_argument("--resume", action="store_true", help="resume completed pairs from --output")  # 断点续跑
    parser.add_argument("--max-pairs", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    poses = parse_bun_conf(args.data_dir / "bun.conf")
    # --all-pairs 时评测所有有向点对(排列),否则只评默认四对
    pairs = list(permutations(poses.keys(), 2)) if args.all_pairs else DEFAULT_PAIRS
    selection_overlaps: dict[str, float] = {}
    # 可选:只保留重合率低于阈值的对(用于专门评测低重合场景)
    if args.overlap_max is not None:
        if not 0 < args.overlap_max <= 1:
            parser.error("--overlap-max must be in (0, 1]")
        # 一次性缓存所有点云,避免逐对重复读盘
        cloud_cache = {
            name: read_points(args.data_dir / f"{name}.ply") for name in poses
        }
        filtered_pairs = []
        for source_name, target_name in pairs:
            overlap = symmetric_overlap(
                cloud_cache[source_name],
                cloud_cache[target_name],
                relative_transform(poses[source_name], poses[target_name]),
                args.distance,
            )
            selection_overlaps[f"{source_name}->{target_name}"] = overlap
            if overlap < args.overlap_max:
                filtered_pairs.append((source_name, target_name))
        pairs = filtered_pairs
        print(f"selected {len(pairs)} pairs with overlap < {args.overlap_max:.3f}")
    if args.max_pairs is not None:
        pairs = pairs[:args.max_pairs]  # 隐藏参数:仅取前 N 对(调试用)
    rows: list[dict[str, object]] = []
    # 断点续跑:从已有输出加载结果,刷新重合率与 5% 成功判据
    if args.resume and args.output.is_file():
        rows = json.loads(args.output.read_text(encoding="utf-8"))
        for row in rows:
            pair_name = str(row.get("pair", ""))
            if pair_name in selection_overlaps:
                row["overlap"] = round(selection_overlaps[pair_name], 3)
            if row.get("status", "ok") == "ok":
                row["success_practical_5pct"] = bool(
                    float(row.get("final_rot", float("inf"))) < 5.0
                    and float(row.get("final_tr_ratio", float("inf"))) < 0.05
                )
    # 已完成的点对集合,续跑时用于跳过
    completed_pairs = {
        row["pair"] for row in rows if row.get("status", "ok") == "ok"
    }
    summary_path = args.output.with_suffix(".summary.json")
    csv_path = args.output.with_suffix(".csv")
    # ---- 逐对评测主循环 ----
    for index, (source_name, target_name) in enumerate(pairs, start=1):
        pair_name = f"{source_name}->{target_name}"
        if pair_name in completed_pairs:
            print(f"[{index}/{len(pairs)}] skip completed {pair_name}")
            continue
        rows = [row for row in rows if row.get("pair") != pair_name]  # 去掉旧的同名(错误)记录
        try:
            row = evaluate_pair(
                source_name,
                target_name,
                data_dir=args.data_dir,
                checkpoint=args.checkpoint,
                poses=poses,
                voxel_size=args.voxel,
                distance=args.distance,
                backend=args.backend,
            )
            row["status"] = "ok"
            print(
                f"[{index}/{len(pairs)}] {row['pair']:>18} backend={args.backend:<6} "
                f"selected={row['selected_candidate']:<26} ov={row['overlap']:.3f} "
                f"coarse_rot={row['coarse_rot']:7.2f} final_rot={row['final_rot']:7.2f} "
                f"tr={row['final_tr_ratio']:.4f} fit={row['fitness']:.3f} "
                f"{'OK' if row['success_2pct'] else 'FAIL'}"
            )
        except Exception as exc:
            # 单对失败不中断整体评测,记为 error 继续下一对
            row = {"pair": pair_name, "backend": args.backend, "status": "error", "error": str(exc)}
            print(f"[{index}/{len(pairs)}] {pair_name} ERROR {exc}")
        rows.append(row)
        # 每完成一对就落盘一次(JSON/CSV/汇总),保证中途崩溃也能续跑
        _write_json_atomic(args.output, rows)
        _write_csv_atomic(csv_path, rows)
        _write_json_atomic(summary_path, _evaluation_summary(rows, len(pairs)))
    # 循环结束后再整体写一次,确保最终结果完整
    _write_json_atomic(args.output, rows)
    _write_csv_atomic(csv_path, rows)
    _write_json_atomic(summary_path, _evaluation_summary(rows, len(pairs)))
    print("saved", args.output)
    # robust 后端额外输出针对最难对的诊断决策文件
    if args.backend == "robust":
        diagnostic_path = args.output.with_suffix(".diagnostic.json")
        decision = _stage_a_decision(rows)
        diagnostic_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"stage_A action={decision['action']} cause={decision['diagnosed_cause']} "
            f"saved={diagnostic_path}"
        )


if __name__ == "__main__":
    main()
