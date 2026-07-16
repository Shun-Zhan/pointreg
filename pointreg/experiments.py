"""批量实验与评测：生成课程设计所需的各类对比表格、扫频结果与图表。

包含：全点对最终评测(run_final_evaluation)、多方法对比(run_method_comparison)、
速度测试(run_speed_test)、以及重叠/体素/扰动的完整实验套件(run_full_suite)。
结果统一以 CSV/PNG 落盘到 output_dir，供报告直接引用。
"""

from __future__ import annotations

import os
import platform
from dataclasses import asdict, replace
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .dataset import register_dataset_pair
from .io import parse_bun_conf, read_points
from .metrics import symmetric_overlap
from .models import RegistrationConfig
from .nearest import nearest_neighbors
from .pipeline import register_pair
from .preprocessing import preprocess_points
from .transforms import apply_transform, relative_transform


# 方法对比实验里要跑的 (粗配准, 精配准) 组合
METHODS = [("none", "custom_icp"), ("pca", "custom_icp"), ("fpfh", "custom_icp"), ("fpfh", "point_to_plane")]

FINAL_EVALUATION_COLUMNS = [
    "方法", "点对", "源点云", "目标点云", "重合率",
    "Transformer对应数", "真值对应内点数", "GC-RANSAC内点数", "FPFH-RANSAC粗配内点数",
    "入选候选", "粗配旋转误差(°)", "粗配平移误差", "粗配相对平移误差",
    "最终旋转误差(°)", "最终平移误差", "最终相对平移误差", "fitness",
    "violation", "自由空间门控", "耗时(ms)", "粗配耗时(ms)", "ICP耗时(ms)", "桥接图耗时(ms)",
    "成功_2%", "成功_3%", "成功_5%", "状态", "说明",
]


def _ground_truth_inlier_count(
    source: np.ndarray, target: np.ndarray, ground_truth: np.ndarray, threshold: float
) -> int:
    """按真值位姿对齐后，统计落在阈值内的对应点数（衡量该点对的“可配准上限”）。"""
    distances, _ = nearest_neighbors(apply_transform(source, ground_truth), target)
    return int(np.count_nonzero(distances <= threshold))


def _success_at(result, translation_ratio: float) -> bool:
    """在给定平移误差阈值下判定是否成功（旋转固定<5°），用于生成 2%/3%/5% 三档成功列。"""
    return bool(
        result.status != "failed"
        and result.metrics.get("rotation_error_deg", float("inf")) < 5.0
        and result.metrics.get("translation_error_ratio", float("inf")) < translation_ratio
    )


def run_final_evaluation(
    data_dir: str | Path,
    output_dir: str | Path,
    base_config: RegistrationConfig | None = None,
    bridge_overlap_threshold: float = 0.50,
) -> pd.DataFrame:
    """Evaluate all unordered pairs and bridge only low-overlap pairs.

    The current implementation uses Open3D FPFH+RANSAC, not GC-RANSAC, and
    contains no Transformer or free-space candidate gate. Their requested CSV
    columns are intentionally empty instead of relabelling other measurements.
    """
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 最终评测固定使用 FPFH 粗配 + 自研 ICP 精配
    config = replace(
        base_config or RegistrationConfig(), coarse_method="fpfh", fine_method="custom_icp"
    )
    poses = parse_bun_conf(data_dir / "bun.conf")
    names = [name for name in poses if (data_dir / f"{name}.ply").exists()]
    clouds = {name: read_points(data_dir / f"{name}.ply") for name in names}
    # 预处理后的点云用于计算真值内点数，避免与原始密度混淆
    eval_clouds = {
        name: preprocess_points(points, config.voxel_size, config.remove_outliers)
        for name, points in clouds.items()
    }

    # 先算好所有无序点对的真值、重合率与真值内点数，供两种方法共用
    pair_metadata = []
    for source_name, target_name in combinations(names, 2):
        ground_truth = relative_transform(poses[source_name], poses[target_name])
        overlap = symmetric_overlap(
            clouds[source_name], clouds[target_name], ground_truth,
            config.max_correspondence_distance,
        )
        gt_inliers = _ground_truth_inlier_count(
            eval_clouds[source_name], eval_clouds[target_name], ground_truth,
            config.max_correspondence_distance,
        )
        pair_metadata.append((source_name, target_name, ground_truth, overlap, gt_inliers))

    rows = []
    for method in ("FPFH+RANSAC+Point-to-Point ICP", "桥接法"):
        for source_name, target_name, ground_truth, overlap, gt_inliers in pair_metadata:
            # 桥接法只针对低重合点对运行（高重合直接配准即可，跳过以省时）
            if method == "桥接法" and overlap >= bridge_overlap_threshold:
                continue
            if method == "桥接法":
                result = register_dataset_pair(data_dir, source_name, target_name, config)
                # 从说明里取出桥接路径作为“入选候选”一列
                selected = result.message.split(";", 1)[0].removeprefix("bridge path: ")
                # 桥接法没有独立粗配阶段，粗配列直接沿用最终误差
                coarse_rotation = result.metrics.get("rotation_error_deg")
                coarse_translation = result.metrics.get("translation_error")
                coarse_translation_ratio = result.metrics.get("translation_error_ratio")
                coarse_inliers = pd.NA
            else:
                result = register_pair(
                    data_dir / f"{source_name}.ply",
                    data_dir / f"{target_name}.ply",
                    config,
                    ground_truth=ground_truth,
                )
                selected = "Open3D FPFH-RANSAC"
                # 直接配准有独立粗配阶段，记录其单独的误差与内点数
                coarse_rotation = result.metrics.get("coarse_rotation_error_deg")
                coarse_translation = result.metrics.get("coarse_translation_error")
                coarse_translation_ratio = result.metrics.get("coarse_translation_error_ratio")
                coarse_inliers = int(result.metrics.get("coarse_correspondences", 0))
            rows.append({
                "方法": method,
                "点对": f"{source_name}->{target_name}",
                "源点云": source_name,
                "目标点云": target_name,
                "重合率": overlap,
                "Transformer对应数": pd.NA,
                "真值对应内点数": gt_inliers,
                "GC-RANSAC内点数": pd.NA,
                "FPFH-RANSAC粗配内点数": coarse_inliers,
                "入选候选": selected,
                "粗配旋转误差(°)": coarse_rotation,
                "粗配平移误差": coarse_translation,
                "粗配相对平移误差": coarse_translation_ratio,
                "最终旋转误差(°)": result.metrics.get("rotation_error_deg"),
                "最终平移误差": result.metrics.get("translation_error"),
                "最终相对平移误差": result.metrics.get("translation_error_ratio"),
                "fitness": result.metrics.get("fitness"),
                "violation": pd.NA,
                "自由空间门控": pd.NA,
                "耗时(ms)": result.timings_ms.get("total"),
                "粗配耗时(ms)": result.timings_ms.get("coarse"),
                "ICP耗时(ms)": result.timings_ms.get("fine"),
                "桥接图耗时(ms)": result.timings_ms.get("bridge_graph"),
                "成功_2%": _success_at(result, 0.02),
                "成功_3%": _success_at(result, 0.03),
                "成功_5%": _success_at(result, 0.05),
                "状态": result.status,
                "说明": result.message,
            })
    # 用 utf-8-sig（带 BOM）导出，Excel 打开中文表头不乱码
    frame = pd.DataFrame(rows, columns=FINAL_EVALUATION_COLUMNS)
    frame.to_csv(output_dir / "final_evaluation.csv", index=False, encoding="utf-8-sig")
    return frame


def run_method_comparison(data_dir: str | Path, output_dir: str | Path, pairs: list[tuple[str, str]] | None = None, base_config: RegistrationConfig | None = None) -> pd.DataFrame:
    """在若干点对上遍历所有 (粗配,精配) 方法组合，输出对比表与柱状图。"""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    poses = parse_bun_conf(data_dir / "bun.conf")
    # 默认选覆盖易→难的三对（重合度依次降低）
    pairs = pairs or [("bun000", "bun045"), ("bun000", "bun090"), ("bun000", "bun180")]
    base_config = base_config or RegistrationConfig()
    rows = []
    for source_name, target_name in pairs:
        gt = relative_transform(poses[source_name], poses[target_name])
        for coarse, fine in METHODS:
            config = replace(base_config, coarse_method=coarse, fine_method=fine)
            result = register_pair(data_dir / f"{source_name}.ply", data_dir / f"{target_name}.ply", config, ground_truth=gt)
            # 把指标与各阶段耗时展平进同一行
            rows.append({"source": source_name, "target": target_name, "coarse": coarse, "fine": fine,
                         "status": result.status, "success": result.success, **result.metrics, **{f"time_{k}_ms": v for k, v in result.timings_ms.items()}})
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "method_comparison.csv", index=False)
    _save_plots(frame, output_dir)
    return frame


def run_speed_test(source: Path, target: Path, config: RegistrationConfig, repeats: int = 10, warmups: int = 1) -> pd.DataFrame:
    """重复多次配准以测速；先跑若干次热身排除冷启动，再统计正式各次耗时。"""
    for _ in range(warmups):
        register_pair(source, target, config)  # 热身：不计入结果
    rows = []
    for repeat in range(repeats):
        result = register_pair(source, target, config)
        rows.append({"repeat": repeat, "status": result.status, **result.metrics, **result.timings_ms})
    return pd.DataFrame(rows)


def run_full_suite(data_dir: str | Path, output_dir: str | Path, base_config: RegistrationConfig | None = None) -> dict[str, pd.DataFrame]:
    """一次性跑完整实验：方法对比、重合率、体素扫频、位姿扰动、速度测试。

    各子实验结果分别写 CSV，并返回汇总字典，便于报告统一取用。
    """
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = base_config or RegistrationConfig()
    poses = parse_bun_conf(data_dir / "bun.conf")
    method_frame = run_method_comparison(data_dir, output_dir, base_config=base)

    # —— 实验 1：以 bun000 为源，测它与其他视角的重合率并按降序排列 ——
    overlap_rows = []
    source_name = "bun000"
    source = read_points(data_dir / f"{source_name}.ply")
    for target_name in ["bun045", "bun090", "bun180", "bun270"]:
        target = read_points(data_dir / f"{target_name}.ply")
        gt = relative_transform(poses[source_name], poses[target_name])
        overlap_rows.append({"source": source_name, "target": target_name,
                             "overlap": symmetric_overlap(source, target, gt, base.max_correspondence_distance)})
    overlap_frame = pd.DataFrame(overlap_rows).sort_values("overlap", ascending=False)
    overlap_frame.to_csv(output_dir / "overlap.csv", index=False)

    # —— 实验 2：体素尺寸扫频，考察下采样粗细对精度/耗时的影响 ——
    voxel_rows = []
    target = read_points(data_dir / "bun045.ply")
    gt = relative_transform(poses["bun000"], poses["bun045"])
    for voxel in [.0015, .0025, .004, .006]:
        cfg = replace(base, voxel_size=voxel)
        result = register_pair(source, target, cfg, ground_truth=gt)
        voxel_rows.append({"voxel_size": voxel, "status": result.status, "success": result.success,
                           **result.metrics, **{f"time_{k}_ms": v for k, v in result.timings_ms.items()}})
    voxel_frame = pd.DataFrame(voxel_rows)
    voxel_frame.to_csv(output_dir / "voxel_sweep.csv", index=False)

    # —— 实验 3：位姿扰动测试，考察 ICP 对初值偏差的收敛能力（不用粗配准）——
    rng = np.random.default_rng(base.random_seed)
    perturb_rows = []
    from .transforms import make_transform
    for angle_deg in [5, 15, 30, 45, 60]:      # 扰动旋转角逐级加大
        for repeat in range(3):                # 每档随机重复 3 次
            axis = rng.normal(size=3); axis /= np.linalg.norm(axis)  # 随机单位旋转轴
            angle = np.radians(angle_deg)
            # 罗德里格斯公式：由轴角构造旋转矩阵
            cross = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
            rotation = np.eye(3) + np.sin(angle)*cross + (1-np.cos(angle))*(cross@cross)
            # 随机平移，模长定为源包围盒尺度的 2%
            translation = rng.normal(size=3); translation *= (.02 * np.linalg.norm(np.ptp(source, axis=0)) / np.linalg.norm(translation))
            # 在真值基础上叠加扰动作为 ICP 初值
            initial = make_transform(rotation, translation) @ gt
            cfg = replace(base, coarse_method="none")
            result = register_pair(source, target, cfg, ground_truth=gt, initial=initial)
            perturb_rows.append({"angle_deg": angle_deg, "repeat": repeat, "status": result.status,
                                 "success": result.success, **result.metrics, **{f"time_{k}_ms": v for k,v in result.timings_ms.items()}})
    perturb_frame = pd.DataFrame(perturb_rows)
    perturb_frame.to_csv(output_dir / "perturbation.csv", index=False)

    # —— 实验 4：速度测试，统计总耗时的中位/最小/最大值 ——
    speed_frame = run_speed_test(data_dir / "bun000.ply", data_dir / "bun045.ply", replace(base, coarse_method="none"), repeats=10)
    speed_frame.to_csv(output_dir / "speed.csv", index=False)
    summary = pd.DataFrame([{"experiment":"speed", "median_ms":speed_frame["total"].median(), "min_ms":speed_frame["total"].min(), "max_ms":speed_frame["total"].max()}])
    summary.to_csv(output_dir / "summary.csv", index=False)
    return {"methods": method_frame, "overlap": overlap_frame, "voxel": voxel_frame, "perturbation": perturb_frame, "speed": speed_frame}


def _save_plots(frame: pd.DataFrame, output_dir: Path) -> None:
    """把方法对比结果画成耗时/误差双柱状图并存为 PNG；缺少 matplotlib 时静默跳过。"""
    try:
        # 无显示环境（如 Linux 无 DISPLAY）切到 Agg 后端，避免因无 GUI 报错
        if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
            import matplotlib

            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    labels = frame["coarse"] + "+" + frame["fine"]  # x 轴标签：粗配+精配组合名
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(range(len(frame)), frame["time_total_ms"], color="#3b82f6")
    axes[0].set(title="Registration time", ylabel="ms")
    metric = "rotation_error_deg" if "rotation_error_deg" in frame else "rmse"
    axes[1].bar(range(len(frame)), frame[metric], color="#10b981")
    axes[1].set(title=metric, ylabel=metric)
    for axis in axes:
        axis.set_xticks(range(len(frame)), labels, rotation=65, ha="right", fontsize=7)
        axis.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(output_dir / "method_comparison.png", dpi=180)
    plt.close(fig)
