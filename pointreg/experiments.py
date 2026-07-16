"""批量实验：跑方法对比、全组合、体素扫描、扰动鲁棒性、速度测试并落盘为 CSV/图表。

这是课程设计“实验与分析”部分的驱动脚本，用于系统性地比较不同算法组合、参数与初值
扰动下的配准表现，产出可直接用于报告的表格和图。
"""
from __future__ import annotations

import os
import platform
from dataclasses import replace
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

from .io import parse_bun_conf, read_points
from .metrics import symmetric_overlap
from .models import RegistrationConfig
from .pipeline import register_pair
from .transforms import relative_transform


# 待对比的（粗配, 精配）方法组合：无粗配、PCA、FPFH 三种粗配，外加 FPFH+点到面。
METHODS = [("none", "custom_icp"), ("pca", "custom_icp"), ("fpfh", "custom_icp"), ("fpfh", "point_to_plane")]
# 重叠率阈值：低于此值认为两帧本身重叠太少、不在“应当能配准成功”的范围内。
SUPPORTED_OVERLAP_THRESHOLD = 0.5


def run_method_comparison(data_dir: str | Path, output_dir: str | Path, pairs: list[tuple[str, str]] | None = None, base_config: RegistrationConfig | None = None) -> pd.DataFrame:
    """在若干代表性配对上对比各粗配/精配方法组合，输出 method_comparison.csv 与图。

    对每个配对、每种方法组合各跑一次配准，汇总状态、成功与否、误差指标和分阶段耗时。
    """
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    poses = parse_bun_conf(data_dir / "bun.conf")
    # 默认选难度递增的三对（45°/90°/180°），也允许调用方自定义配对列表。
    pairs = pairs or [("bun000", "bun045"), ("bun000", "bun090"), ("bun000", "bun180")]
    base_config = base_config or RegistrationConfig()
    rows = []
    for source_name, target_name in pairs:
        gt = relative_transform(poses[source_name], poses[target_name])  # 该对的真值相对位姿。
        for coarse, fine in METHODS:
            # 在基础配置上只替换粗配/精配方法，其余超参保持一致以保证可比性。
            config = replace(base_config, coarse_method=coarse, fine_method=fine)
            result = register_pair(data_dir / f"{source_name}.ply", data_dir / f"{target_name}.ply", config, ground_truth=gt)
            # 把指标与计时展平进一行；计时键统一加 time_..._ms 前缀避免与指标重名。
            rows.append({"source": source_name, "target": target_name, "coarse": coarse, "fine": fine,
                         "status": result.status, "success": result.success, **result.metrics, **{f"time_{k}_ms": v for k, v in result.timings_ms.items()}})
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "method_comparison.csv", index=False)
    _save_plots(frame, output_dir)  # 顺带出一张耗时/误差对比柱状图。
    return frame


def run_all_pairs(data_dir: str | Path, output_dir: str | Path, pairs: list[tuple[str, str]] | None = None, base_config: RegistrationConfig | None = None) -> pd.DataFrame:
    """对所有有序两帧组合做严格的“仅两片点云”配准评估，输出 all_pairs.csv。

    这里不使用桥接图复合，而是逐对直接配准，用以暴露哪些低重叠对确实配不动，
    并结合真实重叠率把失败归因为“重叠不足不支持”还是“配准本身失败”。
    """
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    poses = parse_bun_conf(data_dir / "bun.conf")
    names = sorted(poses)
    pairs = pairs or list(permutations(names, 2))  # 默认遍历所有有序对（含方向）。
    base = base_config or RegistrationConfig()
    points_cache: dict[str, np.ndarray] = {}  # 缓存已读点云，避免同一帧被反复读盘。
    rows = []
    for source_name, target_name in pairs:
        source_path = data_dir / f"{source_name}.ply"
        target_path = data_dir / f"{target_name}.ply"
        ground_truth = relative_transform(poses[source_name], poses[target_name])
        result = register_pair(source_path, target_path, base, ground_truth=ground_truth)
        # 按需读取并缓存源/目标点云，用于计算真实重叠率。
        if source_name not in points_cache:
            points_cache[source_name] = read_points(source_path)
        if target_name not in points_cache:
            points_cache[target_name] = read_points(target_path)
        overlap = symmetric_overlap(points_cache[source_name], points_cache[target_name],
                                    ground_truth, base.max_correspondence_distance)
        supported_by_overlap = overlap >= SUPPORTED_OVERLAP_THRESHOLD  # 重叠是否足够“应能配成”。
        # 失败归因：重叠本就不足 vs. 重叠够但算法没配好，便于区分“数据难”与“算法弱”。
        failure_reason = ""
        if not result.success:
            failure_reason = "low_overlap_unsupported" if not supported_by_overlap else "registration_failed"
        rows.append({"source": source_name, "target": target_name, "overlap": overlap,
                     "supported_by_overlap": supported_by_overlap, "status": result.status,
                     "success": result.success, "failure_reason": failure_reason, **result.metrics,
                     **{f"time_{key}_ms": value for key, value in result.timings_ms.items()},
                     "message": result.message})
    # 排序把失败的、误差大的排在前面，方便快速定位问题配对。
    frame = pd.DataFrame(rows).sort_values(["success", "rotation_error_deg", "translation_error_ratio"],
                                           ascending=[True, False, False])
    frame.to_csv(output_dir / "all_pairs.csv", index=False)
    return frame


def run_speed_test(source: Path, target: Path, config: RegistrationConfig, repeats: int = 10, warmups: int = 1) -> pd.DataFrame:
    """对同一配对重复配准以测速：先热身若干次再正式计时 repeats 次，返回逐次计时表。

    warmups 次不计入结果，用于排除首次运行的缓存/加载抖动，让速度数据更稳定。
    """
    for _ in range(warmups):
        register_pair(source, target, config)  # 热身：结果丢弃，只为让系统进入稳态。
    rows = []
    for repeat in range(repeats):
        result = register_pair(source, target, config)
        rows.append({"repeat": repeat, "status": result.status, **result.metrics, **result.timings_ms})
    return pd.DataFrame(rows)


def run_full_suite(data_dir: str | Path, output_dir: str | Path, base_config: RegistrationConfig | None = None) -> dict[str, pd.DataFrame]:
    """一次性跑齐完整实验套件：方法对比、重叠率、体素扫描、初值扰动、速度测试。

    返回以实验名为键的 DataFrame 字典，同时把每项结果分别写成 CSV，speed 另出汇总。
    """
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = base_config or RegistrationConfig()
    poses = parse_bun_conf(data_dir / "bun.conf")
    # 实验一：方法对比（复用上面的函数）。
    method_frame = run_method_comparison(data_dir, output_dir, base_config=base)

    # 实验二：重叠率——固定以 bun000 为源，考察它与不同角度目标帧的真实重叠。
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

    # 实验三：体素扫描——固定 bun000->bun045，考察不同下采样粒度对精度/速度的影响。
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

    # 实验四：初值扰动鲁棒性——给真值位姿叠加不同幅度的旋转/平移扰动作为初值，
    # 关闭粗配（coarse=none）直接精配，看 ICP 能从多大偏差里“拉回来”。
    rng = np.random.default_rng(base.random_seed)  # 固定随机种子保证实验可复现。
    perturb_rows = []
    from .transforms import make_transform
    for angle_deg in [5, 15, 30, 45, 60]:
        for repeat in range(3):  # 每个扰动幅度重复 3 次以观察稳定性。
            axis = rng.normal(size=3); axis /= np.linalg.norm(axis)  # 随机单位旋转轴。
            angle = np.radians(angle_deg)
            # 反对称叉乘矩阵，用于罗德里格斯公式由轴角构造旋转矩阵。
            cross = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
            rotation = np.eye(3) + np.sin(angle)*cross + (1-np.cos(angle))*(cross@cross)
            # 随机方向平移，幅度定为源点云跨度的 2%，与角度扰动一起构成初值偏差。
            translation = rng.normal(size=3); translation *= (.02 * np.linalg.norm(np.ptp(source, axis=0)) / np.linalg.norm(translation))
            initial = make_transform(rotation, translation) @ gt  # 在真值基础上叠加扰动得到初值。
            cfg = replace(base, coarse_method="none")
            result = register_pair(source, target, cfg, ground_truth=gt, initial=initial)
            perturb_rows.append({"angle_deg": angle_deg, "repeat": repeat, "status": result.status,
                                 "success": result.success, **result.metrics, **{f"time_{k}_ms": v for k,v in result.timings_ms.items()}})
    perturb_frame = pd.DataFrame(perturb_rows)
    perturb_frame.to_csv(output_dir / "perturbation.csv", index=False)

    # 实验五：速度测试并汇总中位/最小/最大总耗时。
    speed_frame = run_speed_test(data_dir / "bun000.ply", data_dir / "bun045.ply", replace(base, coarse_method="none"), repeats=10)
    speed_frame.to_csv(output_dir / "speed.csv", index=False)
    summary = pd.DataFrame([{"experiment":"speed", "median_ms":speed_frame["total"].median(), "min_ms":speed_frame["total"].min(), "max_ms":speed_frame["total"].max()}])
    summary.to_csv(output_dir / "summary.csv", index=False)
    return {"methods": method_frame, "overlap": overlap_frame, "voxel": voxel_frame, "perturbation": perturb_frame, "speed": speed_frame}


def _save_plots(frame: pd.DataFrame, output_dir: Path) -> None:
    """把方法对比结果画成“耗时 + 误差”并排柱状图并存盘；缺 matplotlib 时静默跳过。"""
    try:
        # 无显示环境的 Linux（如服务器/CI）切换到 Agg 无界面后端，避免报错。
        if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
            import matplotlib

            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return  # 没装 matplotlib 就不画图，不影响 CSV 产出。
    labels = frame["coarse"] + "+" + frame["fine"]  # x 轴标签用“粗配+精配”组合名。
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
