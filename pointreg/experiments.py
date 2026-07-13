from __future__ import annotations

import os
import platform
import json
from dataclasses import asdict, replace
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

from .io import parse_bun_conf, read_points
from .metrics import pose_errors, symmetric_overlap
from .models import RegistrationConfig
from .pipeline import register_pair
from .transforms import relative_transform


METHODS = [("fpfh", "custom_icp"),
           ("fpfh_multi_verified", "custom_icp"),
           ("fpfh_multi_verified", "point_to_plane")]
AB_VARIANTS = [
    ("ransac_fixed", "fpfh", "custom_icp", False),
    ("ransac_adaptive", "fpfh", "custom_icp", True),
    ("sc2_gnc_fixed", "sc2_gnc", "custom_icp", False),
    ("sc2_gnc_adaptive", "sc2_gnc", "custom_icp", True),
    ("ransac_point_to_plane", "fpfh", "point_to_plane", False),
]
VERIFIED_VARIANTS = [
    ("baseline_ransac_fixed", "fpfh", "custom_icp", False),
    ("baseline_sc2_fixed", "sc2_gnc", "custom_icp", False),
    ("multi_ransac_fixed", "fpfh_multi_verified", "custom_icp", False),
    ("multi_ransac_adaptive", "fpfh_multi_verified", "custom_icp", True),
    ("multi_ransac_point_to_plane", "fpfh_multi_verified", "point_to_plane", False),
]
SUPPORTED_OVERLAP_THRESHOLD = 0.5


def run_method_comparison(data_dir: str | Path, output_dir: str | Path, pairs: list[tuple[str, str]] | None = None, base_config: RegistrationConfig | None = None) -> pd.DataFrame:
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    poses = parse_bun_conf(data_dir / "bun.conf")
    pairs = pairs or [("bun000", "bun045"), ("bun000", "bun090"), ("bun000", "bun180")]
    base_config = base_config or RegistrationConfig()
    rows = []
    for source_name, target_name in pairs:
        gt = relative_transform(poses[source_name], poses[target_name])
        for coarse, fine in METHODS:
            config = replace(base_config, coarse_method=coarse, fine_method=fine)
            result = register_pair(data_dir / f"{source_name}.ply", data_dir / f"{target_name}.ply", config, ground_truth=gt)
            rows.append({"source": source_name, "target": target_name, "coarse": coarse, "fine": fine,
                         "status": result.status, "success": result.success, **result.metrics, **{f"time_{k}_ms": v for k, v in result.timings_ms.items()}})
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "method_comparison.csv", index=False)
    _save_plots(frame, output_dir)
    return frame


def run_all_pairs(data_dir: str | Path, output_dir: str | Path, pairs: list[tuple[str, str]] | None = None, base_config: RegistrationConfig | None = None) -> pd.DataFrame:
    """Evaluate strict two-cloud registration for every ordered scan pair."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    poses = parse_bun_conf(data_dir / "bun.conf")
    names = sorted(poses)
    pairs = pairs or list(permutations(names, 2))
    base = base_config or RegistrationConfig()
    points_cache: dict[str, np.ndarray] = {}
    rows = []
    for source_name, target_name in pairs:
        source_path = data_dir / f"{source_name}.ply"
        target_path = data_dir / f"{target_name}.ply"
        ground_truth = relative_transform(poses[source_name], poses[target_name])
        result = register_pair(source_path, target_path, base, ground_truth=ground_truth)
        if source_name not in points_cache:
            points_cache[source_name] = read_points(source_path)
        if target_name not in points_cache:
            points_cache[target_name] = read_points(target_path)
        overlap = symmetric_overlap(points_cache[source_name], points_cache[target_name],
                                    ground_truth, base.max_correspondence_distance)
        supported_by_overlap = overlap >= SUPPORTED_OVERLAP_THRESHOLD
        failure_reason = ""
        if not result.success:
            failure_reason = "low_overlap_unsupported" if not supported_by_overlap else "registration_failed"
        rows.append({"source": source_name, "target": target_name, "overlap": overlap,
                     "supported_by_overlap": supported_by_overlap, "status": result.status,
                     "success": result.success, "failure_reason": failure_reason, **result.metrics,
                     **{f"time_{key}_ms": value for key, value in result.timings_ms.items()},
                     "message": result.message})
    frame = pd.DataFrame(rows).sort_values(["success", "rotation_error_deg", "translation_error_ratio"],
                                           ascending=[True, False, False])
    frame.to_csv(output_dir / "all_pairs.csv", index=False)
    return frame


def _overlap_group(overlap: float) -> str:
    if overlap < 0.3:
        return "low_<0.30"
    if overlap < 0.7:
        return "medium_0.30-0.70"
    return "high_>=0.70"


def _ab_row(source_name: str, target_name: str, method: str, overlap: float, result) -> dict:
    coarse_success = (
        result.metrics.get("coarse_rotation_error_deg", float("inf")) < 5.0
        and result.metrics.get("coarse_translation_error_ratio", float("inf")) < 0.02)
    failure_reason = ""
    if not result.success:
        failure_reason = "low_overlap_boundary" if overlap < 0.3 else "registration_failed"
    return {
        "source": source_name, "target": target_name, "method": method,
        "overlap": overlap, "overlap_group": _overlap_group(overlap),
        "status": result.status, "success": result.success, "coarse_success": coarse_success,
        "failure_reason": failure_reason, **result.metrics,
        "transform_determinant": float(np.linalg.det(result.transformation[:3, :3])),
        "transform_orthogonality_error": float(np.linalg.norm(
            result.transformation[:3, :3].T @ result.transformation[:3, :3] - np.eye(3))),
        **{f"time_{key}_ms": value for key, value in result.timings_ms.items()},
        "message": result.message,
    }


def run_ab_experiments(data_dir: str | Path, output_dir: str | Path,
                       base_config: RegistrationConfig | None = None) -> dict[str, pd.DataFrame]:
    """Run the A/B ablation and every ordered Bunny pair without using pose priors."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    poses = parse_bun_conf(data_dir / "bun.conf")
    base = base_config or RegistrationConfig()
    names = sorted(poses)
    points = {name: read_points(data_dir / f"{name}.ply") for name in names}
    overlaps: dict[tuple[str, str], float] = {}
    for source_name, target_name in permutations(names, 2):
        gt = relative_transform(poses[source_name], poses[target_name])
        overlaps[(source_name, target_name)] = symmetric_overlap(
            points[source_name], points[target_name], gt, base.max_correspondence_distance)

    def evaluate(pairs):
        rows = []
        for source_name, target_name in pairs:
            gt = relative_transform(poses[source_name], poses[target_name])
            for method, coarse, fine, adaptive in AB_VARIANTS:
                config = replace(base, coarse_method=coarse, fine_method=fine, adaptive_trim=adaptive)
                result = register_pair(points[source_name], points[target_name], config, ground_truth=gt)
                rows.append(_ab_row(source_name, target_name, method,
                                    overlaps[(source_name, target_name)], result))
        return pd.DataFrame(rows)

    ablation = evaluate([("bun000", "bun045"), ("bun000", "bun090"), ("bun000", "bun180")])
    ablation.to_csv(output_dir / "ablation重点三组.csv", index=False)
    repeatability_parts = []
    for repeat in range(3):
        part = evaluate([("bun000", "bun045"), ("bun000", "bun090"), ("bun000", "bun180")])
        part.insert(0, "repeat", repeat)
        repeatability_parts.append(part)
    repeatability = pd.concat(repeatability_parts, ignore_index=True)
    repeatability.to_csv(output_dir / "repeatability重点三组.csv", index=False)
    all_pairs = evaluate(list(permutations(names, 2)))
    all_pairs.to_csv(output_dir / "all_pairs_methods全部90对.csv", index=False)
    summary = (all_pairs.groupby(["method", "overlap_group"], observed=True)
               .agg(pairs=("success", "size"), successes=("success", "sum"),
                    success_rate=("success", "mean"), coarse_success_rate=("coarse_success", "mean"),
                    median_rotation_error_deg=("rotation_error_deg", "median"),
                    median_translation_error_ratio=("translation_error_ratio", "median"),
                    median_total_ms=("time_total_ms", "median"))
               .reset_index())
    summary.to_csv(output_dir / "summary_by_overlap.csv", index=False)
    (output_dir / "config.json").write_text(
        json.dumps({"base_config": asdict(base), "variants": AB_VARIANTS,
                    "overlap_groups": ["low_<0.30", "medium_0.30-0.70", "high_>=0.70"]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    _save_ab_plots(summary, output_dir)
    return {"ablation": ablation, "repeatability": repeatability,
            "all_pairs": all_pairs, "summary": summary}


def _save_ab_plots(summary: pd.DataFrame, output_dir: Path) -> None:
    try:
        if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
            import matplotlib
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    pivot = summary.pivot(index="method", columns="overlap_group", values="success_rate").fillna(0)
    axes = pivot.plot(kind="bar", figsize=(12, 5), ylim=(0, 1), rot=35, width=.8)
    axes.set(ylabel="Success rate", title="A/B methods by ground-truth overlap (evaluation only)")
    axes.grid(axis="y", alpha=.25)
    axes.figure.tight_layout()
    axes.figure.savefig(output_dir / "success_rate_by_overlap.png", dpi=180)
    plt.close(axes.figure)


def run_verified_experiments(data_dir: str | Path, output_dir: str | Path,
                             base_config: RegistrationConfig | None = None) -> dict[str, pd.DataFrame]:
    """Compare SC2-GNC ablation with full-cloud verified multi-RANSAC."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    poses = parse_bun_conf(data_dir / "bun.conf")
    base = base_config or RegistrationConfig()
    names = sorted(poses)
    points = {name: read_points(data_dir / f"{name}.ply") for name in names}
    overlaps = {}
    for source_name, target_name in permutations(names, 2):
        gt = relative_transform(poses[source_name], poses[target_name])
        overlaps[(source_name, target_name)] = symmetric_overlap(
            points[source_name], points[target_name], gt, base.max_correspondence_distance)

    candidate_rows: list[dict] = []

    def evaluate(pairs, capture_candidates=False):
        rows = []
        for source_name, target_name in pairs:
            ground_truth = relative_transform(poses[source_name], poses[target_name])
            for method, coarse, fine, adaptive in VERIFIED_VARIANTS:
                config = replace(base, coarse_method=coarse, fine_method=fine, adaptive_trim=adaptive)
                result = register_pair(points[source_name], points[target_name], config, ground_truth=ground_truth)
                row = _ab_row(source_name, target_name, method, overlaps[(source_name, target_name)], result)
                candidate_successes = []
                for candidate in result.coarse_candidates:
                    rotation_error, translation_error = pose_errors(candidate.transformation, ground_truth)
                    translation_ratio = translation_error / result.metrics.get("bbox_diagonal", float("inf"))
                    candidate_success = rotation_error < base.success_rotation_deg and translation_ratio < base.success_translation_ratio
                    candidate_successes.append(candidate_success)
                    if capture_candidates:
                        candidate_rows.append({
                            "source": source_name, "target": target_name, "method": method,
                            "origin": candidate.origin, "rank": candidate.validation_rank,
                            "selected": candidate.selected, "local_inliers": candidate.local_inliers,
                            "local_score": candidate.local_score, "pre_fitness": candidate.pre_fitness,
                            "pre_rmse": candidate.pre_rmse, "verified_fitness": candidate.verified_fitness,
                            "verified_rmse": candidate.verified_rmse, "validation_ms": candidate.validation_ms,
                            "rotation_error_deg": rotation_error,
                            "translation_error_ratio": translation_ratio, "candidate_success": candidate_success,
                        })
                contains_success = bool(candidate_successes and any(candidate_successes))
                row["candidate_contains_success"] = contains_success
                if not result.coarse_candidates:
                    row["candidate_contains_success"] = np.nan
                if coarse == "fpfh_multi_verified" and not row["coarse_success"]:
                    row["coarse_failure_stage"] = "scoring_selected_wrong" if contains_success else "correct_candidate_missing"
                else:
                    row["coarse_failure_stage"] = ""
                rows.append(row)
        return pd.DataFrame(rows)

    key_pairs = [("bun000", "bun045"), ("bun000", "bun090"),
                 ("bun090", "bun000"), ("bun000", "bun180")]
    ablation = evaluate(key_pairs, capture_candidates=True)
    ablation.to_csv(output_dir / "verified重点消融.csv", index=False)
    candidate_frame = pd.DataFrame(candidate_rows)
    candidate_frame.to_csv(output_dir / "candidate_rankings候选排行.csv", index=False)

    repeat_parts = []
    for repeat in range(3):
        part = evaluate(key_pairs)
        part.insert(0, "repeat", repeat)
        repeat_parts.append(part)
    repeatability = pd.concat(repeat_parts, ignore_index=True)
    repeatability.to_csv(output_dir / "verified重点重复性.csv", index=False)

    all_pairs = evaluate(list(permutations(names, 2)))
    all_pairs.to_csv(output_dir / "verified_all_pairs全部90对.csv", index=False)
    summary = (all_pairs.groupby(["method", "overlap_group"], observed=True)
               .agg(pairs=("success", "size"), successes=("success", "sum"),
                    success_rate=("success", "mean"), coarse_success_rate=("coarse_success", "mean"),
                    candidate_recall=("candidate_contains_success", "mean"),
                    median_rotation_error_deg=("rotation_error_deg", "median"),
                    median_coarse_rotation_error_deg=("coarse_rotation_error_deg", "median"),
                    median_total_ms=("time_total_ms", "median"),
                    median_validation_ms=("time_coarse_validation_ms", "median"))
               .reset_index())
    summary.to_csv(output_dir / "verified_summary_by_overlap.csv", index=False)
    (output_dir / "verified_config.json").write_text(
        json.dumps({"base_config": asdict(base), "variants": VERIFIED_VARIANTS,
                    "score": "harmonic bidirectional fitness, then pooled bidirectional inlier RMSE"},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    _save_ab_plots(summary, output_dir)
    return {"ablation": ablation, "candidate_rankings": candidate_frame,
            "repeatability": repeatability, "all_pairs": all_pairs, "summary": summary}


def run_speed_test(source: Path, target: Path, config: RegistrationConfig, repeats: int = 10, warmups: int = 1) -> pd.DataFrame:
    for _ in range(warmups):
        register_pair(source, target, config)
    rows = []
    for repeat in range(repeats):
        result = register_pair(source, target, config)
        rows.append({"repeat": repeat, "status": result.status, **result.metrics, **result.timings_ms})
    return pd.DataFrame(rows)


def run_full_suite(data_dir: str | Path, output_dir: str | Path, base_config: RegistrationConfig | None = None) -> dict[str, pd.DataFrame]:
    """Run method, overlap, voxel-size, perturbation and speed experiments."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = base_config or RegistrationConfig()
    poses = parse_bun_conf(data_dir / "bun.conf")
    method_frame = run_method_comparison(data_dir, output_dir, base_config=base)

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

    rng = np.random.default_rng(base.random_seed)
    perturb_rows = []
    from .transforms import make_transform
    for angle_deg in [5, 15, 30, 45, 60]:
        for repeat in range(3):
            axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
            angle = np.radians(angle_deg)
            cross = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
            rotation = np.eye(3) + np.sin(angle)*cross + (1-np.cos(angle))*(cross@cross)
            translation = rng.normal(size=3); translation *= (.02 * np.linalg.norm(np.ptp(source, axis=0)) / np.linalg.norm(translation))
            initial = make_transform(rotation, translation) @ gt
            cfg = replace(base, coarse_method="none")
            result = register_pair(source, target, cfg, ground_truth=gt, initial=initial)
            perturb_rows.append({"angle_deg": angle_deg, "repeat": repeat, "status": result.status,
                                 "success": result.success, **result.metrics, **{f"time_{k}_ms": v for k,v in result.timings_ms.items()}})
    perturb_frame = pd.DataFrame(perturb_rows)
    perturb_frame.to_csv(output_dir / "perturbation.csv", index=False)

    speed_frame = run_speed_test(data_dir / "bun000.ply", data_dir / "bun045.ply", replace(base, coarse_method="none"), repeats=10)
    speed_frame.to_csv(output_dir / "speed.csv", index=False)
    summary = pd.DataFrame([{"experiment":"speed", "median_ms":speed_frame["total"].median(), "min_ms":speed_frame["total"].min(), "max_ms":speed_frame["total"].max()}])
    summary.to_csv(output_dir / "summary.csv", index=False)
    return {"methods": method_frame, "overlap": overlap_frame, "voxel": voxel_frame, "perturbation": perturb_frame, "speed": speed_frame}


def _save_plots(frame: pd.DataFrame, output_dir: Path) -> None:
    try:
        if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
            import matplotlib

            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    labels = frame["coarse"] + "+" + frame["fine"]
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
