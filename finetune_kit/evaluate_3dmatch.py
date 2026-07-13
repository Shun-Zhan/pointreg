"""Evaluate the 3DMatch GeoTransformer checkpoint on low-overlap Bunny pairs.

The ``lgr`` backend preserves the original upstream pose path.  The ``robust``
backend re-solves the dense GeoTransformer correspondences with GC-RANSAC and
selects between LGR, GC-RANSAC and FPFH using geometry only.
"""

from __future__ import annotations

import argparse
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
from pointreg.global_search import GlobalRegistrationSearch
from pointreg.icp import custom_icp
from pointreg.io import parse_bun_conf, read_points
from pointreg.metrics import alignment_metrics, pose_errors, symmetric_overlap
from pointreg.models import RegistrationConfig
from pointreg.preprocessing import bounding_box_diagonal, preprocess_points
from pointreg.transforms import apply_transform, relative_transform


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
    transform = np.asarray(initial, dtype=float).copy()
    stages: list[dict[str, object]] = []
    base = RegistrationConfig(
        coarse_method="none",
        fine_method="custom_icp",
        voxel_size=voxel_size,
        max_iterations=30,
        trim_fraction=0.8,
    )
    for distance in (0.015, 0.008):
        config = replace(base, max_correspondence_distance=distance)
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
    """Refine and rank candidate poses without consulting ground truth."""
    if not candidates:
        raise ValueError("at least one registration candidate is required")
    rows: list[dict[str, object]] = []
    best_key: tuple[int, int, float] | None = None
    best_name = ""
    best_transform: np.ndarray | None = None
    for name, coarse in candidates.items():
        refined, stages = _two_stage_icp(source, target, coarse, voxel_size=voxel_size)
        reciprocal, total, error = _geometric_candidate_score(source, target, refined, score_distance)
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
        "gt_inliers_0.005": int((residuals <= 0.005).sum()),
        "gt_inliers_0.010": int((residuals <= 0.010).sum()),
    }


def _stage_a_decision(rows: list[dict[str, object]]) -> dict[str, object]:
    hard = next(row for row in rows if row["pair"] == "bun000->bun180")
    rotation = float(hard["final_rot"])
    translation = float(hard["final_tr_ratio"])
    trigger_fusion = rotation < 5.0 and 0.02 <= translation <= 0.03
    if bool(hard["success_2pct"]):
        action = "stage_a_succeeded_stop"
    elif trigger_fusion:
        action = "activate_stage_c_fusion"
    else:
        action = "diagnose_only_do_not_activate_stage_c"
    diagnostics = hard["correspondences"]
    gc_candidate = next(
        (candidate for candidate in hard["candidates"] if candidate["name"] == "geotransformer_gcransac"),
        None,
    )
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
    rotation, translation = pose_errors(transform, ground_truth)
    return {
        "rot": round(float(rotation), 3),
        "tr_ratio": round(float(translation / diagonal), 5),
    }


def _fusion_seed_candidates(
    output: GeoTransformerCorrespondences,
    source: np.ndarray,
    target: np.ndarray,
    voxel_size: float,
) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, str]]:
    candidates: dict[str, np.ndarray] = {"lgr": output.lgr_transform}
    inlier_counts: dict[str, int] = {}
    errors: dict[str, str] = {}
    for threshold in (0.008, 0.010, 0.012):
        for seed in (0, 1, 42):
            label = f"geotransformer_gcransac_t{threshold:.3f}_s{seed}"
            permutation = np.random.default_rng(seed).permutation(len(output.source_points))
            try:
                transform, inliers = gcransac_from_correspondences(
                    output.source_points[permutation],
                    output.target_points[permutation],
                    output.scores[permutation],
                    correspondence_distance=threshold,
                    max_iters=10000,
                    seed=seed,
                )
                candidates[label] = transform
                inlier_counts[label] = int(inliers.sum())
            except Exception as exc:
                errors[label] = str(exc)
    try:
        candidates["fpfh"] = fpfh_registration(source, target, voxel_size, seed=42)
    except Exception as exc:
        errors["fpfh"] = str(exc)
    return candidates, inlier_counts, errors


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
    ground_truth = relative_transform(poses[source_name], poses[target_name])
    source = preprocess_points(read_points(data_dir / f"{source_name}.ply"), voxel_size)
    target = preprocess_points(read_points(data_dir / f"{target_name}.ply"), voxel_size)
    overlap = symmetric_overlap(source, target, ground_truth, distance)
    diagonal = bounding_box_diagonal(source, target)
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

    if backend == "fusion":
        candidates, fusion_inliers, errors = _fusion_seed_candidates(
            output, source, target, voxel_size
        )
        config = RegistrationConfig(
            coarse_method="none",
            voxel_size=voxel_size,
            max_correspondence_distance=distance,
            adaptive_trim=True,
        )
        started = perf_counter()
        search = GlobalRegistrationSearch(source, target, config)
        initialization_ms = (perf_counter() - started) * 1000
        result = search.run(candidates)
        global_search = result.to_dict()
        global_search["timings_ms"]["initialization"] = initialization_ms
        global_search["seed_inliers"] = fusion_inliers
        for candidate in global_search["candidates"]:
            candidate["pose"] = _pose_metrics(
                np.asarray(candidate["transform"]), ground_truth, diagonal
            )
        coarse = result.coarse_transform
        final = result.transform
        selected = result.selected_label
        candidate_rows = global_search["candidates"]
    elif backend == "lgr":
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
        candidates: dict[str, np.ndarray] = {"lgr": output.lgr_transform}
        try:
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
            errors["geotransformer_gcransac"] = str(exc)
        try:
            candidates["fpfh"] = fpfh_registration(source, target, voxel_size, seed=42)
        except Exception as exc:
            errors["fpfh"] = str(exc)
        selected, final, candidate_rows = refine_and_select_candidates(
            source,
            target,
            candidates,
            voxel_size=voxel_size,
            score_distance=distance,
        )
        coarse = candidates[selected]

    if backend != "fusion":
        for candidate in candidate_rows:
            candidate["coarse_pose"] = _pose_metrics(
                np.asarray(candidate["coarse_transform"]), ground_truth, diagonal
            )
            candidate["refined_pose"] = _pose_metrics(
                np.asarray(candidate["refined_transform"]), ground_truth, diagonal
            )
    coarse_rotation, coarse_translation = pose_errors(coarse, ground_truth)
    final_rotation, final_translation = pose_errors(final, ground_truth)
    fitness = alignment_metrics(source, target, final, distance)["fitness"]
    final_ratio = float(final_translation / diagonal)
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
        "success_2pct": bool(final_rotation < 5.0 and final_ratio < 0.02),
        "success_3pct": bool(final_rotation < 5.0 and final_ratio < 0.03),
    }


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _evaluation_summary(rows: list[dict[str, object]], requested_pairs: int) -> dict[str, object]:
    completed = [row for row in rows if row.get("status") == "ok"]
    errors = [row for row in rows if row.get("status") == "error"]
    strict = sum(bool(row.get("success_2pct")) for row in completed)
    relaxed = sum(bool(row.get("success_3pct")) for row in completed)
    return {
        "requested_pairs": requested_pairs,
        "completed_pairs": len(completed),
        "error_pairs": len(errors),
        "success_2pct": strict,
        "success_3pct": relaxed,
        "success_rate_2pct_completed": strict / len(completed) if completed else 0.0,
        "success_rate_3pct_completed": relaxed / len(completed) if completed else 0.0,
        "success_rate_2pct_requested": strict / requested_pairs if requested_pairs else 0.0,
        "success_rate_3pct_requested": relaxed / requested_pairs if requested_pairs else 0.0,
        "failed_pairs": [row["pair"] for row in completed if not row.get("success_2pct")],
        "errors": [{"pair": row["pair"], "error": row.get("error", "")} for row in errors],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "bunny" / "data")
    parser.add_argument("--voxel", type=float, default=0.0025)
    parser.add_argument("--distance", type=float, default=0.01)
    parser.add_argument("--backend", choices=["lgr", "robust", "fusion"], default="lgr")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "eval_3dmatch.json")
    parser.add_argument("--all-pairs", action="store_true", help="evaluate all directed scan pairs")
    parser.add_argument("--overlap-max", type=float, default=None,
                        help="only evaluate pairs whose measured overlap is below this value")
    parser.add_argument("--resume", action="store_true", help="resume completed pairs from --output")
    parser.add_argument("--max-pairs", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    poses = parse_bun_conf(args.data_dir / "bun.conf")
    pairs = list(permutations(poses.keys(), 2)) if args.all_pairs else DEFAULT_PAIRS
    if args.overlap_max is not None:
        if not 0 < args.overlap_max <= 1:
            parser.error("--overlap-max must be in (0, 1]")
        cloud_cache = {
            name: preprocess_points(read_points(args.data_dir / f"{name}.ply"), args.voxel)
            for name in poses
        }
        filtered_pairs = []
        for source_name, target_name in pairs:
            overlap = symmetric_overlap(
                cloud_cache[source_name],
                cloud_cache[target_name],
                relative_transform(poses[source_name], poses[target_name]),
                args.distance,
            )
            if overlap < args.overlap_max:
                filtered_pairs.append((source_name, target_name))
        pairs = filtered_pairs
        print(f"selected {len(pairs)} pairs with overlap < {args.overlap_max:.3f}")
    if args.max_pairs is not None:
        pairs = pairs[:args.max_pairs]
    rows: list[dict[str, object]] = []
    if args.resume and args.output.is_file():
        rows = json.loads(args.output.read_text(encoding="utf-8"))
    completed_pairs = {
        row["pair"] for row in rows if row.get("status", "ok") == "ok"
    }
    summary_path = args.output.with_suffix(".summary.json")
    for index, (source_name, target_name) in enumerate(pairs, start=1):
        pair_name = f"{source_name}->{target_name}"
        if pair_name in completed_pairs:
            print(f"[{index}/{len(pairs)}] skip completed {pair_name}")
            continue
        rows = [row for row in rows if row.get("pair") != pair_name]
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
            row = {"pair": pair_name, "backend": args.backend, "status": "error", "error": str(exc)}
            print(f"[{index}/{len(pairs)}] {pair_name} ERROR {exc}")
        rows.append(row)
        _write_json_atomic(args.output, rows)
        _write_json_atomic(summary_path, _evaluation_summary(rows, len(pairs)))
    _write_json_atomic(args.output, rows)
    _write_json_atomic(summary_path, _evaluation_summary(rows, len(pairs)))
    print("saved", args.output)
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
