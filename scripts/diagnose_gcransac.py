"""Sweep FPFH/GC-RANSAC correspondence settings on direct Bunny pairs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import pygcransac

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pointreg.coarse import _fpfh_cloud_and_features
from pointreg.io import parse_bun_conf, read_points
from pointreg.metrics import pose_errors
from pointreg.transforms import relative_transform


PAIRS = [("bun000", "bun180"), ("chin", "top2"), ("bun000", "ear_back")]
SCALES = (0.0018, 0.0025, 0.004, 0.006)
RATIOS = (None, 0.9, 0.8)
THRESHOLDS = (0.005, 0.01, 0.015)


def solve(source, target, voxel, ratio, threshold):
    source_cloud, source_features = _fpfh_cloud_and_features(source, voxel)
    target_cloud, target_features = _fpfh_cloud_and_features(target, voxel)
    source_points, target_points = np.asarray(source_cloud.points), np.asarray(target_cloud.points)
    target_distances, target_indices = cKDTree(target_features).query(source_features, k=2)
    source_indices = cKDTree(source_features).query(target_features)[1]
    indices = np.arange(len(source_points))
    masks = source_indices[target_indices[:, 0]] == indices
    if ratio is not None:
        masks &= target_distances[:, 0] / np.maximum(target_distances[:, 1], 1e-12) < ratio
    indices = indices[masks]
    if len(indices) < 3:
        return None, len(indices), 0
    correspondences = np.c_[source_points[indices], target_points[target_indices[indices, 0]]].astype(np.float64)
    probabilities = np.exp(-target_distances[indices, 0] / max(float(np.median(target_distances[indices, 0])), 1e-6))
    model, inliers = pygcransac.findRigidTransform(
        correspondences, probabilities.astype(np.float64), threshold=threshold, conf=0.999,
        max_iters=20000, neighborhood=0, use_space_partitioning=True,
    )
    if model is None:
        return None, len(indices), 0
    return np.asarray(model).T, len(indices), int(inliers.sum())


def main():
    data_dir = ROOT / "bunny" / "data"
    poses = parse_bun_conf(data_dir / "bun.conf")
    rows = []
    for source_name, target_name in PAIRS:
        source, target = read_points(data_dir / f"{source_name}.ply"), read_points(data_dir / f"{target_name}.ply")
        ground_truth = relative_transform(poses[source_name], poses[target_name])
        for voxel in SCALES:
            for ratio in RATIOS:
                for threshold in THRESHOLDS:
                    transform, matches, inliers = solve(source, target, voxel, ratio, threshold)
                    row = {"pair": f"{source_name}->{target_name}", "voxel": voxel, "ratio": ratio,
                           "threshold": threshold, "matches": matches, "inliers": inliers}
                    if transform is not None:
                        rre, rte = pose_errors(transform, ground_truth)
                        row.update(rotation_error_deg=rre, translation_error=rte)
                    rows.append(row)
    output = ROOT / "outputs" / "gcransac_diagnostic.json"
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
