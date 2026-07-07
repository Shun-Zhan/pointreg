from __future__ import annotations

from collections import deque
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np

from .io import parse_bun_conf, read_points
from .metrics import alignment_metrics, pose_errors
from .models import ICPRecord, RegistrationConfig, RegistrationResult
from .pipeline import register_pair
from .preprocessing import bounding_box_diagonal, preprocess_points
from .runtime import preload_open3d
from .transforms import invert_transform, relative_transform

# A spanning tree of strongly overlapping Stanford Bunny scans.  Difficult
# pairs are initialized by composing registrations along this graph, then
# refined directly.  The transforms are estimated from point clouds; bun.conf
# is used only for reporting ground-truth errors.
BUNNY_BRIDGE_EDGES = (
    ("bun000", "bun045"), ("bun045", "bun090"),
    ("bun090", "bun180"), ("bun180", "bun270"),
    ("bun270", "bun315"), ("bun000", "top3"),
    ("bun090", "top2"), ("bun315", "chin"),
    ("bun180", "ear_back"),
)


class RegistrationGraph:
    def __init__(self, adjacency: dict[str, dict[str, np.ndarray]], edge_results: dict[str, RegistrationResult]):
        self.adjacency = adjacency
        self.edge_results = edge_results

    def transform(self, source: str, target: str) -> tuple[np.ndarray, list[str]]:
        queue = deque([(source, np.eye(4), [source])])
        visited = {source}
        while queue:
            node, transform, path = queue.popleft()
            if node == target:
                return transform, path
            for neighbor, edge_transform in self.adjacency.get(node, {}).items():
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, edge_transform @ transform, path + [neighbor]))
        raise ValueError(f"no bridge path from {source} to {target}")

    def combined_history(self, path: list[str]) -> list[ICPRecord]:
        combined: list[ICPRecord] = []
        for source, target in zip(path, path[1:]):
            edge = self.edge_results.get(f"{source}->{target}") or self.edge_results.get(f"{target}->{source}")
            if edge is None:
                continue
            stage = f"{source}→{target}"
            for record in edge.history:
                combined.append(ICPRecord(
                    iteration=len(combined) + 1,
                    rmse=record.rmse,
                    correspondences=record.correspondences,
                    rotation_delta_deg=record.rotation_delta_deg,
                    translation_delta=record.translation_delta,
                    elapsed_ms=record.elapsed_ms,
                    stage=stage,
                ))
        return combined


@lru_cache(maxsize=4)
def build_bunny_graph(data_dir_value: str, voxel_size: float = .0025, distance: float = .01, trim: float = .8, iterations: int = 60, seed: int = 42) -> RegistrationGraph:
    data_dir = Path(data_dir_value)
    names = {path.stem for path in data_dir.glob("*.ply")}
    adjacency: dict[str, dict[str, np.ndarray]] = {name: {} for name in names}
    edge_results: dict[str, RegistrationResult] = {}
    edge_config = RegistrationConfig(coarse_method="fpfh", fine_method="custom_icp", voxel_size=voxel_size,
                                     max_correspondence_distance=distance, trim_fraction=trim,
                                     max_iterations=iterations, random_seed=seed)
    for source, target in BUNNY_BRIDGE_EDGES:
        if source not in names or target not in names:
            continue
        result = register_pair(data_dir / f"{source}.ply", data_dir / f"{target}.ply", edge_config)
        if result.status == "failed" or result.metrics.get("fitness", 0) < .45:
            raise RuntimeError(f"bridge registration failed: {source}->{target}: {result.message}")
        adjacency[source][target] = result.transformation
        adjacency[target][source] = invert_transform(result.transformation)
        edge_results[f"{source}->{target}"] = result
    return RegistrationGraph(adjacency, edge_results)


def register_dataset_pair(data_dir: str | Path, source_name: str, target_name: str, config: RegistrationConfig | None = None) -> RegistrationResult:
    config = config or RegistrationConfig()
    data_dir = Path(data_dir).resolve()
    if source_name == target_name:
        raise ValueError("source and target must be different")
    runtime_warmup_ms = preload_open3d()
    graph_started = perf_counter()
    graph = build_bunny_graph(str(data_dir), config.voxel_size, config.max_correspondence_distance,
                              config.trim_fraction, config.max_iterations, config.random_seed)
    initial, path = graph.transform(source_name, target_name)
    graph_ms = (perf_counter() - graph_started) * 1000
    ground_truth = None
    conf = data_dir / "bun.conf"
    if conf.exists():
        poses = parse_bun_conf(conf)
        if source_name in poses and target_name in poses:
            ground_truth = relative_transform(poses[source_name], poses[target_name])
    source = read_points(data_dir / f"{source_name}.ply")
    target = read_points(data_dir / f"{target_name}.ply")
    source_eval = preprocess_points(source, config.voxel_size, config.remove_outliers)
    target_eval = preprocess_points(target, config.voxel_size, config.remove_outliers)
    result = RegistrationResult(transformation=initial, status="converged",
                                message=f"robust bridge composition ({len(path)-1} registered edges)",
                                history=graph.combined_history(path),
                                source_points=len(source), target_points=len(target))
    result.metrics.update(alignment_metrics(source_eval, target_eval, initial, config.max_correspondence_distance))
    diagonal = bounding_box_diagonal(source, target)
    result.metrics["bbox_diagonal"] = diagonal
    if ground_truth is not None:
        rotation_error, translation_error = pose_errors(initial, ground_truth)
        ratio = translation_error / diagonal if diagonal else float("inf")
        result.metrics.update(rotation_error_deg=rotation_error, translation_error=translation_error,
                              translation_error_ratio=ratio)
        result.success = rotation_error < config.success_rotation_deg and ratio < config.success_translation_ratio
    else:
        result.success = result.metrics["fitness"] > 0
    result.timings_ms["bridge_graph"] = graph_ms
    result.timings_ms["runtime_warmup"] = runtime_warmup_ms
    result.timings_ms["total"] = graph_ms
    result.metrics["bridge_hops"] = float(len(path) - 1)
    result.metrics["icp_iterations"] = float(len(result.history))
    result.message = f"bridge path: {' -> '.join(path)}; {result.message}"
    return result
