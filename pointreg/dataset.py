"""桥接法：借助高重合的“中转”扫描来配准低重合的困难点对。

课题难点在于像 bun000↔bun180 这类几乎不重合的点对，直接配准会失败。
思路是预先在一批两两高度重合的相邻扫描间建立可靠配准，构成一张“配准图”，
再把困难点对拆解为沿图中路径的若干跳，逐跳组合变换得到初值。
真值位姿（bun.conf）仅用于报告误差，不参与图的构建。
"""

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

# 预先选定的一组“强重合”相邻扫描边，构成连接各视角的生成树/图。困难点对
# 通过沿这些边组合已配好的变换来获得初值，再直接精配准。这些变换均由点云
# 估计得到；bun.conf 仅用于事后报告真值误差。
BUNNY_BRIDGE_EDGES = (
    ("bun000", "bun045"), ("bun045", "bun090"),
    ("bun090", "bun180"), ("bun180", "bun270"),
    ("bun270", "bun315"), ("bun000", "top3"),
    ("bun090", "top2"), ("bun315", "chin"),
    ("bun180", "ear_back"),
)


class RegistrationGraph:
    """配准图：节点是各扫描，边保存两扫描间已估计好的相对变换。

    adjacency[a][b] 给出把 a 对齐到 b 的 4×4 变换；edge_results 保存每条边
    配准的完整结果（含 ICP 历史），供组合与展示使用。
    """

    def __init__(self, adjacency: dict[str, dict[str, np.ndarray]], edge_results: dict[str, RegistrationResult]):
        self.adjacency = adjacency
        self.edge_results = edge_results

    def transform(self, source: str, target: str) -> tuple[np.ndarray, list[str]]:
        """在图上用 BFS 找 source→target 的最短跳数路径，并沿途组合变换。

        返回 (组合后的 4×4 变换, 途经的节点名列表)。因为 BFS 按跳数扩展，
        得到的是边数最少的桥接路径，能减少变换组合带来的误差累积。
        """
        # 队列元素：(当前节点, 从起点累积到此的变换, 路径)
        queue = deque([(source, np.eye(4), [source])])
        visited = {source}
        while queue:
            node, transform, path = queue.popleft()
            if node == target:
                return transform, path
            for neighbor, edge_transform in self.adjacency.get(node, {}).items():
                if neighbor not in visited:
                    visited.add(neighbor)
                    # 左乘新边变换，得到起点到 neighbor 的组合变换
                    queue.append((neighbor, edge_transform @ transform, path + [neighbor]))
        raise ValueError(f"no bridge path from {source} to {target}")

    def combined_history(self, path: list[str]) -> list[ICPRecord]:
        """把路径上各条边的 ICP 迭代历史拼接成一条完整历史，供 UI 展示收敛过程。"""
        combined: list[ICPRecord] = []
        for source, target in zip(path, path[1:]):
            # 边可能以任一方向存储，两个方向都查一下
            edge = self.edge_results.get(f"{source}->{target}") or self.edge_results.get(f"{target}->{source}")
            if edge is None:
                continue
            stage = f"{source}→{target}"  # 标注该段历史属于哪条桥接边
            for record in edge.history:
                combined.append(ICPRecord(
                    iteration=len(combined) + 1,  # 重新连续编号，便于整体画收敛曲线
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
    """构建 Bunny 配准图：对每条预设强重合边跑一次配准并缓存结果。

    用 @lru_cache 缓存：同参数只会构建一次，避免 UI/批量评测里反复重建（较耗时）。
    每条边要求配准成功且 fitness≥0.45，否则说明这条“中转”不可靠，直接报错，
    因为它会污染所有经过它的桥接路径。同时存正/反两个方向便于双向 BFS。
    """
    data_dir = Path(data_dir_value)
    names = {path.stem for path in data_dir.glob("*.ply")}
    adjacency: dict[str, dict[str, np.ndarray]] = {name: {} for name in names}
    edge_results: dict[str, RegistrationResult] = {}
    edge_config = RegistrationConfig(coarse_method="fpfh", fine_method="custom_icp", voxel_size=voxel_size,
                                     max_correspondence_distance=distance, trim_fraction=trim,
                                     max_iterations=iterations, random_seed=seed)
    for source, target in BUNNY_BRIDGE_EDGES:
        if source not in names or target not in names:
            continue  # 数据集缺少该扫描则跳过这条边
        result = register_pair(data_dir / f"{source}.ply", data_dir / f"{target}.ply", edge_config)
        if result.status == "failed" or result.metrics.get("fitness", 0) < .45:
            raise RuntimeError(f"bridge registration failed: {source}->{target}: {result.message}")
        adjacency[source][target] = result.transformation                 # 正向变换
        adjacency[target][source] = invert_transform(result.transformation)  # 反向取逆
        edge_results[f"{source}->{target}"] = result
    return RegistrationGraph(adjacency, edge_results)


def register_dataset_pair(data_dir: str | Path, source_name: str, target_name: str, config: RegistrationConfig | None = None) -> RegistrationResult:
    """用桥接法配准数据集中一对具名扫描（按名字定位 PLY 与真值位姿）。

    与 register_pair 的直接两帧配准不同：这里先构建/复用配准图，沿最短路径
    组合出初始变换，把它作为最终结果（桥接边本身已各自精配好），再对该初值
    做对齐评估与真值误差计算。适用于直接配准失败的低重合困难点对。
    """
    config = config or RegistrationConfig()
    data_dir = Path(data_dir).resolve()
    if source_name == target_name:
        raise ValueError("source and target must be different")
    runtime_warmup_ms = preload_open3d()
    total_started = perf_counter()
    # —— 构建/复用配准图并求桥接路径与组合变换 ——
    graph_started = perf_counter()
    graph = build_bunny_graph(str(data_dir), config.voxel_size, config.max_correspondence_distance,
                              config.trim_fraction, config.max_iterations, config.random_seed)
    initial, path = graph.transform(source_name, target_name)  # 组合变换作为最终位姿
    graph_ms = (perf_counter() - graph_started) * 1000
    # —— 读取真值位姿（若有），仅用于评估误差 ——
    metadata_started = perf_counter()
    ground_truth = None
    conf = data_dir / "bun.conf"
    if conf.exists():
        poses = parse_bun_conf(conf)
        if source_name in poses and target_name in poses:
            ground_truth = relative_transform(poses[source_name], poses[target_name])
    metadata_ms = (perf_counter() - metadata_started) * 1000
    # —— 加载并预处理两片点云，用于计算对齐指标 ——
    load_started = perf_counter()
    source = read_points(data_dir / f"{source_name}.ply")
    target = read_points(data_dir / f"{target_name}.ply")
    load_ms = (perf_counter() - load_started) * 1000
    preprocess_started = perf_counter()
    source_eval = preprocess_points(source, config.voxel_size, config.remove_outliers)
    target_eval = preprocess_points(target, config.voxel_size, config.remove_outliers)
    preprocess_ms = (perf_counter() - preprocess_started) * 1000
    # 桥接组合出的变换直接作为结果（各边已分别精配准），并拼接各边 ICP 历史
    result = RegistrationResult(transformation=initial, status="converged",
                                message=f"robust bridge composition ({len(path)-1} registered edges)",
                                history=graph.combined_history(path),
                                source_points=len(source), target_points=len(target))
    # —— 评估：对齐指标 + 真值误差 + 成功判定（与 register_pair 一致）——
    evaluate_started = perf_counter()
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
    evaluate_ms = (perf_counter() - evaluate_started) * 1000
    # 汇总各阶段耗时（桥接图构建通常已被缓存，二次调用会很快）
    result.timings_ms["bridge_graph"] = graph_ms
    result.timings_ms["metadata"] = metadata_ms
    result.timings_ms["load"] = load_ms
    result.timings_ms["preprocess"] = preprocess_ms
    result.timings_ms["evaluate"] = evaluate_ms
    result.timings_ms["runtime_warmup"] = runtime_warmup_ms
    result.timings_ms["total"] = (perf_counter() - total_started) * 1000
    result.metrics["bridge_hops"] = float(len(path) - 1)          # 桥接跳数
    result.metrics["icp_iterations"] = float(len(result.history))  # 累计 ICP 迭代数
    # 把桥接路径写进说明，便于在结果里看到经过了哪些中转扫描
    result.message = f"bridge path: {' -> '.join(path)}; {result.message}"
    return result
