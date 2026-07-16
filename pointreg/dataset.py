"""数据集级配准：用“桥接图（bridge graph）”策略解决低重叠两帧的直接配准难题。

核心思路：兔子相邻视角（如相差 45°）重叠大、容易配准；而相隔很远的两帧（如
bun000 与 bun270）重叠太小，直接配几乎必失败。于是先在若干“强重叠相邻对”上做
可靠配准，构成一张连通图，再对任意难配对沿图中路径把这些可靠变换串联（复合）起来
得到初值。真值 bun.conf 只用于最后报告误差，不参与变换估计。
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

# 强重叠兔子扫描对构成的“生成树/桥接边”集合。难配对通过沿这张图复合各段可靠配准
# 得到初值，再直接精配。这些边上的变换全部由点云估计得出；bun.conf 仅用于报告真值误差。
BUNNY_BRIDGE_EDGES = (
    ("bun000", "bun045"), ("bun045", "bun090"),
    ("bun090", "bun180"), ("bun180", "bun270"),
    ("bun270", "bun315"), ("bun000", "top3"),
    ("bun090", "top2"), ("bun315", "chin"),
    ("bun180", "ear_back"),
)


class RegistrationGraph:
    """配准桥接图：节点是扫描帧，边是相邻帧之间已可靠估计出的变换。

    adjacency 是双向邻接表 {源: {邻居: 源->邻居的变换}}；edge_results 保存每条边
    完整的配准结果（含迭代历史），供后续拼接出复合路径的过程记录。
    """

    def __init__(self, adjacency: dict[str, dict[str, np.ndarray]], edge_results: dict[str, RegistrationResult]):
        self.adjacency = adjacency
        self.edge_results = edge_results

    def transform(self, source: str, target: str) -> tuple[np.ndarray, list[str]]:
        """在桥接图上用 BFS 找 source 到 target 的路径，并把沿途变换复合成总变换。

        返回 (复合变换, 途经节点路径)。BFS 保证得到边数最少的路径，即复合的段数最少，
        以减少变换累乘带来的误差累积。图中不连通时抛异常。
        """
        # 队列元素为 (当前节点, 从 source 到当前节点的累积变换, 途经路径)。
        queue = deque([(source, np.eye(4), [source])])
        visited = {source}
        while queue:
            node, transform, path = queue.popleft()
            if node == target:
                return transform, path
            for neighbor, edge_transform in self.adjacency.get(node, {}).items():
                if neighbor not in visited:
                    visited.add(neighbor)
                    # 左乘新一段边的变换实现复合：先前累积变换，再叠加 node->neighbor。
                    queue.append((neighbor, edge_transform @ transform, path + [neighbor]))
        raise ValueError(f"no bridge path from {source} to {target}")

    def combined_history(self, path: list[str]) -> list[ICPRecord]:
        """把路径上各段边的 ICP 迭代历史依序拼接成一条统一的收敛历史。

        为便于整体展示与答辩讲解，重新编排全局迭代序号 iteration，并给每条记录
        打上 stage 标签（形如“源→目标”）标明它来自哪一段桥接边。
        """
        combined: list[ICPRecord] = []
        for source, target in zip(path, path[1:]):
            # 边可能以任一方向存储，正反查一次；找不到就跳过该段。
            edge = self.edge_results.get(f"{source}->{target}") or self.edge_results.get(f"{target}->{source}")
            if edge is None:
                continue
            stage = f"{source}→{target}"
            for record in edge.history:
                # 复制原始记录内容，仅重排全局迭代号并补充 stage 归属。
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
    """构建兔子数据集的桥接图：在所有预设强重叠对上做配准，得到边及其变换。

    用 lru_cache 缓存：同一组参数只建一次图（建图要跑多次配准，开销大），
    后续不同的难配对可复用同一张图。任一桥接边配准失败或质量太差（fitness<0.45）
    都会抛异常，因为图不可靠则整套复合策略都不可信。
    """
    data_dir = Path(data_dir_value)
    names = {path.stem for path in data_dir.glob("*.ply")}  # 数据目录里实际存在的扫描帧。
    adjacency: dict[str, dict[str, np.ndarray]] = {name: {} for name in names}
    edge_results: dict[str, RegistrationResult] = {}
    # 建图统一用 FPFH 粗配 + 自定义 ICP 精配的稳健配置。
    edge_config = RegistrationConfig(coarse_method="fpfh", fine_method="custom_icp", voxel_size=voxel_size,
                                     max_correspondence_distance=distance, trim_fraction=trim,
                                     max_iterations=iterations, random_seed=seed)
    for source, target in BUNNY_BRIDGE_EDGES:
        if source not in names or target not in names:
            continue  # 数据集里缺这帧就跳过该边。
        result = register_pair(data_dir / f"{source}.ply", data_dir / f"{target}.ply", edge_config)
        if result.status == "failed" or result.metrics.get("fitness", 0) < .45:
            raise RuntimeError(f"bridge registration failed: {source}->{target}: {result.message}")
        # 正向存 source->target，反向存其逆，使图无向可双向遍历。
        adjacency[source][target] = result.transformation
        adjacency[target][source] = invert_transform(result.transformation)
        edge_results[f"{source}->{target}"] = result
    return RegistrationGraph(adjacency, edge_results)


def register_dataset_pair(data_dir: str | Path, source_name: str, target_name: str, config: RegistrationConfig | None = None) -> RegistrationResult:
    """配准数据集中指定的两帧，返回带完整误差指标与分阶段计时的结果。

    整体流程：预热 Open3D -> 建/取桥接图 -> 在图上找路径并复合出初始变换 ->
    读真值与点云 -> 预处理 -> 评估对齐质量与位姿误差 -> 记录各阶段耗时。
    每个阶段都单独计时，便于在答辩中分析性能瓶颈。
    """
    config = config or RegistrationConfig()
    data_dir = Path(data_dir).resolve()
    if source_name == target_name:
        raise ValueError("source and target must be different")
    runtime_warmup_ms = preload_open3d()  # 先预热，避免把 Open3D 加载开销算进后续阶段。
    total_started = perf_counter()
    # 阶段一：建图并沿路径复合出初始变换（这里正是低重叠对能配准的关键）。
    graph_started = perf_counter()
    graph = build_bunny_graph(str(data_dir), config.voxel_size, config.max_correspondence_distance,
                              config.trim_fraction, config.max_iterations, config.random_seed)
    initial, path = graph.transform(source_name, target_name)
    graph_ms = (perf_counter() - graph_started) * 1000
    # 阶段二：若存在 bun.conf，则读取真值并算出这两帧的相对真值位姿（仅用于评估）。
    metadata_started = perf_counter()
    ground_truth = None
    conf = data_dir / "bun.conf"
    if conf.exists():
        poses = parse_bun_conf(conf)
        if source_name in poses and target_name in poses:
            ground_truth = relative_transform(poses[source_name], poses[target_name])
    metadata_ms = (perf_counter() - metadata_started) * 1000
    # 阶段三：读取源、目标原始点云。
    load_started = perf_counter()
    source = read_points(data_dir / f"{source_name}.ply")
    target = read_points(data_dir / f"{target_name}.ply")
    load_ms = (perf_counter() - load_started) * 1000
    # 阶段四：预处理（下采样等），得到用于评估对齐质量的点云。
    preprocess_started = perf_counter()
    source_eval = preprocess_points(source, config.voxel_size, config.remove_outliers)
    target_eval = preprocess_points(target, config.voxel_size, config.remove_outliers)
    preprocess_ms = (perf_counter() - preprocess_started) * 1000
    # 组装结果对象：变换取复合初值，历史取拼接后的分阶段迭代记录。
    result = RegistrationResult(transformation=initial, status="converged",
                                message=f"robust bridge composition ({len(path)-1} registered edges)",
                                history=graph.combined_history(path),
                                source_points=len(source), target_points=len(target))
    # 阶段五：评估。先算与真值无关的对齐质量（RMSE/fitness 等）。
    evaluate_started = perf_counter()
    result.metrics.update(alignment_metrics(source_eval, target_eval, initial, config.max_correspondence_distance))
    diagonal = bounding_box_diagonal(source, target)  # 整体尺度，用于归一化平移误差。
    result.metrics["bbox_diagonal"] = diagonal
    if ground_truth is not None:
        # 有真值：计算旋转/平移误差，并把平移误差按包围盒对角线归一化成无量纲比例。
        rotation_error, translation_error = pose_errors(initial, ground_truth)
        ratio = translation_error / diagonal if diagonal else float("inf")
        result.metrics.update(rotation_error_deg=rotation_error, translation_error=translation_error,
                              translation_error_ratio=ratio)
        # 旋转、平移误差都在阈值内才判为成功。
        result.success = rotation_error < config.success_rotation_deg and ratio < config.success_translation_ratio
    else:
        # 无真值：退而用 fitness 是否为正粗略判断是否成功对齐。
        result.success = result.metrics["fitness"] > 0
    evaluate_ms = (perf_counter() - evaluate_started) * 1000
    # 汇总各阶段耗时（毫秒），便于性能分析。
    result.timings_ms["bridge_graph"] = graph_ms
    result.timings_ms["metadata"] = metadata_ms
    result.timings_ms["load"] = load_ms
    result.timings_ms["preprocess"] = preprocess_ms
    result.timings_ms["evaluate"] = evaluate_ms
    result.timings_ms["runtime_warmup"] = runtime_warmup_ms
    result.timings_ms["total"] = (perf_counter() - total_started) * 1000
    result.metrics["bridge_hops"] = float(len(path) - 1)  # 复合了几段桥接边（跳数）。
    result.metrics["icp_iterations"] = float(len(result.history))  # 总迭代次数。
    result.message = f"bridge path: {' -> '.join(path)}; {result.message}"  # 附上具体桥接路径。
    return result
