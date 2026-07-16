"""Adapter for the official GeoTransformer ModelNet checkpoint.

The upstream implementation remains vendored under ``third_party``.  This
module only normalizes PointReg clouds, prepares the original packed input, and
converts its source-to-reference estimate back to PointReg coordinates.

中文说明（课程设计讲解）：
本模块是 GeoTransformer 深度学习配准模型的“适配器 / 桥接层”。
GeoTransformer 的官方源码原封不动地放在 third_party 目录下，本文件不改动它，
只负责三件事：
  1) 把 PointReg 项目里的点云“归一化”成模型期望的坐标与尺度；
  2) 按官方要求把点云打包（collate）成模型 forward 所需的输入字典；
  3) 把模型输出的 source→reference（源→目标）位姿，换算回 PointReg 的原始坐标系。
这样上层配准流程就能像调用一个普通函数一样使用这个深度模型。
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


# 下面这些常量用来定位官方源码、两套实验配置目录以及两个预训练权重文件。
# GeoTransformer 提供了 ModelNet（合成物体）和 3DMatch（真实室内场景）两个版本，
# 本项目两个都会用到，因此各自维护一份实验目录与 checkpoint 路径。
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # 项目根目录（本文件的上一级的上一级）
UPSTREAM_ROOT = PROJECT_ROOT / "third_party" / "GeoTransformer-main"  # 官方 GeoTransformer 源码根目录
EXPERIMENT_ROOT = UPSTREAM_ROOT / "experiments" / "geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn"  # ModelNet 实验（含 config/model）
EXPERIMENT_3DMATCH_ROOT = UPSTREAM_ROOT / "experiments" / "geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn"  # 3DMatch 实验（含 config/model）
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "geotransformer-modelnet.pth.tar"  # ModelNet 默认权重
DEFAULT_3DMATCH_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "geotransformer-3dmatch.pth.tar"  # 3DMatch 默认权重
# 3DMatch 骨干网络每一层（4 个下采样阶段）的近邻数上限，需与官方预处理保持一致。
NEIGHBOR_LIMITS_3DMATCH = [38, 36, 36, 38]


@dataclass(frozen=True, slots=True)
class GeoTransformerCorrespondences:
    """Dense 3DMatch correspondences and LGR pose in PointReg coordinates.

    中文说明：封装 3DMatch 版 GeoTransformer 的稠密对应点结果，已换算回 PointReg 坐标系。
    该数据类是只读的（frozen），用于把模型输出打包传给上层融合流程。

    字段含义：
      source_points: (N,3) 源点云上被匹配到的点坐标；
      target_points: (N,3) 目标点云上与之对应的点坐标（与 source_points 一一对应）；
      scores:        (N,)  每对对应点的置信度/得分，已归一化到 [0,1]；
      lgr_transform: (4,4) 模型自带的 LGR（Local-to-Global Registration）位姿估计，
                          即源→目标的刚体变换矩阵；
      scale:         模型内部工作尺度相对于 PointReg 坐标的缩放系数（用于坐标还原）。
    """

    source_points: np.ndarray
    target_points: np.ndarray
    scores: np.ndarray
    lgr_transform: np.ndarray
    scale: float


def _sample_points(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Sample a fixed number of points deterministically, matching ModelNet size.

    中文说明：把任意点数的点云“采样”成固定数量 count 的点，以匹配 ModelNet 模型
    训练时的输入规模。使用带固定随机种子的 rng，保证结果可复现（确定性）。

    参数：
      points: (M,3) 输入点云；
      count:  目标采样点数；
      seed:   随机种子，保证每次采样结果一致。
    返回：(count,3) 的 float32 点云。
    """
    if len(points) < 3:
        # 少于 3 个点无法构成有效的三维配准约束，直接报错。
        raise ValueError("GeoTransformer needs at least three points per cloud")
    rng = np.random.default_rng(seed)
    # 点数不足时允许重复采样（放回），点数充足时不放回，避免重复。
    replace = len(points) < count
    indices = rng.choice(len(points), size=count, replace=replace)
    return np.asarray(points[indices], dtype=np.float32)


def _normalize_pair(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Apply one shared similarity transform so the rigid pose is preserved.

    中文说明：对源、目标两片点云施加“同一个”相似变换（先平移到公共中心、再统一缩放），
    把它们归一化到 ModelNet 模型习惯的单位球范围内。
    关键点：两片点云必须用同一个 center 和 scale，否则会破坏它们之间真实的刚体位姿关系，
    导致模型学到的旋转/平移无法正确还原。

    返回：(归一化后的 source, 归一化后的 target, 公共中心 center, 缩放系数 scale)，
          后两者用于把模型输出的位姿还原回原始坐标系。
    """
    joined = np.concatenate([source, target], axis=0)  # 两片点云合并后统计公共几何范围
    center = joined.mean(axis=0)  # 公共质心，作为平移原点
    # 以离中心最远点的距离作为缩放尺度，使归一化后点云落在半径为 1 的球内。
    scale = float(np.max(np.linalg.norm(joined - center, axis=1)))
    if not np.isfinite(scale) or scale <= 0:
        # 尺度非法（点云退化为一点等）时无法归一化。
        raise ValueError("cannot normalize a degenerate point-cloud pair")
    return (source - center) / scale, (target - center) / scale, center, scale


def _move_to_device(value, device):
    """递归地把嵌套结构（list/tuple/dict）里的所有张量搬到指定设备（CPU/GPU）。

    中文说明：GeoTransformer 打包后的输入是一个层层嵌套的字典（里面还有列表、张量），
    forward 之前必须整体搬到运行设备上。这里用递归遍历，遇到张量就 .to(device)，
    非张量（如普通数值）原样返回。
    """
    import torch

    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return value


def _to_numpy(value) -> np.ndarray:
    """把 torch 张量安全转换成 numpy 数组（先 detach 脱离计算图，再搬回 CPU）。

    中文说明：模型输出是带梯度、可能位于 GPU 上的张量；转成 numpy 前必须
    detach + cpu，否则会报错。若本来就是 numpy/普通数组则直接封装返回。
    """
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _decode_3dmatch_output(output_dict: dict, scale: float) -> GeoTransformerCorrespondences:
    """Convert upstream 3DMatch outputs back from its working scale.

    中文说明：模型是在“放大 scale 倍”的工作尺度上推理的（为了匹配它训练用的体素大小），
    因此所有坐标类输出（对应点、位姿的平移分量）都要除以 scale 才能回到 PointReg 原始尺度。
    本函数负责：读取输出、做尺度还原、做形状/数值合法性校验、把得分归一化，最后打包成数据类。

    参数：
      output_dict: 模型 forward 返回的字典；
      scale:       推理时使用的缩放系数（工作尺度 / PointReg 尺度）。
    返回：还原到 PointReg 坐标系的 GeoTransformerCorrespondences。
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("3DMatch scale must be finite and positive")
    # 对应点坐标除以 scale，从模型工作尺度还原到 PointReg 原始尺度。
    # 注意：src=源(source)、ref=参考/目标(reference/target)，这里保持该命名对应关系。
    source = _to_numpy(output_dict["src_corr_points"]).astype(np.float64, copy=False) / scale
    target = _to_numpy(output_dict["ref_corr_points"]).astype(np.float64, copy=False) / scale
    scores = _to_numpy(output_dict["corr_scores"]).astype(np.float64, copy=False).reshape(-1)
    estimate_s = _to_numpy(output_dict["estimated_transform"]).astype(np.float64, copy=False)
    # 以下几步是防御式校验：确保对应点是成对的 (N,3)、得分与对应点数量一致、位姿是 4x4、无 NaN/Inf。
    if source.ndim != 2 or source.shape[1] != 3 or target.shape != source.shape:
        raise ValueError("GeoTransformer correspondence points must be matching (N, 3) arrays")
    if len(scores) != len(source):
        raise ValueError("GeoTransformer scores must match the correspondence count")
    if estimate_s.shape != (4, 4):
        raise ValueError("GeoTransformer estimated transform must have shape (4, 4)")
    if not (np.isfinite(source).all() and np.isfinite(target).all() and np.isfinite(scores).all()):
        raise ValueError("GeoTransformer returned non-finite correspondences")
    # 得分归一化：先截断负值，再除以最大值缩放到 [0,1]；若全为 0 则退化为全 1（等权重）。
    scores = np.maximum(scores, 0.0)
    maximum = float(scores.max()) if len(scores) else 0.0
    scores = scores / maximum if maximum > 0 else np.ones(len(scores), dtype=np.float64)
    # 位姿矩阵的旋转部分是无量纲的，只需把平移分量（第 4 列前 3 行）除以 scale 还原尺度。
    estimate = estimate_s.copy()
    estimate[:3, 3] /= scale
    return GeoTransformerCorrespondences(source, target, scores, estimate, float(scale))


def _enable_upstream_cpu_compat(torch) -> None:
    """Make the CUDA-hardcoded 2022 upstream inference code run on CPU.

    GeoTransformer allocates intermediate tensors with ``tensor.cuda()`` in
    several inference-only modules.  Those allocations should follow the
    existing CPU tensors when the optional CUDA wheel is not installed.

    中文说明：2022 年的官方推理代码里硬编码了很多 tensor.cuda() 调用，
    在没有安装 CUDA 的纯 CPU 环境下会直接报错。这里用“猴子补丁”（monkey patch）
    把 Tensor.cuda 改成空操作，让这些张量留在 CPU 上，从而无需改动官方源码就能在 CPU 跑通。
    """
    # 若本机有 CUDA，或已经打过补丁，则无需处理，直接返回。
    if torch.cuda.is_available() or getattr(torch.Tensor.cuda, "_pointreg_cpu_cuda_patch", False):
        return

    # 用一个“空操作”替换 tensor.cuda()：直接返回张量自身（保持在 CPU 上）。
    def _cpu_cuda(self, *args, **kwargs):
        return self

    # 打上标记，避免重复 patch；然后猴子补丁替换掉 Tensor.cuda 方法。
    _cpu_cuda._pointreg_cpu_cuda_patch = True
    torch.Tensor.cuda = _cpu_cuda


def _import_experiment(experiment_root: Path):
    """Import an upstream experiment without leaking its generic module names.

    中文说明：官方实验目录里的模块名很“通用”（config、model、backbone），
    直接 import 会污染全局 sys.modules，还可能和 ModelNet / 3DMatch 两套实验互相覆盖。
    本函数做了“隔离导入”：
      1) 先备份并清掉这些通用名的已有模块；
      2) 临时把实验目录塞进 sys.path 最前面，导入 config 和 model；
      3) 无论成功与否，都在 finally 里恢复 sys.path 和原有模块，避免留下副作用。
    返回：(config 模块, model 模块)。
    """
    module_names = ("config", "model", "backbone")
    previous = {name: sys.modules.get(name) for name in module_names}  # 备份可能已存在的同名模块
    for name in module_names:
        sys.modules.pop(name, None)  # 先清空，避免拿到旧的/别的实验的缓存
    old_path = sys.path.copy()  # 备份原始搜索路径
    try:
        # 把官方根目录与具体实验目录插到搜索路径最前，保证优先导入到它们。
        sys.path.insert(0, str(UPSTREAM_ROOT))
        sys.path.insert(0, str(experiment_root))
        config = importlib.import_module("config")
        model_module = importlib.import_module("model")
    finally:
        # 关键的清理步骤：恢复搜索路径，删掉本次导入的通用名，再还原备份，做到“无痕”。
        sys.path[:] = old_path
        for name in module_names:
            sys.modules.pop(name, None)
        for name, module in previous.items():
            if module is not None:
                sys.modules[name] = module
    return config, model_module


@lru_cache(maxsize=2)  # 缓存已加载模型：同一 checkpoint 只加载一次，避免重复读盘和建网
def _load_model(checkpoint_path: str):
    """加载 ModelNet 版 GeoTransformer 模型与其配置（cfg）。

    中文说明：构建网络、加载预训练权重、切到推理模式（eval），并做完整性校验。
    加了 lru_cache，同一权重路径重复调用时会命中缓存直接返回。
    返回：(cfg 配置对象, 已加载好的 model)。
    """
    import torch

    checkpoint = Path(checkpoint_path)
    if not UPSTREAM_ROOT.is_dir():
        raise RuntimeError(f"GeoTransformer source not found: {UPSTREAM_ROOT}")
    # 校验权重文件存在且大小合理（<20MB 视为缺失或下载不完整）。
    if not checkpoint.is_file() or checkpoint.stat().st_size < 20_000_000:
        raise RuntimeError(
            f"GeoTransformer ModelNet checkpoint is missing or incomplete: {checkpoint}. "
            "Download geotransformer-modelnet.pth.tar into checkpoints/."
        )
    config, model_module = _import_experiment(EXPERIMENT_ROOT)  # 隔离导入 ModelNet 实验的 config/model
    cfg = config.make_cfg()  # 生成官方配置
    model = model_module.create_model(cfg)  # 按配置构建网络结构
    snapshot = torch.load(checkpoint, map_location="cpu", weights_only=False)  # 权重先加载到 CPU
    state_dict = snapshot.get("model", snapshot)  # 兼容两种保存格式：{"model": ...} 或直接是 state_dict
    # ModelNet 权重用 strict=False 容错加载，但随后严格检查是否有缺失/多余的参数。
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "GeoTransformer checkpoint architecture mismatch "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
    model.eval()  # 切换到推理模式（关闭 dropout / BN 更新等）
    return cfg, model


@lru_cache(maxsize=2)  # 同样缓存，避免重复加载 3DMatch 模型
def _load_3dmatch_model(checkpoint_path: str):
    """Load the official 3DMatch architecture without changing ModelNet defaults.

    中文说明：加载 3DMatch 版模型。与 ModelNet 版几乎一致，区别在于导入的是 3DMatch 实验，
    且这里用 strict=True 严格加载权重（要求参数完全对齐）。3DMatch 版是本项目做真实场景
    低重叠配准的主力模型。
    返回：(cfg, model)。
    """
    import torch

    checkpoint = Path(checkpoint_path)
    if not UPSTREAM_ROOT.is_dir():
        raise RuntimeError(f"GeoTransformer source not found: {UPSTREAM_ROOT}")
    if not checkpoint.is_file() or checkpoint.stat().st_size < 20_000_000:
        raise RuntimeError(f"GeoTransformer 3DMatch checkpoint is missing or incomplete: {checkpoint}")
    config, model_module = _import_experiment(EXPERIMENT_3DMATCH_ROOT)  # 隔离导入 3DMatch 实验
    cfg = config.make_cfg()
    model = model_module.create_model(cfg)
    snapshot = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = snapshot.get("model", snapshot)
    model.load_state_dict(state_dict, strict=True)  # 严格加载：结构与权重必须完全匹配
    model.eval()
    return cfg, model


def geotransformer_3dmatch_correspondences(
    source: np.ndarray,
    target: np.ndarray,
    *,
    checkpoint: str | Path | None = None,
    voxel_size: float = 0.0025,
) -> GeoTransformerCorrespondences:
    """Run 3DMatch GeoTransformer and expose its dense source-to-target matches.

    中文说明：这是给上层融合流程用的主入口之一。它运行 3DMatch 版 GeoTransformer，
    输出“稠密对应点”（source 与 target 上一一对应的点对）及其置信度，供后续 RANSAC/
    GC-RANSAC 等鲁棒后端拟合位姿。

    参数：
      source, target: (N,3) 源/目标点云（PointReg 坐标系，单位与真实尺度一致）；
      checkpoint:     权重路径，默认用 3DMatch 权重；
      voxel_size:     PointReg 侧的体素大小，用于计算缩放系数 scale。
    返回：GeoTransformerCorrespondences（已还原回 PointReg 坐标）。
    """
    import torch

    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive for 3DMatch inference")
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    # 输入合法性校验：必须是 (N,3) 且点数 >= 3。
    if source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("source must be an (N, 3) array with N >= 3")
    if target.ndim != 2 or target.shape[1] != 3 or len(target) < 3:
        raise ValueError("target must be an (N, 3) array with N >= 3")
    checkpoint_path = Path(checkpoint) if checkpoint is not None else DEFAULT_3DMATCH_CHECKPOINT
    _enable_upstream_cpu_compat(torch)  # 无 GPU 时打补丁让官方代码在 CPU 跑通
    cfg, model = _load_3dmatch_model(str(checkpoint_path.resolve()))
    # 关键：尺度缩放系数 scale = 模型初始体素 / PointReg 体素。
    # 3DMatch 模型是在某个固定体素尺度上训练的，需把点云放大 scale 倍，
    # 使 PointReg 的体素间距与模型期望的体素间距对齐，模型才能正确匹配。
    scale = float(cfg.backbone.init_voxel_size / voxel_size)

    # 官方提供的“打包函数”，把单个样本按多阶段下采样（stack mode）整理成模型输入。
    from geotransformer.utils.data import registration_collate_fn_stack_mode

    # 组装单个样本字典：ref=目标(reference)，src=源(source)；坐标统一乘以 scale 进入模型工作尺度。
    # feats 用全 1 占位（本模型主要靠几何结构而非特征）；transform 为单位阵仅用于形式对齐。
    sample = {
        "ref_points": (target * scale).astype(np.float32),
        "src_points": (source * scale).astype(np.float32),
        "ref_feats": np.ones((len(target), 1), dtype=np.float32),
        "src_feats": np.ones((len(source), 1), dtype=np.float32),
        "transform": np.eye(4, dtype=np.float32),
    }
    # 按骨干网络的阶段数、初始体素、初始半径和近邻上限做多层预处理打包。
    data_dict = registration_collate_fn_stack_mode(
        [sample],
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        NEIGHBOR_LIMITS_3DMATCH,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 优先 GPU，否则 CPU
    model = model.to(device)
    data_dict = _move_to_device(data_dict, device)  # 输入整体搬到同一设备
    with torch.no_grad():  # 推理阶段不需要梯度，省内存提速
        output_dict = model(data_dict)
    return _decode_3dmatch_output(output_dict, scale)  # 解码 + 尺度还原


def geotransformer_registration(
    source: np.ndarray,
    target: np.ndarray,
    *,
    checkpoint: str | Path | None = None,
    num_points: int = 717,
    seed: int = 42,
) -> np.ndarray:
    """Estimate the PointReg source-to-target transform with GeoTransformer.

    中文说明：使用 ModelNet 版 GeoTransformer 直接估计源→目标的 4x4 刚体位姿。
    与 3DMatch 入口不同，这里走的是“归一化 + 定点采样”的路线（贴合 ModelNet 训练设定），
    并且需要把模型在归一化坐标系下给出的位姿，反变换回 PointReg 原始坐标系。

    参数：
      source, target: (N,3) 源/目标点云；
      checkpoint:     权重路径，默认用 ModelNet 权重；
      num_points:     采样后固定点数（默认 717，与 ModelNet 设定一致）；
      seed:           采样随机种子（目标云用 seed+1 以错开）。
    返回：(4,4) 源→目标刚体变换矩阵（PointReg 坐标系）。
    """
    import torch

    checkpoint_path = Path(checkpoint) if checkpoint is not None else DEFAULT_CHECKPOINT
    _enable_upstream_cpu_compat(torch)
    cfg, model = _load_model(str(checkpoint_path.resolve()))
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    # 用共享的相似变换把两片点云归一化，并保存 center/scale 供后面反变换。
    source_n, target_n, center, scale = _normalize_pair(source, target)
    # 归一化后再定点采样到固定点数（源、目标用不同种子，避免采到完全对齐的索引）。
    source_n = _sample_points(source_n, num_points, seed)
    target_n = _sample_points(target_n, num_points, seed + 1)

    from geotransformer.utils.data import registration_collate_fn_stack_mode

    sample = {
        "ref_points": target_n,
        "src_points": source_n,
        "ref_feats": np.ones((len(target_n), 1), dtype=np.float32),
        "src_feats": np.ones((len(source_n), 1), dtype=np.float32),
        # Only used to form training labels inside the upstream forward pass;
        # it does not participate in inference matching.
        # 中文：transform 仅在官方 forward 内部用于构造训练标签，推理匹配时不参与，
        # 因此这里给单位阵占位即可。
        "transform": np.eye(4, dtype=np.float32),
    }
    data_dict = registration_collate_fn_stack_mode(
        [sample],
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        [64] * cfg.backbone.num_stages,  # ModelNet 各阶段近邻上限统一取 64
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data_dict = _move_to_device(data_dict, device)
    with torch.no_grad():
        # 取出模型在“归一化坐标系”下估计的位姿（下标 n 表示 normalized）。
        estimate_n = model(data_dict)["estimated_transform"].detach().cpu().numpy()
    rotation = estimate_n[:3, :3]  # 旋转部分：相似变换下旋转不变，可直接沿用
    # 关键的坐标系反变换（把归一化坐标下的位姿还原到原始坐标）：
    # 归一化时 p_n = (p - center)/scale，模型给出 p_n' = R·p_n + t_n。
    # 代回原始坐标 p' = R·p + (scale·t_n + center - R·center)，故平移项如下推导：
    translation = scale * estimate_n[:3, 3] + center - rotation @ center
    estimate = np.eye(4)
    estimate[:3, :3] = rotation
    estimate[:3, 3] = translation
    return estimate
