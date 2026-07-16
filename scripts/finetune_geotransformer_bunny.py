"""Small, leakage-free domain-adaptation run for GeoTransformer on Bunny scans.

The held-out source/target scans never enter the optimizer.  Every training
item is a direct two-frame pair with its transform from ``bun.conf`` and fresh
independent plane crops, emulating low-overlap partial scans.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 从项目的 geotransformer 封装里导入：默认权重路径、上游/实验代码根目录、
# CPU 兼容开关、以及点云归一化与采样的工具函数。
from pointreg.geotransformer import (
    DEFAULT_CHECKPOINT,
    EXPERIMENT_ROOT,
    UPSTREAM_ROOT,
    _enable_upstream_cpu_compat,
    _normalize_pair,
    _sample_points,
)
from pointreg.io import parse_bun_conf, read_points
from pointreg.transforms import relative_transform


def crop_with_plane(points: np.ndarray, keep_ratio: float, rng: np.random.Generator) -> np.ndarray:
    """用一个随机平面切掉一部分点，模拟低重叠的“部分扫描”。

    做法：随机取一个单位法向量，把每个点投影到该方向得到一个标量分数，
    再按 keep_ratio 取分位数作阈值，只保留分数在阈值以上的点。
    这样每次都从一个随机方向切出点云的一侧，制造视角受限的局部视图。
    """
    normal = rng.normal(size=3)          # 随机方向
    normal /= np.linalg.norm(normal)     # 归一化成单位向量
    scores = points @ normal             # 每个点在该方向上的投影分数
    # 取分位数：保留比例为 keep_ratio，所以阈值取 (1 - keep_ratio) 分位点
    threshold = np.quantile(scores, 1.0 - keep_ratio)
    return points[scores >= threshold]   # 只保留平面一侧的点


def make_batch(source: np.ndarray, target: np.ndarray, transform: np.ndarray, cfg, rng: np.random.Generator):
    """把一对点云 + 真值变换加工成 GeoTransformer 前向所需的一个 batch。

    这里同时完成数据增强（随机平面裁剪、随机下采样）与坐标归一化，
    并相应地把真值变换换算到归一化坐标系下，最后交给上游的
    stack-mode collate 函数打包成多层级 batch。
    """
    from geotransformer.utils.data import registration_collate_fn_stack_mode

    # 随机保留 70%~85% 的点，制造程度不一的低重叠场景
    keep_ratio = float(rng.uniform(0.70, 0.85))
    source = crop_with_plane(source, keep_ratio, rng)   # 源、目标各自独立裁剪
    target = crop_with_plane(target, keep_ratio, rng)
    # 把两片点云平移到公共中心并按同一 scale 缩放；返回所用的 center 和 scale
    source, target, center, scale = _normalize_pair(source, target)
    # 各自随机采样到固定 717 个点（配合上游网络的点数预算），种子来自 rng 保证可复现
    source = _sample_points(source, 717, int(rng.integers(2**31 - 1)))
    target = _sample_points(target, 717, int(rng.integers(2**31 - 1)))
    # 把真值变换搬到归一化坐标系：旋转部分不变，平移部分需按 center/scale 重新推导
    transform_n = np.eye(4, dtype=np.float32)
    transform_n[:3, :3] = transform[:3, :3]
    transform_n[:3, 3] = (transform[:3, :3] @ center + transform[:3, 3] - center) / scale
    # 组装成上游数据格式：ref=目标、src=源，特征这里用全 1 占位（几何网络主要靠坐标）
    sample = {
        "ref_points": target,
        "src_points": source,
        "ref_feats": np.ones((len(target), 1), dtype=np.float32),
        "src_feats": np.ones((len(source), 1), dtype=np.float32),
        "transform": transform_n,
    }
    # 用上游的 collate 函数按 backbone 的多层级（num_stages）体素/半径参数打包成 batch
    return registration_collate_fn_stack_mode(
        [sample], cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius, [64] * cfg.backbone.num_stages
    )


def main() -> None:
    """在兔子数据上对 GeoTransformer 做小规模、无数据泄漏的域自适应微调。

    关键点：holdout 指定的两帧（默认待评测的 bun000/bun180）绝不进入训练，
    每个训练样本都是“其它帧”里随机抽取的一对，配以 bun.conf 的真值变换。
    """
    # 解析命令行参数：数据目录、留出帧、训练步数、学习率、随机种子、
    # 初始权重路径、微调后权重的输出路径。
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "bunny" / "data")
    parser.add_argument("--holdout", nargs=2, default=["bun000", "bun180"])  # 留出（不参与训练）的评测帧
    parser.add_argument("--steps", type=int, default=12)                     # 训练步数（样本对数量）
    parser.add_argument("--lr", type=float, default=1e-5)                    # 学习率，取很小以避免破坏预训练权重
    parser.add_argument("--seed", type=int, default=42)                      # 随机种子，保证可复现
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)  # 微调起点权重
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints" / "geotransformer-bunny-finetuned.pth.tar")
    args = parser.parse_args()

    # 打开上游 GeoTransformer 的 CPU 兼容补丁（无 GPU 时也能跑）
    _enable_upstream_cpu_compat(torch)
    # 把上游实验代码目录加入搜索路径，才能按名字 import 到 config/model/loss 模块
    for path in (str(EXPERIMENT_ROOT), str(UPSTREAM_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
    # 动态导入上游模块（它们依赖上面插入的 sys.path）
    config = importlib.import_module("config")
    model_module = importlib.import_module("model")
    loss_module = importlib.import_module("loss")
    cfg = config.make_cfg()  # 构造上游配置对象
    # 加载预训练权重快照到 CPU
    snapshot = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = model_module.create_model(cfg)
    model.load_state_dict(snapshot["model"], strict=True)  # 严格加载，确保结构完全匹配
    model.train()                              # 切到训练模式（启用 dropout/BN 更新等）
    loss_fn = loss_module.OverallLoss(cfg)     # 上游的总损失（含粗/精两部分）
    # AdamW 优化器，配小学习率 + 轻微权重衰减，做保守微调
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)

    # 读取真值位姿，并排除 holdout 帧，剩下的帧才用于生成训练对
    poses = parse_bun_conf(args.data_dir / "bun.conf")
    names = sorted(name for name in poses if name not in set(args.holdout))
    # 预先把参与训练的点云全部读入内存，避免每步重复读盘
    clouds = {name: read_points(args.data_dir / f"{name}.ply") for name in names}
    rng = np.random.default_rng(args.seed)  # 独立随机数发生器，控制采样与增强
    records = []
    for step in range(1, args.steps + 1):
        # 每步随机取两帧不重复的点云作为一对训练样本
        source_name, target_name = rng.choice(names, size=2, replace=False)
        transform = relative_transform(poses[source_name], poses[target_name])  # 该对的真值变换
        batch = make_batch(clouds[source_name], clouds[target_name], transform, cfg, rng)
        # 标准训练一步：清梯度 → 前向 → 算损失 → 反传 → 梯度裁剪 → 更新
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        losses = loss_fn(outputs, batch)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 裁剪梯度范数，稳定训练
        optimizer.step()
        # 记录该步的总损失与粗/精两部分损失，便于观察收敛
        record = {"step": step, "pair": f"{source_name}->{target_name}", "loss": float(losses["loss"].detach()),
                  "coarse_loss": float(losses["c_loss"].detach()), "fine_loss": float(losses["f_loss"].detach())}
        records.append(record)
        print(json.dumps(record))  # 逐步打印，方便实时监控

    # 保存微调后的权重及元信息（步数、留出帧、每步损失记录）
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "steps": args.steps, "holdout": args.holdout, "records": records}, args.output)
    print(json.dumps({"output": str(args.output), "holdout": args.holdout, "steps": args.steps}))


# 仅在直接运行脚本时启动微调
if __name__ == "__main__":
    main()
