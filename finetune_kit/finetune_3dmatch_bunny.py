"""在 Bunny 数据上微调 GeoTransformer(3DMatch 权重)以适配低重合配准。

这是基于我们验证过“有效”的 3DMatch 场景级权重做领域自适应,而不是弱的
ModelNet 权重。核心思路:
  * 把每一对 Bunny 扫描按 bun.conf 的真值配准;
  * 用随机平面裁剪制造“部分/低重合”的训练样本;
  * 把点云放大到 3DMatch 的工作尺度(体素 2.5cm),否则几何编码尺度不匹配;
  * 用 GeoTransformer 官方的 OverallLoss 做几个 epoch 的微调。

用法(在你本机、有 N 卡的环境里):
    python finetune_kit/finetune_3dmatch_bunny.py --steps 400 --lr 1e-4

产出:checkpoints/geotransformer-bunny-3dmatch-ft.pth.tar
之后把它填进 evaluate 脚本即可对比效果。
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

from pointreg.io import parse_bun_conf, read_points
from pointreg.preprocessing import preprocess_points
from pointreg.transforms import relative_transform

UPSTREAM = ROOT / "third_party" / "GeoTransformer-main"
EXP3D = UPSTREAM / "experiments" / "geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn"
# 3DMatch 训练时用的邻居上限(官方 demo 设定)
NEIGHBOR_LIMITS = [38, 36, 36, 38]


def crop_with_plane(points: np.ndarray, keep_ratio: float, rng: np.random.Generator) -> np.ndarray:
    """用一个随机方向的平面切掉一部分点,模拟部分扫描 / 低重合。"""
    normal = rng.normal(size=3)
    normal /= np.linalg.norm(normal)
    scores = points @ normal
    threshold = np.quantile(scores, 1.0 - keep_ratio)
    return points[scores >= threshold]


def make_batch(source, target, transform, cfg, scale, rng):
    from geotransformer.utils.data import registration_collate_fn_stack_mode

    # 随机保留 55%~85%,制造从中到低的重合分布
    source = crop_with_plane(source, float(rng.uniform(0.55, 0.85)), rng)
    target = crop_with_plane(target, float(rng.uniform(0.55, 0.85)), rng)
    source = (source * scale).astype(np.float32)
    target = (target * scale).astype(np.float32)
    transform_s = transform.copy().astype(np.float32)
    transform_s[:3, 3] = transform[:3, 3] * scale  # 平移随尺度放大
    sample = {
        "ref_points": target,
        "src_points": source,
        "ref_feats": np.ones((len(target), 1), dtype=np.float32),
        "src_feats": np.ones((len(source), 1), dtype=np.float32),
        "transform": transform_s,
    }
    return registration_collate_fn_stack_mode(
        [sample], cfg.backbone.num_stages, cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius, NEIGHBOR_LIMITS,
    )


def to_device(value, device):
    if isinstance(value, list):
        return [to_device(v, device) for v in value]
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    if torch.is_tensor(value):
        return value.to(device)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "bunny" / "data")
    # 把最终要评测的低重合对留出来,绝不进入训练,避免作弊
    parser.add_argument("--holdout", nargs="*", default=["bun000", "bun180", "chin", "top2", "ear_back"])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--voxel", type=float, default=0.0025)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "checkpoints" / "geotransformer-3dmatch.pth.tar")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "checkpoints" / "geotransformer-bunny-3dmatch-ft.pth.tar")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}  (cuda_available={torch.cuda.is_available()})")
    if not args.checkpoint.is_file():
        raise SystemExit(f"[error] 找不到 3DMatch 权重: {args.checkpoint}\n"
                         f"请先下载 geotransformer-3dmatch.pth.tar 放到 checkpoints/ 下。")

    # 若无 CUDA(仅用于在 CPU 上冒烟测试),让上游硬编码的 .cuda() 退化为 no-op。
    # 在你本机有 N 卡时 cuda 可用,这段不会生效,不影响真正的 GPU 训练。
    if not torch.cuda.is_available():
        torch.Tensor.cuda = lambda self, *a, **k: self

    for path in (str(EXP3D), str(UPSTREAM)):
        if path not in sys.path:
            sys.path.insert(0, path)
    config = importlib.import_module("config")
    model_module = importlib.import_module("model")
    loss_module = importlib.import_module("loss")
    cfg = config.make_cfg()

    scale = cfg.backbone.init_voxel_size / args.voxel  # 通常 = 10
    print(f"[info] scale = {scale}  (voxel {args.voxel} -> {cfg.backbone.init_voxel_size})")

    model = model_module.create_model(cfg).to(device)
    snapshot = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(snapshot["model"], strict=True)
    model.train()

    loss_fn = loss_module.OverallLoss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)

    poses = parse_bun_conf(args.data_dir / "bun.conf")
    holdout = set(args.holdout)
    names = sorted(n for n in poses if n not in holdout)
    print(f"[info] 训练可用扫描: {names}")
    print(f"[info] 留出(不训练)的扫描: {sorted(holdout)}")
    clouds = {n: preprocess_points(read_points(args.data_dir / f"{n}.ply"), args.voxel) for n in names}

    rng = np.random.default_rng(args.seed)
    records = []
    for step in range(1, args.steps + 1):
        a, b = rng.choice(names, size=2, replace=False)
        transform = relative_transform(poses[a], poses[b])
        try:
            batch = to_device(make_batch(clouds[a], clouds[b], transform, cfg, scale, rng), device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            losses = loss_fn(out, batch)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        except RuntimeError as exc:  # 偶发的退化裁剪,跳过
            print(f"[warn] step {step} skipped: {exc}")
            continue
        rec = {"step": step, "pair": f"{a}->{b}", "loss": float(losses["loss"].detach())}
        records.append(rec)
        if step % 10 == 0 or step == 1:
            print(json.dumps(rec))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "steps": args.steps,
                "holdout": sorted(holdout), "records": records}, args.output)
    print(json.dumps({"saved": str(args.output), "steps": args.steps}))


if __name__ == "__main__":
    main()
