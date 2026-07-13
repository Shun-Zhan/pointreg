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
    normal = rng.normal(size=3)
    normal /= np.linalg.norm(normal)
    scores = points @ normal
    threshold = np.quantile(scores, 1.0 - keep_ratio)
    return points[scores >= threshold]


def make_batch(source: np.ndarray, target: np.ndarray, transform: np.ndarray, cfg, rng: np.random.Generator):
    from geotransformer.utils.data import registration_collate_fn_stack_mode

    keep_ratio = float(rng.uniform(0.70, 0.85))
    source = crop_with_plane(source, keep_ratio, rng)
    target = crop_with_plane(target, keep_ratio, rng)
    source, target, center, scale = _normalize_pair(source, target)
    source = _sample_points(source, 717, int(rng.integers(2**31 - 1)))
    target = _sample_points(target, 717, int(rng.integers(2**31 - 1)))
    transform_n = np.eye(4, dtype=np.float32)
    transform_n[:3, :3] = transform[:3, :3]
    transform_n[:3, 3] = (transform[:3, :3] @ center + transform[:3, 3] - center) / scale
    sample = {
        "ref_points": target,
        "src_points": source,
        "ref_feats": np.ones((len(target), 1), dtype=np.float32),
        "src_feats": np.ones((len(source), 1), dtype=np.float32),
        "transform": transform_n,
    }
    return registration_collate_fn_stack_mode(
        [sample], cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius, [64] * cfg.backbone.num_stages
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "bunny" / "data")
    parser.add_argument("--holdout", nargs=2, default=["bun000", "bun180"])
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints" / "geotransformer-bunny-finetuned.pth.tar")
    args = parser.parse_args()

    _enable_upstream_cpu_compat(torch)
    for path in (str(EXPERIMENT_ROOT), str(UPSTREAM_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
    config = importlib.import_module("config")
    model_module = importlib.import_module("model")
    loss_module = importlib.import_module("loss")
    cfg = config.make_cfg()
    snapshot = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = model_module.create_model(cfg)
    model.load_state_dict(snapshot["model"], strict=True)
    model.train()
    loss_fn = loss_module.OverallLoss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)

    poses = parse_bun_conf(args.data_dir / "bun.conf")
    names = sorted(name for name in poses if name not in set(args.holdout))
    clouds = {name: read_points(args.data_dir / f"{name}.ply") for name in names}
    rng = np.random.default_rng(args.seed)
    records = []
    for step in range(1, args.steps + 1):
        source_name, target_name = rng.choice(names, size=2, replace=False)
        transform = relative_transform(poses[source_name], poses[target_name])
        batch = make_batch(clouds[source_name], clouds[target_name], transform, cfg, rng)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        losses = loss_fn(outputs, batch)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        record = {"step": step, "pair": f"{source_name}->{target_name}", "loss": float(losses["loss"].detach()),
                  "coarse_loss": float(losses["c_loss"].detach()), "fine_loss": float(losses["f_loss"].detach())}
        records.append(record)
        print(json.dumps(record))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "steps": args.steps, "holdout": args.holdout, "records": records}, args.output)
    print(json.dumps({"output": str(args.output), "holdout": args.holdout, "steps": args.steps}))


if __name__ == "__main__":
    main()
