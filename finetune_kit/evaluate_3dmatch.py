"""用 3DMatch 结构的 GeoTransformer 权重评测 Bunny 低重合对(粗配 + ICP)。

可评测原版 3DMatch 权重,也可评测微调后的权重,便于前后对比:
    python finetune_kit/evaluate_3dmatch.py --checkpoint checkpoints/geotransformer-3dmatch.pth.tar
    python finetune_kit/evaluate_3dmatch.py --checkpoint checkpoints/geotransformer-bunny-3dmatch-ft.pth.tar
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

from pointreg.io import read_points, parse_bun_conf
from pointreg.preprocessing import preprocess_points, bounding_box_diagonal
from pointreg.transforms import relative_transform
from pointreg.metrics import symmetric_overlap, pose_errors, alignment_metrics
from pointreg.icp import custom_icp
from pointreg.models import RegistrationConfig

UPSTREAM = ROOT / "third_party" / "GeoTransformer-main"
EXP3D = UPSTREAM / "experiments" / "geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn"
NEIGHBOR_LIMITS = [38, 36, 36, 38]

DEFAULT_PAIRS = [("bun000", "bun045"), ("bun000", "bun180"),
                 ("chin", "top2"), ("bun000", "ear_back")]


def load_model(checkpoint, device):
    for path in (str(EXP3D), str(UPSTREAM)):
        if path not in sys.path:
            sys.path.insert(0, path)
    if not torch.cuda.is_available():
        torch.Tensor.cuda = lambda self, *a, **k: self  # CPU 兜底
    config = importlib.import_module("config")
    model_module = importlib.import_module("model")
    cfg = config.make_cfg()
    model = model_module.create_model(cfg).to(device)
    snap = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(snap["model"], strict=True)
    model.eval()
    return cfg, model


def to_device(v, device):
    if isinstance(v, list):
        return [to_device(x, device) for x in v]
    if isinstance(v, dict):
        return {k: to_device(x, device) for k, x in v.items()}
    return v.to(device) if torch.is_tensor(v) else v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "bunny" / "data")
    ap.add_argument("--voxel", type=float, default=0.0025)
    ap.add_argument("--distance", type=float, default=0.01)
    ap.add_argument("--output", type=Path, default=ROOT / "outputs" / "eval_3dmatch_ft.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, model = load_model(str(args.checkpoint), device)
    from geotransformer.utils.data import registration_collate_fn_stack_mode
    scale = cfg.backbone.init_voxel_size / args.voxel
    poses = parse_bun_conf(args.data_dir / "bun.conf")

    rows = []
    for a, b in DEFAULT_PAIRS:
        gt = relative_transform(poses[a], poses[b])
        sp = preprocess_points(read_points(args.data_dir / f"{a}.ply"), args.voxel)
        tp = preprocess_points(read_points(args.data_dir / f"{b}.ply"), args.voxel)
        ov = symmetric_overlap(sp, tp, gt, args.distance)
        diag = bounding_box_diagonal(sp, tp)
        sample = {"ref_points": (tp * scale).astype(np.float32),
                  "src_points": (sp * scale).astype(np.float32),
                  "ref_feats": np.ones((len(tp), 1), np.float32),
                  "src_feats": np.ones((len(sp), 1), np.float32),
                  "transform": np.eye(4, dtype=np.float32)}
        dd = to_device(registration_collate_fn_stack_mode(
            [sample], cfg.backbone.num_stages, cfg.backbone.init_voxel_size,
            cfg.backbone.init_radius, NEIGHBOR_LIMITS), device)
        with torch.no_grad():
            est_s = model(dd)["estimated_transform"].detach().cpu().numpy()
        est = np.eye(4)
        est[:3, :3] = est_s[:3, :3]
        est[:3, 3] = est_s[:3, 3] / scale
        rc, _ = pose_errors(est, gt)
        t2, _, _, _ = custom_icp(sp, tp, est,
                                 RegistrationConfig(voxel_size=args.voxel,
                                                    max_correspondence_distance=args.distance))
        r2, te2 = pose_errors(t2, gt)
        fit = alignment_metrics(sp, tp, t2, args.distance)["fitness"]
        row = dict(pair=f"{a}->{b}", overlap=round(float(ov), 3),
                   coarse_rot=round(float(rc), 2), final_rot=round(float(r2), 2),
                   final_tr_ratio=round(float(te2 / diag), 4), fitness=round(float(fit), 3),
                   success=bool(r2 < 5.0 and te2 / diag < 0.02))
        rows.append(row)
        print(f"{row['pair']:>18} ov={row['overlap']:.3f} "
              f"coarse_rot={row['coarse_rot']:7.2f} final_rot={row['final_rot']:7.2f} "
              f"tr={row['final_tr_ratio']:.4f} fit={row['fitness']:.3f} "
              f"{'OK' if row['success'] else 'FAIL'}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(args.output, "w"), ensure_ascii=False, indent=2)
    print("saved", args.output)


if __name__ == "__main__":
    main()
