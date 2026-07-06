from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cloudcompare import export_cloudcompare, launch_cloudcompare
from .experiments import run_full_suite, run_method_comparison
from .io import parse_bun_conf, read_points
from .models import RegistrationConfig
from .pipeline import register_pair
from .dataset import register_dataset_pair
from .transforms import relative_transform


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="部分重合点云配准课程设计")
    commands = root.add_subparsers(dest="command", required=True)
    pair = commands.add_parser("pair", help="运行单组配准")
    pair.add_argument("source", type=Path)
    pair.add_argument("target", type=Path)
    pair.add_argument("--conf", type=Path)
    pair.add_argument("--coarse", choices=["none", "pca", "fpfh"], default="fpfh")
    pair.add_argument("--fine", choices=["custom_icp", "point_to_plane"], default="custom_icp")
    pair.add_argument("--voxel", type=float, default=.0025)
    pair.add_argument("--distance", type=float, default=.01)
    pair.add_argument("--trim", type=float, default=.8)
    pair.add_argument("--iterations", type=int, default=60)
    pair.add_argument("--output", type=Path, default=Path("outputs/latest"))
    pair.add_argument("--open-cloudcompare", action="store_true")
    batch = commands.add_parser("batch", help="运行算法对比实验")
    batch.add_argument("--data-dir", type=Path, default=Path("bunny/data"))
    batch.add_argument("--output", type=Path, default=Path("outputs/experiments"))
    batch.add_argument("--full", action="store_true", help="运行扰动、重叠、体素与速度完整实验")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "batch":
        if args.full:
            frames = run_full_suite(args.data_dir, args.output)
            print("\n".join(f"{name}: {len(frame)} rows" for name, frame in frames.items()))
        else:
            frame = run_method_comparison(args.data_dir, args.output)
            print(frame.to_string(index=False))
        return 0
    config = RegistrationConfig(coarse_method=args.coarse, fine_method=args.fine, voxel_size=args.voxel,
                                max_correspondence_distance=args.distance, trim_fraction=args.trim, max_iterations=args.iterations)
    ground_truth = None
    if args.conf:
        poses = parse_bun_conf(args.conf)
        ground_truth = relative_transform(poses[args.source.stem], poses[args.target.stem])
    if args.coarse == "fpfh" and args.source.parent.resolve() == args.target.parent.resolve() and (args.source.parent / "bun.conf").exists():
        result = register_dataset_pair(args.source.parent, args.source.stem, args.target.stem, config)
    else:
        result = register_pair(args.source, args.target, config, ground_truth=ground_truth)
    source, target = read_points(args.source), read_points(args.target)
    files = export_cloudcompare(args.output, source, target, result.transformation, result.to_dict())
    (args.output / "result.json").write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    if args.open_cloudcompare:
        print(launch_cloudcompare([files["target"], files["aligned"]])[1])
    return 0 if result.status != "failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
