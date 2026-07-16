"""命令行入口：提供 pair（单组配准）与 batch（批量对比实验）两个子命令。

pair 用于配准指定的两帧并导出可视化产物；batch 用于跑一系列实验（全组合、方法对比、
完整实验套件），生成 CSV/图表。是整个项目对外的操作界面。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cloudcompare import export_cloudcompare, launch_cloudcompare
from .experiments import run_all_pairs, run_full_suite, run_method_comparison
from .io import parse_bun_conf, read_points
from .models import RegistrationConfig
from .pipeline import register_pair
from .transforms import relative_transform


def parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器，定义 pair 与 batch 两个子命令及其各自参数。"""
    root = argparse.ArgumentParser(description="部分重合点云配准课程设计")
    commands = root.add_subparsers(dest="command", required=True)  # 子命令必选。
    # 子命令 pair：单组两帧配准，参数覆盖源/目标、粗配/精配方法、体素、迭代等超参。
    pair = commands.add_parser("pair", help="运行单组配准")
    pair.add_argument("source", type=Path)
    pair.add_argument("target", type=Path)
    pair.add_argument("--conf", type=Path)
    pair.add_argument("--coarse", choices=["none", "pca", "fpfh", "fpfh_multiscale", "gcransac", "geotransformer"], default="fpfh")
    pair.add_argument("--fine", choices=["custom_icp", "point_to_plane"], default="custom_icp")
    pair.add_argument("--voxel", type=float, default=.0025)
    pair.add_argument("--distance", type=float, default=.01)
    pair.add_argument("--trim", type=float, default=.8)
    pair.add_argument("--iterations", type=int, default=60)
    pair.add_argument("--geotransformer-checkpoint", type=Path, help="ModelNet checkpoint path; defaults to checkpoints/geotransformer-modelnet.pth.tar")
    pair.add_argument("--output", type=Path, default=Path("outputs/latest"))
    pair.add_argument("--open-cloudcompare", action="store_true")
    # 子命令 batch：批量实验，通过互斥的 --all-pairs / --full 标志选择实验类型。
    batch = commands.add_parser("batch", help="运行算法对比实验")
    batch.add_argument("--data-dir", type=Path, default=Path("bunny/data"))
    batch.add_argument("--output", type=Path, default=Path("outputs/experiments"))
    batch.add_argument("--all-pairs", action="store_true", help="遍历 bun.conf 中所有有序两帧组合")
    batch.add_argument("--full", action="store_true", help="运行扰动、重叠、体素与速度完整实验")
    return root


def main(argv: list[str] | None = None) -> int:
    """程序主入口：解析参数、分派到对应子命令，返回进程退出码。

    返回值约定：0 表示成功，2 表示配准失败（供脚本/CI 判断）。
    """
    args = parser().parse_args(argv)
    if args.command == "batch":
        # batch 分支：按标志选择跑哪套实验，并把结果概要打印到终端。
        if args.all_pairs:
            frame = run_all_pairs(args.data_dir, args.output)
            print(frame.to_string(index=False))
        elif args.full:
            frames = run_full_suite(args.data_dir, args.output)
            print("\n".join(f"{name}: {len(frame)} rows" for name, frame in frames.items()))
        else:
            frame = run_method_comparison(args.data_dir, args.output)
            print(frame.to_string(index=False))
        return 0
    # pair 分支：把命令行超参组装成 RegistrationConfig。
    config = RegistrationConfig(coarse_method=args.coarse, fine_method=args.fine, voxel_size=args.voxel,
                                max_correspondence_distance=args.distance, trim_fraction=args.trim, max_iterations=args.iterations,
                                geotransformer_checkpoint=str(args.geotransformer_checkpoint) if args.geotransformer_checkpoint else None)
    # 若提供了 --conf 真值文件，则据此算出两帧的相对真值位姿以便评估误差。
    ground_truth = None
    if args.conf:
        poses = parse_bun_conf(args.conf)
        ground_truth = relative_transform(poses[args.source.stem], poses[args.target.stem])
    result = register_pair(args.source, args.target, config, ground_truth=ground_truth)
    # 导出可视化产物，并把完整结果写入 result.json、同时打印到终端。
    source, target = read_points(args.source), read_points(args.target)
    files = export_cloudcompare(args.output, source, target, result.transformation, result.to_dict())
    (args.output / "result.json").write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    if args.open_cloudcompare:
        # 只在用户显式要求时才自动拉起 CloudCompare（这里传目标与对齐后源两片对比）。
        print(launch_cloudcompare([files["target"], files["aligned"]])[1])
    return 0 if result.status != "failed" else 2


if __name__ == "__main__":
    # 作为脚本直接运行时，用 main 的返回值作为进程退出码。
    raise SystemExit(main())
