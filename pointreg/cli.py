"""命令行入口：提供 pair（单组配准）、batch（对比实验）、evaluate（全点对评测）三个子命令。

用 `python -m pointreg.cli <子命令> ...` 调用，是不依赖 Web UI 的批处理入口。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cloudcompare import export_cloudcompare, launch_cloudcompare
from .dataset import register_dataset_pair
from .experiments import run_final_evaluation, run_full_suite, run_method_comparison
from .io import parse_bun_conf, read_points
from .models import RegistrationConfig
from .pipeline import register_pair
from .transforms import relative_transform


def parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器，定义三个子命令及各自选项。"""
    root = argparse.ArgumentParser(description="部分重合点云配准课程设计")
    commands = root.add_subparsers(dest="command", required=True)
    # 子命令 pair：配准指定的两个点云文件
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
    # 子命令 batch：多方法对比实验（加 --full 跑完整实验套件）
    batch = commands.add_parser("batch", help="运行算法对比实验")
    batch.add_argument("--data-dir", type=Path, default=Path("bunny/data"))
    batch.add_argument("--output", type=Path, default=Path("outputs/experiments"))
    batch.add_argument("--full", action="store_true", help="运行扰动、重叠、体素与速度完整实验")
    # 子命令 evaluate：对全部点对做最终评测并导出 CSV
    evaluate = commands.add_parser("evaluate", help="运行全点对最终评测")
    evaluate.add_argument("--data-dir", type=Path, default=Path("bunny/data"))
    evaluate.add_argument("--output", type=Path, default=Path("outputs/final_evaluation"))
    evaluate.add_argument("--bridge-overlap", type=float, default=.50,
                          help="仅对低于该真值重合率的点对运行桥接法")
    return root


def main(argv: list[str] | None = None) -> int:
    """解析命令行并分派到对应子命令；返回进程退出码（0 成功、2 配准失败）。"""
    args = parser().parse_args(argv)
    if args.command == "evaluate":
        frame = run_final_evaluation(
            args.data_dir, args.output, bridge_overlap_threshold=args.bridge_overlap
        )
        print(frame.groupby("方法").size().to_string())
        print(f"CSV: {args.output / 'final_evaluation.csv'}")
        return 0
    if args.command == "batch":
        if args.full:
            frames = run_full_suite(args.data_dir, args.output)
            print("\n".join(f"{name}: {len(frame)} rows" for name, frame in frames.items()))
        else:
            frame = run_method_comparison(args.data_dir, args.output)
            print(frame.to_string(index=False))
        return 0
    # —— 默认分支：pair 单组配准 ——
    config = RegistrationConfig(coarse_method=args.coarse, fine_method=args.fine, voxel_size=args.voxel,
                                max_correspondence_distance=args.distance, trim_fraction=args.trim, max_iterations=args.iterations)
    ground_truth = None
    if args.conf:
        # 提供 bun.conf 时解析真值位姿以便计算误差
        poses = parse_bun_conf(args.conf)
        ground_truth = relative_transform(poses[args.source.stem], poses[args.target.stem])
    # 当两文件同目录、含 bun.conf 且用 FPFH 时，走桥接法（对低重合更稳）；否则直接两帧配准
    if args.coarse == "fpfh" and args.source.parent.resolve() == args.target.parent.resolve() and (args.source.parent / "bun.conf").exists():
        result = register_dataset_pair(args.source.parent, args.source.stem, args.target.stem, config)
    else:
        result = register_pair(args.source, args.target, config, ground_truth=ground_truth)
    # 导出 CloudCompare 可视化文件与 JSON 结果
    source, target = read_points(args.source), read_points(args.target)
    files = export_cloudcompare(args.output, source, target, result.transformation, result.to_dict())
    (args.output / "result.json").write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    if args.open_cloudcompare:
        print(launch_cloudcompare([files["target"], files["aligned"]])[1])
    return 0 if result.status != "failed" else 2


if __name__ == "__main__":
    # 以模块方式运行时，把 main 的返回值作为进程退出码
    raise SystemExit(main())
