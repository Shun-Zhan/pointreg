"""Compare direct low-overlap registration coarse initializers on Bunny.

低重叠基准测试脚本：在斯坦福兔子（Bunny）的几组“困难”点对上，
分别用不同的粗配准初始化方法跑完整配准流程，并把成功率、误差、
耗时等指标汇总成一个 JSON 文件，方便对比各方法在低重叠场景下的表现。
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

# 定位到项目根目录（本文件在 scripts/ 下，parents[1] 即上一级），
# 并把它加入模块搜索路径，这样才能 import 到 pointreg 包。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pointreg.io import parse_bun_conf          # 解析 bun.conf 得到每帧扫描的真值位姿
from pointreg.models import RegistrationConfig   # 配准参数配置对象
from pointreg.pipeline import register_pair      # 完整“粗配准 + 精配准”流水线入口
from pointreg.transforms import relative_transform  # 由两帧绝对位姿计算相对变换（真值）


# 待测试的点对：都是刻意挑选的低重叠/大视角差组合（如 bun000 与 bun180 几乎背对背）
PAIRS = [("bun000", "bun180"), ("chin", "top2"), ("bun000", "ear_back")]
# 三种参与对比的粗配准方法：单尺度 FPFH、多尺度 FPFH、以及基于 GC-RANSAC 的 FPFH
METHODS = ("fpfh", "fpfh_multiscale", "gcransac")


def main() -> None:
    """遍历所有点对与方法，逐一配准并收集指标，最后写出对比结果 JSON。"""
    data_dir = ROOT / "bunny" / "data"
    # 读入所有帧的真值位姿（相机/扫描仪坐标系到公共坐标系的刚体变换）
    poses = parse_bun_conf(data_dir / "bun.conf")
    rows = []  # 累积每一次实验的一行结果
    for source_name, target_name in PAIRS:
        # 由两帧的绝对位姿推出“源到目标”的相对刚体变换，作为评测用的真值
        ground_truth = relative_transform(poses[source_name], poses[target_name])
        for method in METHODS:
            # 用当前粗配准方法 + 自研 ICP 精配准跑一次完整流水线；
            # 精配准统一用 custom_icp，从而只比较粗配准初始化的差异。
            result = register_pair(
                data_dir / f"{source_name}.ply",
                data_dir / f"{target_name}.ply",
                RegistrationConfig(coarse_method=method, fine_method="custom_icp", voxel_size=0.0025, max_correspondence_distance=0.01),
                ground_truth=ground_truth,
            )
            # 记录关键指标：是否成功、状态、旋转/平移误差、拟合度以及耗时
            rows.append(
                {
                    "pair": f"{source_name}->{target_name}",
                    "method": method,
                    "success": result.success,
                    "status": result.status,
                    "rotation_error_deg": result.metrics.get("rotation_error_deg"),
                    "translation_error_ratio": result.metrics.get("translation_error_ratio"),
                    "fitness": result.metrics.get("fitness"),
                    "coarse_ms": result.timings_ms.get("coarse"),
                    "total_ms": result.timings_ms.get("total"),
                }
            )
    # 把汇总结果写到 outputs/ 目录，ensure_ascii=False 保证中文/符号原样输出
    output = ROOT / "outputs" / "low_overlap_coarse_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)  # 打印结果文件路径，方便后续查看


# 作为脚本直接运行时才执行 main()，被 import 时不会触发
if __name__ == "__main__":
    main()
