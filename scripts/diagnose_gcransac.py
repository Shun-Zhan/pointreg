"""Sweep FPFH/GC-RANSAC correspondence settings on direct Bunny pairs.

GC-RANSAC 诊断脚本：对若干兔子点对做网格搜索（grid search），
在不同的体素尺寸、Lowe 比值、内点距离阈值组合下，观察
FPFH 特征匹配 + GC-RANSAC 求刚体变换的对应数量、内点数量与误差，
用来诊断为什么 GC-RANSAC 在低重叠场景下表现不稳定、以及哪组参数更优。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree   # KD 树，用于在特征空间做最近邻检索
import pygcransac                    # GC-RANSAC 的 Python 绑定

# 定位项目根目录并加入模块搜索路径，才能 import pointreg 包
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pointreg.coarse import _fpfh_cloud_and_features  # 下采样并计算 FPFH 特征
from pointreg.io import parse_bun_conf, read_points   # 读位姿与点云
from pointreg.metrics import pose_errors              # 计算旋转/平移误差
from pointreg.transforms import relative_transform    # 由绝对位姿求相对真值变换


# 待诊断的点对（低重叠困难样例）
PAIRS = [("bun000", "bun180"), ("chin", "top2"), ("bun000", "ear_back")]
# 网格搜索的三组超参：
SCALES = (0.0018, 0.0025, 0.004, 0.006)  # 体素下采样尺寸（越小点越密）
RATIOS = (None, 0.9, 0.8)                # Lowe 比值检验阈值，None 表示不做比值筛选
THRESHOLDS = (0.005, 0.01, 0.015)        # GC-RANSAC 判定内点的距离阈值


def solve(source, target, voxel, ratio, threshold):
    """对单个点对、单组参数运行一次 FPFH+GC-RANSAC，返回 (变换, 对应数, 内点数)。

    流程：
    1. 分别对源/目标点云下采样并算 FPFH 特征；
    2. 在特征空间做双向最近邻 + 互检（mutual check）建立对应；
    3. 可选地用 Lowe 比值检验剔除模糊匹配；
    4. 把对应喂给 GC-RANSAC 估计刚体变换。
    对应少于 3 对时无法解算，直接返回 None。
    """
    # 下采样并计算两片点云的 FPFH 特征（cloud 为下采样后的点云，features 为特征矩阵）
    source_cloud, source_features = _fpfh_cloud_and_features(source, voxel)
    target_cloud, target_features = _fpfh_cloud_and_features(target, voxel)
    source_points, target_points = np.asarray(source_cloud.points), np.asarray(target_cloud.points)
    # 对每个源特征在目标特征里找最近的两个邻居（k=2 用于后面 Lowe 比值检验）
    target_distances, target_indices = cKDTree(target_features).query(source_features, k=2)
    # 反向：对每个目标特征找最近的源特征，用于互检（mutual nearest neighbor）
    source_indices = cKDTree(source_features).query(target_features)[1]
    indices = np.arange(len(source_points))
    # 互检掩码：源 i 的最近目标点，其反向最近邻恰好又指回 i，才算稳定对应
    masks = source_indices[target_indices[:, 0]] == indices
    if ratio is not None:
        # Lowe 比值检验：最近距离 / 次近距离 < ratio，过滤掉区分度低的模糊匹配
        masks &= target_distances[:, 0] / np.maximum(target_distances[:, 1], 1e-12) < ratio
    indices = indices[masks]
    if len(indices) < 3:
        # 少于 3 对对应无法估计刚体变换，直接返回失败
        return None, len(indices), 0
    # 拼成 GC-RANSAC 需要的对应矩阵：每行 = [源点xyz, 目标点xyz]
    correspondences = np.c_[source_points[indices], target_points[target_indices[indices, 0]]].astype(np.float64)
    # 用特征距离折算每对匹配的可信概率（距离越小概率越高），作为 GC-RANSAC 的先验权重
    probabilities = np.exp(-target_distances[indices, 0] / max(float(np.median(target_distances[indices, 0])), 1e-6))
    # 运行 GC-RANSAC 求刚体变换：conf 置信度、max_iters 最大迭代、
    # use_space_partitioning 开启空间划分加速
    model, inliers = pygcransac.findRigidTransform(
        correspondences, probabilities.astype(np.float64), threshold=threshold, conf=0.999,
        max_iters=20000, neighborhood=0, use_space_partitioning=True,
    )
    if model is None:
        return None, len(indices), 0  # 求解失败
    # pygcransac 返回的是转置形式，这里 .T 转回标准 4x4 变换矩阵
    return np.asarray(model).T, len(indices), int(inliers.sum())


def main():
    """三重循环遍历 (点对 × 体素 × 比值 × 阈值)，逐组解算并把诊断结果写成 JSON。"""
    data_dir = ROOT / "bunny" / "data"
    poses = parse_bun_conf(data_dir / "bun.conf")  # 读取真值位姿
    rows = []
    for source_name, target_name in PAIRS:
        # 读入原始点云；每个点对只读一次，后面复用于各参数组合
        source, target = read_points(data_dir / f"{source_name}.ply"), read_points(data_dir / f"{target_name}.ply")
        ground_truth = relative_transform(poses[source_name], poses[target_name])
        # 对每个点对做参数网格搜索
        for voxel in SCALES:
            for ratio in RATIOS:
                for threshold in THRESHOLDS:
                    transform, matches, inliers = solve(source, target, voxel, ratio, threshold)
                    # 记录本组参数下的对应数与内点数
                    row = {"pair": f"{source_name}->{target_name}", "voxel": voxel, "ratio": ratio,
                           "threshold": threshold, "matches": matches, "inliers": inliers}
                    if transform is not None:
                        # 解算成功时，额外计算与真值之间的旋转/平移误差
                        rre, rte = pose_errors(transform, ground_truth)
                        row.update(rotation_error_deg=rre, translation_error=rte)
                    rows.append(row)
    # 把全部诊断结果写到 outputs/gcransac_diagnostic.json
    output = ROOT / "outputs" / "gcransac_diagnostic.json"
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(output)


# 仅在直接运行脚本时执行诊断
if __name__ == "__main__":
    main()
