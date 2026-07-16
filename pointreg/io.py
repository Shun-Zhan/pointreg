"""点云与真值位姿的输入输出：读取 PLY、解析 bun.conf、写出带色 PLY。

read_points 优先用 Open3D 读取（支持各种格式与二进制 PLY），并自带一个
纯 Python 的 ASCII PLY 解析回退，保证没装 Open3D 时也能读 Bunny 数据。
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from .transforms import make_transform, quaternion_xyzw_to_matrix


def parse_bun_conf(path: str | Path) -> dict[str, np.ndarray]:
    """解析 Stanford Bunny 的 bun.conf，返回 {扫描名: 4×4 世界位姿} 字典。

    文件中每行 `bmesh <file> tx ty tz qx qy qz qw` 描述一帧的平移+四元数。
    这些真值位姿仅用于评估配准误差，不参与配准求解本身。
    """
    poses: dict[str, np.ndarray] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        # 只取合法的 bmesh 行（关键字 + 文件名 + 3 平移 + 4 四元数 = 9 字段）
        if not fields or fields[0] != "bmesh" or len(fields) != 9:
            continue
        name = Path(fields[1]).stem
        values = np.asarray([float(v) for v in fields[2:]], dtype=float)
        # Stanford 早期 ZipPack/Vrip 约定使用的旋转是常规 xyzw 公式所得
        # “主动列向量旋转”的转置，故这里对旋转矩阵取 .T 以对齐其坐标约定。
        poses[name] = make_transform(quaternion_xyzw_to_matrix(values[3:]).T, values[:3])
    if not poses:
        raise ValueError(f"no bmesh poses found in {path}")
    return poses


def read_points(path: str | Path) -> np.ndarray:
    """读取点云文件为 (N, 3) 坐标数组，优先 Open3D，失败则回退纯 Python 解析。"""
    path = Path(path)
    try:
        # 首选 Open3D：支持二进制/ASCII PLY 及其他格式
        import open3d as o3d
        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points, dtype=float)
        if len(points):
            return points
    except ImportError:
        pass
    # 回退路径只支持 ASCII 编码的 PLY
    if path.suffix.lower() != ".ply":
        raise RuntimeError("Open3D is required for non-PLY files")
    with path.open("r", encoding="ascii", errors="strict") as handle:
        # —— 解析 PLY 文件头，确定顶点数与各属性列的顺序 ——
        vertex_count = None
        properties: list[str] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("invalid PLY header")
            fields = line.strip().split()
            if fields[:2] == ["format", "binary_little_endian"]:
                # 纯 Python 回退不解析二进制 PLY，明确提示需要 Open3D
                raise RuntimeError("Open3D is required for binary PLY")
            if fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])                 # 顶点数量
            elif fields[:1] == ["property"] and vertex_count is not None:
                properties.append(fields[-1])                 # 按声明顺序记录各属性名
            elif fields[:1] == ["end_header"]:
                break
        if vertex_count is None or not {"x", "y", "z"}.issubset(properties):
            raise ValueError("PLY has no valid vertex coordinates")
        # 定位 x/y/z 三列在每行中的位置（文件可能还含 nx/ny/nz、颜色等其他列）
        indices = [properties.index(axis) for axis in "xyz"]
        rows = []
        for _ in range(vertex_count):
            values = handle.readline().split()
            rows.append([float(values[i]) for i in indices])  # 只取坐标三列
    return np.asarray(rows, dtype=float)


def write_ply(path: str | Path, points: np.ndarray, color: tuple[int, int, int] | None = None) -> Path:
    """把点云写成 ASCII PLY 文件，可选统一着色（用于 CloudCompare 可视化对比）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=float)
    # 若指定颜色，则在头部追加 RGB 属性声明
    color_header = "property uchar red\nproperty uchar green\nproperty uchar blue\n" if color else ""
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n{color_header}end_header\n")
        suffix = f" {color[0]} {color[1]} {color[2]}" if color else ""
        for point in points:
            # %.9g 在保留足够精度的同时避免多余零位
            handle.write(f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g}{suffix}\n")
    return path


def safe_name(value: str) -> str:
    """把任意字符串规整为可安全用作文件名的形式（非字母数字等替换为下划线）。"""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
