"""点云与配置文件的读写：解析斯坦福 bun.conf 真值位姿、读写 PLY 点云。

读点云优先用 Open3D（支持二进制等各种格式）；若没装则退回到自带的 ASCII PLY 解析器。
写点云用极简 ASCII PLY，方便 CloudCompare 等工具直接可视化并区分颜色。
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from .transforms import make_transform, quaternion_xyzw_to_matrix


def parse_bun_conf(path: str | Path) -> dict[str, np.ndarray]:
    """解析斯坦福兔子数据集的 bun.conf，返回 {扫描名: 4x4 到世界位姿}。

    文件里每行 bmesh 记录的格式为：bmesh 文件名 tx ty tz qx qy qz qw，
    即平移(3) + xyzw 四元数(4)。这些位姿是各视角相对世界坐标系的真值，
    项目仅用它来评估误差，不参与配准估计本身。
    """
    poses: dict[str, np.ndarray] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        # 只处理格式合法的 bmesh 行（关键字 + 8 个字段，共 9 项），其余行跳过。
        if not fields or fields[0] != "bmesh" or len(fields) != 9:
            continue
        name = Path(fields[1]).stem  # 用文件名（去扩展名）作为该扫描的标识。
        values = np.asarray([float(v) for v in fields[2:]], dtype=float)  # 前3平移 后4四元数。
        # 斯坦福早期 ZipPack/Vrip 约定：存储的旋转是标准 xyzw 公式所得旋转的转置，
        # 因此这里对四元数转出的旋转矩阵取 .T 才能得到正确的到世界位姿。
        poses[name] = make_transform(quaternion_xyzw_to_matrix(values[3:]).T, values[:3])
    if not poses:
        raise ValueError(f"no bmesh poses found in {path}")
    return poses


def read_points(path: str | Path) -> np.ndarray:
    """读取点云文件为 (N, 3) 的 xyz 数组。

    优先使用 Open3D 读取（兼容更多格式与二进制 PLY）；若未安装 Open3D，
    则仅支持 ASCII 编码的 PLY，并用内置解析器手动读取顶点坐标。
    """
    path = Path(path)
    try:
        import open3d as o3d
        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points, dtype=float)
        if len(points):
            return points
    except ImportError:
        pass  # 无 Open3D，转入下方内置 ASCII PLY 解析。
    if path.suffix.lower() != ".ply":
        # 非 PLY 格式（如 pcd、xyz 等）内置解析器不支持，必须依赖 Open3D。
        raise RuntimeError("Open3D is required for non-PLY files")
    with path.open("r", encoding="ascii", errors="strict") as handle:
        # 第一阶段：逐行解析 PLY 头，取顶点数与各属性（列）名称及顺序。
        vertex_count = None
        properties: list[str] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("invalid PLY header")
            fields = line.strip().split()
            if fields[:2] == ["format", "binary_little_endian"]:
                # 内置解析器只认 ASCII，二进制 PLY 需要 Open3D。
                raise RuntimeError("Open3D is required for binary PLY")
            if fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])  # 记录顶点数量。
            elif fields[:1] == ["property"] and vertex_count is not None:
                properties.append(fields[-1])  # 按出现顺序收集属性名（列名）。
            elif fields[:1] == ["end_header"]:
                break  # 头结束，后面是数据区。
        if vertex_count is None or not {"x", "y", "z"}.issubset(properties):
            raise ValueError("PLY has no valid vertex coordinates")
        # 记录 x/y/z 分别在每行中的列索引（PLY 属性顺序不固定，可能夹杂颜色等其它列）。
        indices = [properties.index(axis) for axis in "xyz"]
        # 第二阶段：按顶点数逐行读取数据，仅抽取 xyz 三列。
        rows = []
        for _ in range(vertex_count):
            values = handle.readline().split()
            rows.append([float(values[i]) for i in indices])
    return np.asarray(rows, dtype=float)


def write_ply(path: str | Path, points: np.ndarray, color: tuple[int, int, int] | None = None) -> Path:
    """把点云写成 ASCII PLY 文件，可选给所有点写入统一 RGB 颜色。

    统一着色便于在 CloudCompare 里用不同颜色区分源/目标/对齐后点云。返回写出的路径。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在。
    points = np.asarray(points, dtype=float)
    # 有颜色时才在头里追加 red/green/blue 三个属性声明。
    color_header = "property uchar red\nproperty uchar green\nproperty uchar blue\n" if color else ""
    with path.open("w", encoding="ascii", newline="\n") as handle:
        # 写 PLY 头：格式、顶点数与各属性声明。
        handle.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n{color_header}end_header\n")
        suffix = f" {color[0]} {color[1]} {color[2]}" if color else ""  # 每行末尾追加的颜色文本。
        for point in points:
            # 用 %.9g 输出坐标，兼顾精度与文件体积。
            handle.write(f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g}{suffix}\n")
    return path


def safe_name(value: str) -> str:
    """把任意字符串净化成可安全用作文件名的形式（非字母数字/._- 一律替换为下划线）。"""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
