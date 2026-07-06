from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from .transforms import make_transform, quaternion_xyzw_to_matrix


def parse_bun_conf(path: str | Path) -> dict[str, np.ndarray]:
    poses: dict[str, np.ndarray] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or fields[0] != "bmesh" or len(fields) != 9:
            continue
        name = Path(fields[1]).stem
        values = np.asarray([float(v) for v in fields[2:]], dtype=float)
        # Stanford's legacy ZipPack/Vrip convention uses the transpose of the
        # active column-vector rotation produced by the usual xyzw formula.
        poses[name] = make_transform(quaternion_xyzw_to_matrix(values[3:]).T, values[:3])
    if not poses:
        raise ValueError(f"no bmesh poses found in {path}")
    return poses


def read_points(path: str | Path) -> np.ndarray:
    path = Path(path)
    try:
        import open3d as o3d
        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points, dtype=float)
        if len(points):
            return points
    except ImportError:
        pass
    if path.suffix.lower() != ".ply":
        raise RuntimeError("Open3D is required for non-PLY files")
    with path.open("r", encoding="ascii", errors="strict") as handle:
        vertex_count = None
        properties: list[str] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("invalid PLY header")
            fields = line.strip().split()
            if fields[:2] == ["format", "binary_little_endian"]:
                raise RuntimeError("Open3D is required for binary PLY")
            if fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])
            elif fields[:1] == ["property"] and vertex_count is not None:
                properties.append(fields[-1])
            elif fields[:1] == ["end_header"]:
                break
        if vertex_count is None or not {"x", "y", "z"}.issubset(properties):
            raise ValueError("PLY has no valid vertex coordinates")
        indices = [properties.index(axis) for axis in "xyz"]
        rows = []
        for _ in range(vertex_count):
            values = handle.readline().split()
            rows.append([float(values[i]) for i in indices])
    return np.asarray(rows, dtype=float)


def write_ply(path: str | Path, points: np.ndarray, color: tuple[int, int, int] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=float)
    color_header = "property uchar red\nproperty uchar green\nproperty uchar blue\n" if color else ""
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n{color_header}end_header\n")
        suffix = f" {color[0]} {color[1]} {color[2]}" if color else ""
        for point in points:
            handle.write(f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g}{suffix}\n")
    return path


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
