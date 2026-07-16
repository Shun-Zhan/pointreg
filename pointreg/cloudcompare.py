"""CloudCompare 可视化对接：把配准结果导出成带色 PLY，并可选一键拉起 CloudCompare。

导出源（红）、目标（蓝）、对齐后源（绿）三片点云及变换矩阵，方便在 CloudCompare 里
直观检查配准好坏——这是课程设计演示效果的常用手段。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

import numpy as np

from .io import write_ply
from .transforms import apply_transform


def find_cloudcompare() -> Path | None:
    """在当前系统上定位 CloudCompare 可执行文件，找不到返回 None。

    查找顺序：环境变量 CLOUDCOMPARE_PATH -> 系统 PATH -> 各平台常见安装路径。
    """
    # 1) 优先用环境变量显式指定的路径。
    configured = os.environ.get("CLOUDCOMPARE_PATH")
    if configured and Path(configured).exists():
        return Path(configured)
    # 2) 在 PATH 里按常见命令名查找。
    for name in ("CloudCompare", "cloudcompare", "CloudCompare.exe"):
        executable = shutil.which(name)
        if executable:
            return Path(executable)
    # 3) 兜底：按操作系统枚举默认安装位置。
    candidates: list[Path] = []
    system = platform.system()
    if system == "Darwin":
        candidates = [
            Path("/Applications/CloudCompare.app/Contents/MacOS/CloudCompare"),
            Path.home() / "Applications/CloudCompare.app/Contents/MacOS/CloudCompare",
            Path("/Volumes/CloudCompare/CloudCompare.app/Contents/MacOS/CloudCompare"),
        ]
    elif system == "Windows":
        candidates = [
            Path(r"C:\Program Files\CloudCompare\CloudCompare.exe"),
            Path(r"C:\Program Files (x86)\CloudCompare\CloudCompare.exe"),
        ]
    elif system == "Linux":
        candidates = [
            Path("/usr/bin/CloudCompare"),
            Path("/usr/bin/cloudcompare"),
            Path("/snap/bin/cloudcompare"),
            Path("/snap/bin/CloudCompare"),
        ]
    # 返回第一个真实存在的候选路径，全都不存在则返回 None。
    return next((path for path in candidates if path.exists()), None)


def export_cloudcompare(output_dir: str | Path, source: np.ndarray, target: np.ndarray, transform: np.ndarray, metadata: dict | None = None) -> dict[str, Path]:
    """把一次配准的可视化素材导出到 output_dir，返回各产物路径的字典。

    产物包括：红色源点云、蓝色目标点云、绿色“对齐后源点云”（源施加 transform 后），
    以及文本形式的变换矩阵和记录清单/元数据的 manifest.json。
    源与对齐后源分色，能一眼看出配准前后位置的变化。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        # 三片点云分别着不同颜色：红=源、蓝=目标、绿=对齐后的源。
        "source": write_ply(output_dir / "source_red.ply", source, (220, 65, 65)),
        "target": write_ply(output_dir / "target_blue.ply", target, (60, 120, 230)),
        "aligned": write_ply(output_dir / "source_aligned_green.ply", apply_transform(source, transform), (55, 180, 90)),
        "matrix": output_dir / "transformation.txt",
        "manifest": output_dir / "manifest.json",
    }
    np.savetxt(files["matrix"], transform, fmt="%.12g")  # 以高精度文本保存 4x4 变换矩阵。
    # manifest 记录各产物文件名（相对名）与附带的元数据，便于程序化追溯。
    manifest = {"files": {key: str(path.name) for key, path in files.items() if key != "manifest"}, "metadata": metadata or {}}
    files["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return files


def launch_cloudcompare(files: list[str | Path]) -> tuple[bool, str]:
    """尝试用 CloudCompare 打开给定文件，返回 (是否成功, 提示信息)。

    找不到 CloudCompare 或启动失败都不抛异常，只返回 False 及说明，
    因为文件此时已经导出、可手动打开，配准流程不应因可视化失败而中断。
    """
    executable = find_cloudcompare()
    if not executable:
        return False, "未找到 CloudCompare；文件已经导出，可手动打开。"
    try:
        # 非阻塞方式启动：传入解析成绝对路径的文件列表，不等待 CloudCompare 退出。
        subprocess.Popen([str(executable), *[str(Path(f).resolve()) for f in files]])
        return True, f"已启动 {executable}"
    except OSError as exc:
        return False, f"CloudCompare 启动失败：{exc}"
