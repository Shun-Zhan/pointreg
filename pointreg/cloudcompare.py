"""CloudCompare 集成：把配准结果导出为带色 PLY 并可选一键打开可视化。

用于人工核验配准质量——源(红)/目标(蓝)/配准后源(绿)三片叠加显示。
跨平台定位 CloudCompare 可执行文件（环境变量 > PATH > 常见安装路径）。
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
    """按优先级定位 CloudCompare 可执行文件，找不到返回 None。

    查找顺序：环境变量 CLOUDCOMPARE_PATH → 系统 PATH → 各操作系统常见安装路径。
    """
    # 1) 用户通过环境变量显式指定
    configured = os.environ.get("CLOUDCOMPARE_PATH")
    if configured and Path(configured).exists():
        return Path(configured)
    # 2) 在 PATH 中查找（兼容不同大小写与 .exe）
    for name in ("CloudCompare", "cloudcompare", "CloudCompare.exe"):
        executable = shutil.which(name)
        if executable:
            return Path(executable)
    # 3) 按操作系统回退到常见安装位置
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
    return next((path for path in candidates if path.exists()), None)


def export_cloudcompare(output_dir: str | Path, source: np.ndarray, target: np.ndarray, transform: np.ndarray, metadata: dict | None = None) -> dict[str, Path]:
    """把配准结果导出到目录：红色源、蓝色目标、绿色配准后源，外加变换矩阵与清单。

    返回各产物文件路径的字典，便于调用方（CLI/UI）后续启动 CloudCompare 或提示用户。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "source": write_ply(output_dir / "source_red.ply", source, (220, 65, 65)),      # 原始源(红)
        "target": write_ply(output_dir / "target_blue.ply", target, (60, 120, 230)),     # 目标(蓝)
        "aligned": write_ply(output_dir / "source_aligned_green.ply", apply_transform(source, transform), (55, 180, 90)),  # 配准后源(绿)
        "matrix": output_dir / "transformation.txt",
        "manifest": output_dir / "manifest.json",
    }
    np.savetxt(files["matrix"], transform, fmt="%.12g")  # 保存 4×4 变换矩阵为文本
    # 清单记录各文件名与附带的元数据（指标、参数等），方便复现与追溯
    manifest = {"files": {key: str(path.name) for key, path in files.items() if key != "manifest"}, "metadata": metadata or {}}
    files["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return files


def launch_cloudcompare(files: list[str | Path]) -> tuple[bool, str]:
    """尝试用 CloudCompare 打开给定文件，返回 (是否成功, 提示信息)。

    用 Popen 非阻塞启动，避免卡住主流程；找不到程序或启动失败时优雅降级，
    返回 False 与说明而非抛异常（文件此时已导出，用户可手动打开）。
    """
    executable = find_cloudcompare()
    if not executable:
        return False, "未找到 CloudCompare；文件已经导出，可手动打开。"
    try:
        # 解析为绝对路径后作为参数传入，交由 CloudCompare 加载
        subprocess.Popen([str(executable), *[str(Path(f).resolve()) for f in files]])
        return True, f"已启动 {executable}"
    except OSError as exc:
        return False, f"CloudCompare 启动失败：{exc}"
