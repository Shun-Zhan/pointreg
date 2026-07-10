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
    configured = os.environ.get("CLOUDCOMPARE_PATH")
    if configured and Path(configured).exists():
        return Path(configured)
    for name in ("CloudCompare", "cloudcompare", "CloudCompare.exe"):
        executable = shutil.which(name)
        if executable:
            return Path(executable)
    candidates: list[Path] = []
    system = platform.system()
    if system == "Darwin":
        candidates = [
            Path("/Volumes/CloudCompare/CloudCompare.app/Contents/MacOS/CloudCompare"),
            Path("/Applications/CloudCompare.app/Contents/MacOS/CloudCompare"),
            Path.home() / "Applications/CloudCompare.app/Contents/MacOS/CloudCompare",
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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "source": write_ply(output_dir / "source_red.ply", source, (220, 65, 65)),
        "target": write_ply(output_dir / "target_blue.ply", target, (60, 120, 230)),
        "aligned": write_ply(output_dir / "source_aligned_green.ply", apply_transform(source, transform), (55, 180, 90)),
        "matrix": output_dir / "transformation.txt",
        "manifest": output_dir / "manifest.json",
    }
    np.savetxt(files["matrix"], transform, fmt="%.12g")
    manifest = {"files": {key: str(path.name) for key, path in files.items() if key != "manifest"}, "metadata": metadata or {}}
    files["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return files


def launch_cloudcompare(files: list[str | Path]) -> tuple[bool, str]:
    executable = find_cloudcompare()
    if not executable:
        return False, "未找到 CloudCompare；文件已经导出，可手动打开。"
    try:
        subprocess.Popen([str(executable), *[str(Path(f).resolve()) for f in files]])
        return True, f"已启动 {executable}"
    except OSError as exc:
        return False, f"CloudCompare 启动失败：{exc}"
