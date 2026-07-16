import json
import platform
from pathlib import Path

import numpy as np
import pytest

from pointreg import cloudcompare as cc_module
from pointreg.cloudcompare import export_cloudcompare, find_cloudcompare


def test_cloudcompare_export(tmp_path):
    """验证导出 CloudCompare 可视化文件:所有文件都生成,且 manifest 里写入了元数据。"""
    points = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    files = export_cloudcompare(tmp_path, points, points, np.eye(4), {"case": "test"})
    assert all(path.exists() for path in files.values())            # 每个导出文件都真实落盘
    manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
    assert manifest["metadata"]["case"] == "test"                   # 元数据被正确写入清单


def test_find_cloudcompare_from_env(tmp_path, monkeypatch):
    """验证优先从环境变量 CLOUDCOMPARE_PATH 定位 CloudCompare 可执行文件。"""
    exe = tmp_path / "CloudCompare"
    exe.touch()
    monkeypatch.setenv("CLOUDCOMPARE_PATH", str(exe))
    assert find_cloudcompare() == exe


def test_find_cloudcompare_which_lowercase(monkeypatch):
    """验证无环境变量时,能通过 which 找到小写命令名 cloudcompare。"""
    monkeypatch.delenv("CLOUDCOMPARE_PATH", raising=False)

    def which(name: str) -> str | None:
        if name == "cloudcompare":
            return "/usr/bin/cloudcompare"
        return None

    monkeypatch.setattr(cc_module.shutil, "which", which)
    assert find_cloudcompare() == Path("/usr/bin/cloudcompare")


def test_find_cloudcompare_in_macos_user_applications(tmp_path, monkeypatch):
    """验证在 macOS 上能到用户 Applications 目录里的 .app 包内找到可执行文件。"""
    monkeypatch.delenv("CLOUDCOMPARE_PATH", raising=False)
    monkeypatch.setattr(cc_module.shutil, "which", lambda _: None)
    monkeypatch.setattr(cc_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cc_module.Path, "home", classmethod(lambda cls: tmp_path))
    executable = tmp_path / "Applications/CloudCompare.app/Contents/MacOS/CloudCompare"
    executable.parent.mkdir(parents=True)
    executable.touch()
    assert find_cloudcompare() == executable


@pytest.mark.skipif(platform.system() != "Linux", reason="Linux integration smoke")
def test_find_cloudcompare_linux_smoke(monkeypatch):
    """Linux 冒烟测试:查找函数要么返回 None,要么返回一个真实存在的路径。"""
    monkeypatch.delenv("CLOUDCOMPARE_PATH", raising=False)
    result = find_cloudcompare()
    assert result is None or result.exists()
