import json
import platform
from pathlib import Path

import numpy as np
import pytest

from pointreg import cloudcompare as cc_module
from pointreg.cloudcompare import export_cloudcompare, find_cloudcompare


def test_cloudcompare_export(tmp_path):
    points = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    files = export_cloudcompare(tmp_path, points, points, np.eye(4), {"case": "test"})
    assert all(path.exists() for path in files.values())
    manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
    assert manifest["metadata"]["case"] == "test"


def test_find_cloudcompare_from_env(tmp_path, monkeypatch):
    exe = tmp_path / "CloudCompare"
    exe.touch()
    monkeypatch.setenv("CLOUDCOMPARE_PATH", str(exe))
    assert find_cloudcompare() == exe


def test_find_cloudcompare_which_lowercase(monkeypatch):
    monkeypatch.delenv("CLOUDCOMPARE_PATH", raising=False)

    def which(name: str) -> str | None:
        if name == "cloudcompare":
            return "/usr/bin/cloudcompare"
        return None

    monkeypatch.setattr(cc_module.shutil, "which", which)
    assert find_cloudcompare() == Path("/usr/bin/cloudcompare")


@pytest.mark.skipif(platform.system() != "Linux", reason="Linux integration smoke")
def test_find_cloudcompare_linux_smoke(monkeypatch):
    monkeypatch.delenv("CLOUDCOMPARE_PATH", raising=False)
    result = find_cloudcompare()
    assert result is None or result.exists()
