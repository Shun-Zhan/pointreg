import json
import numpy as np

from pointreg.cloudcompare import export_cloudcompare


def test_cloudcompare_export(tmp_path):
    points = np.array([[0.,0,0],[1,0,0],[0,1,0]])
    files = export_cloudcompare(tmp_path, points, points, np.eye(4), {"case":"test"})
    assert all(path.exists() for path in files.values())
    manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
    assert manifest["metadata"]["case"] == "test"

