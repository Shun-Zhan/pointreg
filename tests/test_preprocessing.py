import numpy as np

from pointreg import preprocessing
from pointreg.preprocessing import voxel_downsample


def test_voxel_downsample_averages_points_in_each_cell():
    """验证体素下采样:同一体素内的点取均值,不同体素各自保留一个代表点。"""
    points = np.array([[.01, .01, .01], [.02, .03, .01], [1.01, 1.01, 1.01]])
    result = voxel_downsample(points, 1.0)
    # 前两点落在同一体素,取均值 (.015,.02,.01);第三点单独成一格
    assert np.allclose(result, [[.015, .02, .01], [1.01, 1.01, 1.01]])


def test_voxel_downsample_accepts_numpy2_shaped_inverse(monkeypatch):
    """验证兼容 NumPy 2:即便 np.unique 返回列向量形状的 inverse,下采样结果仍正确。"""
    original_unique = preprocessing.np.unique

    def unique_with_column_inverse(*args, **kwargs):
        unique, inverse = original_unique(*args, **kwargs)
        return unique, inverse.reshape(-1, 1)

    monkeypatch.setattr(preprocessing.np, "unique", unique_with_column_inverse)
    points = np.array([[.01, .01, .01], [.02, .03, .01], [1.01, 1.01, 1.01]])
    result = voxel_downsample(points, 1.0)
    assert np.allclose(result, [[.015, .02, .01], [1.01, 1.01, 1.01]])  # 结果与一维 inverse 时一致
