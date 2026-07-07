import numpy as np

from pointreg import preprocessing
from pointreg.preprocessing import voxel_downsample


def test_voxel_downsample_averages_points_in_each_cell():
    points = np.array([[.01, .01, .01], [.02, .03, .01], [1.01, 1.01, 1.01]])
    result = voxel_downsample(points, 1.0)
    assert np.allclose(result, [[.015, .02, .01], [1.01, 1.01, 1.01]])


def test_voxel_downsample_accepts_numpy2_shaped_inverse(monkeypatch):
    original_unique = preprocessing.np.unique

    def unique_with_column_inverse(*args, **kwargs):
        unique, inverse = original_unique(*args, **kwargs)
        return unique, inverse.reshape(-1, 1)

    monkeypatch.setattr(preprocessing.np, "unique", unique_with_column_inverse)
    points = np.array([[.01, .01, .01], [.02, .03, .01], [1.01, 1.01, 1.01]])
    result = voxel_downsample(points, 1.0)
    assert np.allclose(result, [[.015, .02, .01], [1.01, 1.01, 1.01]])
