from pointreg.runtime import preload_open3d


def test_open3d_warmup_is_paid_only_once_per_process():
    preload_open3d()
    assert preload_open3d() == 0.0
