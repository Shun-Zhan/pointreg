from pointreg.runtime import preload_open3d


def test_open3d_warmup_is_paid_only_once_per_process():
    """验证 Open3D 预热每个进程只付一次成本:首次预热后,再次调用直接返回 0.0(已缓存)。"""
    preload_open3d()                        # 首次预热(可能有耗时)
    assert preload_open3d() == 0.0          # 第二次应命中缓存,零耗时
