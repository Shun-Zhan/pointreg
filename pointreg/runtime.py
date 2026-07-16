"""Open3D 运行时预热工具。

首次导入 Open3D 会加载动态库、耗时较长。这里提供一次性、线程安全的预热，
把这段冷启动开销单独计量出来，避免它混入配准各阶段的耗时统计。
"""

from __future__ import annotations

from threading import Lock
from time import perf_counter

_OPEN3D_READY = False        # 进程级标志：Open3D 是否已完成预热
_OPEN3D_LOCK = Lock()        # 保护上面的标志，防止多线程重复预热


def preload_open3d() -> float:
    """Load Open3D once per process and return this call's warm-up time in ms.

    A return value of 0 means Open3D was already ready, or is unavailable and
    the NumPy/SciPy fallback path should remain usable.
    """
    global _OPEN3D_READY
    if _OPEN3D_READY:
        return 0.0
    with _OPEN3D_LOCK:
        if _OPEN3D_READY:
            return 0.0
        started = perf_counter()
        try:
            import open3d as o3d
        except ImportError:
            return 0.0
        # Touch a native type so dynamic libraries and bindings are initialized.
        o3d.geometry.PointCloud()
        _OPEN3D_READY = True
        return (perf_counter() - started) * 1000
