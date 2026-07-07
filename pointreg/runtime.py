from __future__ import annotations

from threading import Lock
from time import perf_counter

_OPEN3D_READY = False
_OPEN3D_LOCK = Lock()


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
