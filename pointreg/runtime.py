"""运行时工具：负责 Open3D 的一次性预热（懒加载）。

Open3D 首次 import 时需要加载底层 C++ 动态库，耗时可达数百毫秒。若把这段开销
混进第一次配准的计时里，会污染性能统计。这里在正式配准前主动“预热”一次，并把
预热耗时单独记录，从而让后续的配准计时更干净、更可比。
"""
from __future__ import annotations

from threading import Lock
from time import perf_counter

# 进程级的全局标志：Open3D 是否已经完成预热。用模块级变量保证一个进程只热身一次。
_OPEN3D_READY = False
# 互斥锁：防止多线程同时预热时出现竞态（重复加载或状态错乱）。
_OPEN3D_LOCK = Lock()


def preload_open3d() -> float:
    """在本进程内仅加载一次 Open3D，并返回“本次调用”花费的预热时间（毫秒）。

    返回 0 表示：Open3D 之前已经就绪，或者环境里根本没装 Open3D、需要退回到
    纯 NumPy/SciPy 的备用实现（此时不算作预热开销）。
    """
    global _OPEN3D_READY
    if _OPEN3D_READY:  # 快速路径：已经热身过，无需加锁直接返回。
        return 0.0
    with _OPEN3D_LOCK:
        # 双重检查锁定：拿到锁后再确认一次，避免多个线程重复执行预热逻辑。
        if _OPEN3D_READY:
            return 0.0
        started = perf_counter()
        try:
            import open3d as o3d
        except ImportError:
            # 未安装 Open3D 时不视为错误，交由上层使用 NumPy/SciPy 备用路径。
            return 0.0
        # 主动构造一个原生类型对象，触发底层动态库与 Python 绑定的真正初始化，
        # 让首次 import 的隐藏开销在此处一次性发生。
        o3d.geometry.PointCloud()
        _OPEN3D_READY = True
        return (perf_counter() - started) * 1000
