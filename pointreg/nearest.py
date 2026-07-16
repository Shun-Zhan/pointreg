"""最近邻查询：ICP 每轮建立对应关系的核心操作。

优先用 SciPy 的 cKDTree 加速；若环境没装 SciPy，则退化为分块的
暴力欧氏距离计算，保证纯 NumPy 环境也能跑（只是慢一些）。
"""

from __future__ import annotations

import numpy as np


class NearestNeighborIndex:
    """针对固定参考点云构建、可反复查询的最近邻索引。

    ICP 中目标点云在整个迭代过程里不变，把它的 KD-tree 建好复用，
    避免每轮重复建树，显著降低耗时。
    """

    def __init__(self, reference: np.ndarray):
        self.reference = np.asarray(reference, dtype=float)
        self._tree = None
        if len(self.reference):
            try:
                # 有 SciPy 时用 KD-tree，查询复杂度约 O(log n)
                from scipy.spatial import cKDTree
                self._tree = cKDTree(self.reference)
            except ImportError:
                pass  # 没有 SciPy 则保持 _tree=None，走下面的暴力回退

    def query(self, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """对每个查询点找参考点云中最近的一个，返回 (距离数组, 索引数组)。"""
        query = np.asarray(query, dtype=float)
        if not len(query) or not len(self.reference):
            return np.empty(0), np.empty(0, dtype=int)
        if self._tree is not None:
            # workers=-1 使用全部 CPU 核心并行查询
            distances, indices = self._tree.query(query, k=1, workers=-1)
            return np.asarray(distances), np.asarray(indices)
        # 回退路径：分块计算距离矩阵，控制峰值内存（否则 N×M 矩阵会爆）
        distances, indices = [], []
        chunk = 2048
        for start in range(0, len(query), chunk):
            # 对当前块内每个点，计算到所有参考点的平方距离
            squared = np.sum((query[start:start+chunk, None, :] - self.reference[None, :, :]) ** 2, axis=2)
            idx = np.argmin(squared, axis=1)  # 每行取最近参考点索引
            indices.append(idx)
            distances.append(np.sqrt(squared[np.arange(len(idx)), idx]))
        return np.concatenate(distances), np.concatenate(indices)


def nearest_neighbors(query: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """一次性查询的便捷包装，适合无需反复查询的调用方。"""
    return NearestNeighborIndex(reference).query(query)
