"""最近邻检索：ICP 每轮都要为源点找目标点云中的最近点，是配准的性能热点。

优先用 SciPy 的 cKDTree（KD 树，近似 O(N log N)）；若环境缺少 SciPy，则退回到
纯 NumPy 的分块暴力搜索，保证在没有第三方库时仍能跑通。
"""
from __future__ import annotations

import numpy as np


class NearestNeighborIndex:
    """针对某个固定参考点云的可复用最近邻索引。

    把参考点云一次性建成索引后可反复查询，避免 ICP 每轮重复建树的开销。
    """

    def __init__(self, reference: np.ndarray):
        self.reference = np.asarray(reference, dtype=float)
        self._tree = None
        if len(self.reference):
            try:
                # 优先构建 KD 树，查询远快于暴力法。
                from scipy.spatial import cKDTree
                self._tree = cKDTree(self.reference)
            except ImportError:
                # 没装 SciPy 时保持 _tree 为 None，查询时走 NumPy 备用路径。
                pass

    def query(self, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """为每个查询点找到参考点云中的最近点。

        参数:
            query: 形状 (M, 3) 的查询点集。
        返回:
            (distances, indices)：长度均为 M；distances 是到最近点的欧氏距离，
            indices 是对应最近点在参考点云中的下标。查询或参考为空时返回空数组。
        """
        query = np.asarray(query, dtype=float)
        if not len(query) or not len(self.reference):
            return np.empty(0), np.empty(0, dtype=int)
        if self._tree is not None:
            # KD 树查询：k=1 取最近的一个点，workers=-1 使用全部 CPU 核并行。
            distances, indices = self._tree.query(query, k=1, workers=-1)
            return np.asarray(distances), np.asarray(indices)
        # 备用路径：纯 NumPy 暴力最近邻。为避免一次性构造 (M, N) 的巨大距离矩阵
        # 撑爆内存，按 chunk 大小分批计算查询点到全部参考点的平方距离。
        distances, indices = [], []
        chunk = 2048
        for start in range(0, len(query), chunk):
            # 广播出 (chunk, N) 的平方距离矩阵；用平方距离比较即可，省去开方。
            squared = np.sum((query[start:start+chunk, None, :] - self.reference[None, :, :]) ** 2, axis=2)
            idx = np.argmin(squared, axis=1)  # 每个查询点取距离最小的参考点下标。
            indices.append(idx)
            # 仅对选中的最近点开方还原真实距离。
            distances.append(np.sqrt(squared[np.arange(len(idx)), idx]))
        return np.concatenate(distances), np.concatenate(indices)


def nearest_neighbors(query: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """一次性最近邻查询的便捷封装，供无需重复查询的调用方使用。

    内部就是临时建一个索引查一次；如需多次查询同一参考点云，
    请直接复用 NearestNeighborIndex 以免反复建树。
    """
    return NearestNeighborIndex(reference).query(query)
