from __future__ import annotations

import numpy as np


class NearestNeighborIndex:
    """Reusable nearest-neighbor index for a fixed reference point cloud."""

    def __init__(self, reference: np.ndarray):
        self.reference = np.asarray(reference, dtype=float)
        self._tree = None
        if len(self.reference):
            try:
                from scipy.spatial import cKDTree
                self._tree = cKDTree(self.reference)
            except ImportError:
                pass

    def query(self, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        query = np.asarray(query, dtype=float)
        if not len(query) or not len(self.reference):
            return np.empty(0), np.empty(0, dtype=int)
        if self._tree is not None:
            distances, indices = self._tree.query(query, k=1, workers=-1)
            return np.asarray(distances), np.asarray(indices)
        distances, indices = [], []
        chunk = 2048
        for start in range(0, len(query), chunk):
            squared = np.sum((query[start:start+chunk, None, :] - self.reference[None, :, :]) ** 2, axis=2)
            idx = np.argmin(squared, axis=1)
            indices.append(idx)
            distances.append(np.sqrt(squared[np.arange(len(idx)), idx]))
        return np.concatenate(distances), np.concatenate(indices)


def nearest_neighbors(query: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One-shot compatibility wrapper for callers without repeated queries."""
    return NearestNeighborIndex(reference).query(query)
