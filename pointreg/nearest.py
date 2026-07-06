from __future__ import annotations

import numpy as np


def nearest_neighbors(query: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not len(query) or not len(reference):
        return np.empty(0), np.empty(0, dtype=int)
    try:
        from scipy.spatial import cKDTree
        distances, indices = cKDTree(reference).query(query, k=1, workers=-1)
        return np.asarray(distances), np.asarray(indices)
    except ImportError:
        distances, indices = [], []
        chunk = 2048
        for start in range(0, len(query), chunk):
            squared = np.sum((query[start:start+chunk, None, :] - reference[None, :, :]) ** 2, axis=2)
            idx = np.argmin(squared, axis=1)
            indices.append(idx)
            distances.append(np.sqrt(squared[np.arange(len(idx)), idx]))
        return np.concatenate(distances), np.concatenate(indices)

