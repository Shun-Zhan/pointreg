"""Portable CPU fallback for GeoTransformer's preprocessing extension.

The upstream project ships these two routines as a C++/CUDAExtension.  Its
implementation operates entirely on CPU tensors, so this module preserves the
same public API on Windows when a Visual C++ build toolchain is unavailable.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree


def grid_subsampling(points: torch.Tensor, lengths: torch.Tensor, voxel_size: float):
    """Voxel-average each packed point cloud and return packed tensors."""
    if points.device.type != "cpu":
        raise ValueError("grid_subsampling expects CPU tensors")
    arrays = []
    output_lengths = []
    offset = 0
    for length in lengths.tolist():
        cloud = points[offset : offset + length].numpy()
        offset += length
        voxel_ids = np.floor(cloud / voxel_size).astype(np.int64)
        _, inverse = np.unique(voxel_ids, axis=0, return_inverse=True)
        counts = np.bincount(inverse)
        sampled = np.empty((len(counts), 3), dtype=np.float32)
        for axis in range(3):
            sampled[:, axis] = np.bincount(inverse, weights=cloud[:, axis]) / counts
        arrays.append(sampled)
        output_lengths.append(len(sampled))
    sampled_points = np.concatenate(arrays, axis=0) if arrays else np.empty((0, 3), dtype=np.float32)
    return torch.from_numpy(sampled_points), torch.tensor(output_lengths, dtype=torch.long)


def radius_neighbors(
    q_points: torch.Tensor,
    s_points: torch.Tensor,
    q_lengths: torch.Tensor,
    s_lengths: torch.Tensor,
    radius: float,
):
    """Return packed radius-neighbor indices with the upstream sentinel value."""
    if q_points.device.type != "cpu" or s_points.device.type != "cpu":
        raise ValueError("radius_neighbors expects CPU tensors")
    query = q_points.numpy()
    support = s_points.numpy()
    sentinel = len(support)
    all_neighbors: list[list[int]] = []
    q_offset = s_offset = 0
    for q_length, s_length in zip(q_lengths.tolist(), s_lengths.tolist()):
        q_cloud = query[q_offset : q_offset + q_length]
        s_cloud = support[s_offset : s_offset + s_length]
        tree = cKDTree(s_cloud)
        local_neighbors = tree.query_ball_point(q_cloud, radius)
        for point, indices in zip(q_cloud, local_neighbors):
            indices.sort(key=lambda index: float(np.sum((s_cloud[index] - point) ** 2)))
            all_neighbors.append([s_offset + index for index in indices])
        q_offset += q_length
        s_offset += s_length
    width = max((len(indices) for indices in all_neighbors), default=0)
    result = np.full((len(all_neighbors), width), sentinel, dtype=np.int64)
    for row, indices in enumerate(all_neighbors):
        result[row, : len(indices)] = indices
    return torch.from_numpy(result)
