"""Adapter for the official GeoTransformer ModelNet checkpoint.

The upstream implementation remains vendored under ``third_party``.  This
module only normalizes PointReg clouds, prepares the original packed input, and
converts its source-to-reference estimate back to PointReg coordinates.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / "third_party" / "GeoTransformer-main"
EXPERIMENT_ROOT = UPSTREAM_ROOT / "experiments" / "geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn"
EXPERIMENT_3DMATCH_ROOT = UPSTREAM_ROOT / "experiments" / "geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "geotransformer-modelnet.pth.tar"
DEFAULT_3DMATCH_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "geotransformer-3dmatch.pth.tar"
NEIGHBOR_LIMITS_3DMATCH = [38, 36, 36, 38]


@dataclass(frozen=True, slots=True)
class GeoTransformerCorrespondences:
    """Dense 3DMatch correspondences and LGR pose in PointReg coordinates."""

    source_points: np.ndarray
    target_points: np.ndarray
    scores: np.ndarray
    lgr_transform: np.ndarray
    scale: float


def _sample_points(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Sample a fixed number of points deterministically, matching ModelNet size."""
    if len(points) < 3:
        raise ValueError("GeoTransformer needs at least three points per cloud")
    rng = np.random.default_rng(seed)
    replace = len(points) < count
    indices = rng.choice(len(points), size=count, replace=replace)
    return np.asarray(points[indices], dtype=np.float32)


def _normalize_pair(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Apply one shared similarity transform so the rigid pose is preserved."""
    joined = np.concatenate([source, target], axis=0)
    center = joined.mean(axis=0)
    scale = float(np.max(np.linalg.norm(joined - center, axis=1)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("cannot normalize a degenerate point-cloud pair")
    return (source - center) / scale, (target - center) / scale, center, scale


def _move_to_device(value, device):
    import torch

    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return value


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _decode_3dmatch_output(output_dict: dict, scale: float) -> GeoTransformerCorrespondences:
    """Convert upstream 3DMatch outputs back from its working scale."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("3DMatch scale must be finite and positive")
    source = _to_numpy(output_dict["src_corr_points"]).astype(np.float64, copy=False) / scale
    target = _to_numpy(output_dict["ref_corr_points"]).astype(np.float64, copy=False) / scale
    scores = _to_numpy(output_dict["corr_scores"]).astype(np.float64, copy=False).reshape(-1)
    estimate_s = _to_numpy(output_dict["estimated_transform"]).astype(np.float64, copy=False)
    if source.ndim != 2 or source.shape[1] != 3 or target.shape != source.shape:
        raise ValueError("GeoTransformer correspondence points must be matching (N, 3) arrays")
    if len(scores) != len(source):
        raise ValueError("GeoTransformer scores must match the correspondence count")
    if estimate_s.shape != (4, 4):
        raise ValueError("GeoTransformer estimated transform must have shape (4, 4)")
    if not (np.isfinite(source).all() and np.isfinite(target).all() and np.isfinite(scores).all()):
        raise ValueError("GeoTransformer returned non-finite correspondences")
    scores = np.maximum(scores, 0.0)
    maximum = float(scores.max()) if len(scores) else 0.0
    scores = scores / maximum if maximum > 0 else np.ones(len(scores), dtype=np.float64)
    estimate = estimate_s.copy()
    estimate[:3, 3] /= scale
    return GeoTransformerCorrespondences(source, target, scores, estimate, float(scale))


def _enable_upstream_cpu_compat(torch) -> None:
    """Make the CUDA-hardcoded 2022 upstream inference code run on CPU.

    GeoTransformer allocates intermediate tensors with ``tensor.cuda()`` in
    several inference-only modules.  Those allocations should follow the
    existing CPU tensors when the optional CUDA wheel is not installed.
    """
    if torch.cuda.is_available() or getattr(torch.Tensor.cuda, "_pointreg_cpu_cuda_patch", False):
        return

    def _cpu_cuda(self, *args, **kwargs):
        return self

    _cpu_cuda._pointreg_cpu_cuda_patch = True
    torch.Tensor.cuda = _cpu_cuda


def _import_experiment(experiment_root: Path):
    """Import an upstream experiment without leaking its generic module names."""
    module_names = ("config", "model", "backbone")
    previous = {name: sys.modules.get(name) for name in module_names}
    for name in module_names:
        sys.modules.pop(name, None)
    old_path = sys.path.copy()
    try:
        sys.path.insert(0, str(UPSTREAM_ROOT))
        sys.path.insert(0, str(experiment_root))
        config = importlib.import_module("config")
        model_module = importlib.import_module("model")
    finally:
        sys.path[:] = old_path
        for name in module_names:
            sys.modules.pop(name, None)
        for name, module in previous.items():
            if module is not None:
                sys.modules[name] = module
    return config, model_module


@lru_cache(maxsize=2)
def _load_model(checkpoint_path: str):
    import torch

    checkpoint = Path(checkpoint_path)
    if not UPSTREAM_ROOT.is_dir():
        raise RuntimeError(f"GeoTransformer source not found: {UPSTREAM_ROOT}")
    if not checkpoint.is_file() or checkpoint.stat().st_size < 20_000_000:
        raise RuntimeError(
            f"GeoTransformer ModelNet checkpoint is missing or incomplete: {checkpoint}. "
            "Download geotransformer-modelnet.pth.tar into checkpoints/."
        )
    config, model_module = _import_experiment(EXPERIMENT_ROOT)
    cfg = config.make_cfg()
    model = model_module.create_model(cfg)
    snapshot = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = snapshot.get("model", snapshot)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "GeoTransformer checkpoint architecture mismatch "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
    model.eval()
    return cfg, model


@lru_cache(maxsize=2)
def _load_3dmatch_model(checkpoint_path: str):
    """Load the official 3DMatch architecture without changing ModelNet defaults."""
    import torch

    checkpoint = Path(checkpoint_path)
    if not UPSTREAM_ROOT.is_dir():
        raise RuntimeError(f"GeoTransformer source not found: {UPSTREAM_ROOT}")
    if not checkpoint.is_file() or checkpoint.stat().st_size < 20_000_000:
        raise RuntimeError(f"GeoTransformer 3DMatch checkpoint is missing or incomplete: {checkpoint}")
    config, model_module = _import_experiment(EXPERIMENT_3DMATCH_ROOT)
    cfg = config.make_cfg()
    model = model_module.create_model(cfg)
    snapshot = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = snapshot.get("model", snapshot)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return cfg, model


def geotransformer_3dmatch_correspondences(
    source: np.ndarray,
    target: np.ndarray,
    *,
    checkpoint: str | Path | None = None,
    voxel_size: float = 0.0025,
) -> GeoTransformerCorrespondences:
    """Run 3DMatch GeoTransformer and expose its dense source-to-target matches."""
    import torch

    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive for 3DMatch inference")
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("source must be an (N, 3) array with N >= 3")
    if target.ndim != 2 or target.shape[1] != 3 or len(target) < 3:
        raise ValueError("target must be an (N, 3) array with N >= 3")
    checkpoint_path = Path(checkpoint) if checkpoint is not None else DEFAULT_3DMATCH_CHECKPOINT
    _enable_upstream_cpu_compat(torch)
    cfg, model = _load_3dmatch_model(str(checkpoint_path.resolve()))
    scale = float(cfg.backbone.init_voxel_size / voxel_size)

    from geotransformer.utils.data import registration_collate_fn_stack_mode

    sample = {
        "ref_points": (target * scale).astype(np.float32),
        "src_points": (source * scale).astype(np.float32),
        "ref_feats": np.ones((len(target), 1), dtype=np.float32),
        "src_feats": np.ones((len(source), 1), dtype=np.float32),
        "transform": np.eye(4, dtype=np.float32),
    }
    data_dict = registration_collate_fn_stack_mode(
        [sample],
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        NEIGHBOR_LIMITS_3DMATCH,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data_dict = _move_to_device(data_dict, device)
    with torch.no_grad():
        output_dict = model(data_dict)
    return _decode_3dmatch_output(output_dict, scale)


def geotransformer_registration(
    source: np.ndarray,
    target: np.ndarray,
    *,
    checkpoint: str | Path | None = None,
    num_points: int = 717,
    seed: int = 42,
) -> np.ndarray:
    """Estimate the PointReg source-to-target transform with GeoTransformer."""
    import torch

    checkpoint_path = Path(checkpoint) if checkpoint is not None else DEFAULT_CHECKPOINT
    _enable_upstream_cpu_compat(torch)
    cfg, model = _load_model(str(checkpoint_path.resolve()))
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_n, target_n, center, scale = _normalize_pair(source, target)
    source_n = _sample_points(source_n, num_points, seed)
    target_n = _sample_points(target_n, num_points, seed + 1)

    from geotransformer.utils.data import registration_collate_fn_stack_mode

    sample = {
        "ref_points": target_n,
        "src_points": source_n,
        "ref_feats": np.ones((len(target_n), 1), dtype=np.float32),
        "src_feats": np.ones((len(source_n), 1), dtype=np.float32),
        # Only used to form training labels inside the upstream forward pass;
        # it does not participate in inference matching.
        "transform": np.eye(4, dtype=np.float32),
    }
    data_dict = registration_collate_fn_stack_mode(
        [sample],
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        [64] * cfg.backbone.num_stages,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data_dict = _move_to_device(data_dict, device)
    with torch.no_grad():
        estimate_n = model(data_dict)["estimated_transform"].detach().cpu().numpy()
    rotation = estimate_n[:3, :3]
    translation = scale * estimate_n[:3, 3] + center - rotation @ center
    estimate = np.eye(4)
    estimate[:3, :3] = rotation
    estimate[:3, 3] = translation
    return estimate
