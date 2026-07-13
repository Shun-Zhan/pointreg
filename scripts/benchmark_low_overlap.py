"""Compare direct low-overlap registration coarse initializers on Bunny."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pointreg.io import parse_bun_conf
from pointreg.models import RegistrationConfig
from pointreg.pipeline import register_pair
from pointreg.transforms import relative_transform


PAIRS = [("bun000", "bun180"), ("chin", "top2"), ("bun000", "ear_back")]
METHODS = ("fpfh", "fpfh_multiscale", "gcransac")


def main() -> None:
    data_dir = ROOT / "bunny" / "data"
    poses = parse_bun_conf(data_dir / "bun.conf")
    rows = []
    for source_name, target_name in PAIRS:
        ground_truth = relative_transform(poses[source_name], poses[target_name])
        for method in METHODS:
            result = register_pair(
                data_dir / f"{source_name}.ply",
                data_dir / f"{target_name}.ply",
                RegistrationConfig(coarse_method=method, fine_method="custom_icp", voxel_size=0.0025, max_correspondence_distance=0.01),
                ground_truth=ground_truth,
            )
            rows.append(
                {
                    "pair": f"{source_name}->{target_name}",
                    "method": method,
                    "success": result.success,
                    "status": result.status,
                    "rotation_error_deg": result.metrics.get("rotation_error_deg"),
                    "translation_error_ratio": result.metrics.get("translation_error_ratio"),
                    "fitness": result.metrics.get("fitness"),
                    "coarse_ms": result.timings_ms.get("coarse"),
                    "total_ms": result.timings_ms.get("total"),
                }
            )
    output = ROOT / "outputs" / "low_overlap_coarse_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
