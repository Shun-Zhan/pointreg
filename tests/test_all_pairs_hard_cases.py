from pathlib import Path

import pytest

from pointreg.io import parse_bun_conf
from pointreg.models import RegistrationConfig
from pointreg.pipeline import register_pair
from pointreg.transforms import relative_transform

DATA = Path("bunny/data").resolve()

HARD_CASES = [
    ("bun000", "bun180"),
    ("bun180", "bun000"),
    ("bun270", "bun045"),
    ("bun045", "bun270"),
    ("bun000", "bun270"),
    ("chin", "bun045"),
    ("bun045", "chin"),
    ("top2", "top3"),
    ("ear_back", "bun180"),
    ("bun090", "bun000"),
]


@pytest.mark.slow
@pytest.mark.parametrize("source_name,target_name", HARD_CASES)
def test_hard_case_pair_succeeds(source_name: str, target_name: str) -> None:
    poses = parse_bun_conf(DATA / "bun.conf")
    config = RegistrationConfig(coarse_method="fpfh", fine_method="custom_icp")
    ground_truth = relative_transform(poses[source_name], poses[target_name])
    result = register_pair(DATA / f"{source_name}.ply", DATA / f"{target_name}.ply", config, ground_truth=ground_truth)
    assert result.success, result.message
    assert result.metrics["rotation_error_deg"] < config.success_rotation_deg
    assert result.metrics["translation_error_ratio"] < config.success_translation_ratio
