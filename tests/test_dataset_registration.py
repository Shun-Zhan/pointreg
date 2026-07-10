from pathlib import Path

from pointreg.experiments import run_all_pairs
from pointreg.io import parse_bun_conf
from pointreg.models import RegistrationConfig
from pointreg.pipeline import register_pair
from pointreg.transforms import relative_transform


DATA = Path("bunny/data").resolve()


def test_fpfh_registration_uses_only_selected_source_and_target():
    poses = parse_bun_conf(DATA / "bun.conf")
    source_name, target_name = "bun000", "bun045"
    config = RegistrationConfig(coarse_method="fpfh", fine_method="custom_icp")
    ground_truth = relative_transform(poses[source_name], poses[target_name])
    result = register_pair(DATA / f"{source_name}.ply", DATA / f"{target_name}.ply", config, ground_truth=ground_truth)

    assert result.success
    assert result.metrics["rotation_error_deg"] < 5.0
    assert result.metrics["translation_error_ratio"] < 0.02
    assert "bridge_hops" not in result.metrics
    assert "bridge_graph" not in result.timings_ms
    assert len(result.history) > 0
    assert {record.stage for record in result.history} == {"direct"}


def test_direct_registration_keeps_stage_timings_pair_scoped():
    config = RegistrationConfig(coarse_method="pca", fine_method="custom_icp", max_iterations=5)
    result = register_pair(DATA / "bun000.ply", DATA / "bun045.ply", config)

    measured_stages = ["runtime_warmup", "load", "preprocess", "coarse", "fine"]
    assert all(result.timings_ms[name] >= 0 for name in measured_stages)
    assert "bridge_graph" not in result.timings_ms
    assert result.timings_ms["total"] >= sum(result.timings_ms[name] for name in measured_stages if name != "runtime_warmup")


def test_all_pairs_report_uses_direct_registration(tmp_path):
    frame = run_all_pairs(DATA, tmp_path, pairs=[("bun000", "bun045")])

    assert len(frame) == 1
    assert frame.iloc[0]["success"]
    assert "bridge_graph" not in frame.columns
    assert (tmp_path / "all_pairs.csv").exists()
