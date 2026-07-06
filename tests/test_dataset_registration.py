from itertools import permutations
from pathlib import Path

from pointreg.dataset import build_bunny_graph, register_dataset_pair
from pointreg.io import parse_bun_conf
from pointreg.metrics import pose_errors
from pointreg.transforms import relative_transform


DATA = Path("bunny/data").resolve()


def test_bunny_bridge_graph_covers_all_ordered_pairs():
    poses = parse_bun_conf(DATA / "bun.conf")
    graph = build_bunny_graph(str(DATA))
    for source, target in permutations(poses, 2):
        estimated, path = graph.transform(source, target)
        rotation_error, translation_error = pose_errors(estimated, relative_transform(poses[source], poses[target]))
        assert path[0] == source and path[-1] == target
        assert rotation_error < 5.0, f"{source}->{target}: {rotation_error} deg"
        assert translation_error < 0.005, f"{source}->{target}: {translation_error} m"


def test_reported_problem_pair_is_fixed():
    result = register_dataset_pair(DATA, "bun270", "bun045")
    assert result.success
    assert result.metrics["rotation_error_deg"] < 5.0
    assert result.metrics["translation_error_ratio"] < 0.02
    assert result.metrics["bridge_hops"] >= 2
