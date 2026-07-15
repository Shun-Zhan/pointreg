from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"


def test_webui_recommends_fusion_only_for_low_overlap_pair():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert "低覆盖 GeoTransformer（Fusion，推荐）" not in app.sidebar.selectbox[2].options

    app.sidebar.selectbox[1].select("bun180").run()
    assert not app.exception
    assert any("检测到低重合点对" in warning.value for warning in app.warning)
    assert "低覆盖 GeoTransformer（Fusion，推荐）" in app.sidebar.selectbox[2].options


def test_webui_keeps_original_method_for_normal_overlap_pair():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert app.sidebar.selectbox[0].value == "bun000"
    assert app.sidebar.selectbox[1].value == "bun045"
    assert app.sidebar.selectbox[2].value == "fpfh"
    assert not any("检测到低重合点对" in warning.value for warning in app.warning)


def test_webui_recommends_fusion_for_pair_below_fifty_percent():
    app = AppTest.from_file("app.py")
    app.run(timeout=30)
    app.selectbox(key="source").select("bun045")
    app.selectbox(key="target").select("bun270")
    app.run(timeout=30)

    assert any("检测到低重合点对" in warning.value for warning in app.warning)
    coarse = app.selectbox(key="coarse")
    assert len(coarse.options) == 7
    assert "GeoTransformer" in coarse.options[0]
