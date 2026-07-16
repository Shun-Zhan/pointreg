from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"


def test_webui_recommends_fusion_only_for_low_overlap_pair():
    """验证 Web UI:默认高重合对不出现 Fusion 选项;切到低重合对(bun180)后才提示并解锁 Fusion。"""
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert "低覆盖 GeoTransformer（Fusion，推荐）" not in app.sidebar.selectbox[2].options  # 初始无 Fusion

    app.sidebar.selectbox[1].select("bun180").run()   # 把目标换成低重合的 bun180
    assert not app.exception
    assert any("检测到低重合点对" in warning.value for warning in app.warning)          # 出现低重合提示
    assert "低覆盖 GeoTransformer（Fusion，推荐）" in app.sidebar.selectbox[2].options  # Fusion 被解锁


def test_webui_keeps_original_method_for_normal_overlap_pair():
    """验证正常重合对(bun000/bun045)的默认选择保持为 fpfh,且不弹低重合警告。"""
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert app.sidebar.selectbox[0].value == "bun000"     # 默认源
    assert app.sidebar.selectbox[1].value == "bun045"     # 默认目标
    assert app.sidebar.selectbox[2].value == "fpfh"       # 默认方法保持 fpfh
    assert not any("检测到低重合点对" in warning.value for warning in app.warning)  # 无低重合警告


def test_webui_recommends_fusion_for_pair_below_fifty_percent():
    """验证选择重合率低于 50% 的点对(bun045/bun270)时,弹出提示且粗配选项以 GeoTransformer 为首。"""
    app = AppTest.from_file("app.py")
    app.run(timeout=30)
    app.selectbox(key="source").select("bun045")
    app.selectbox(key="target").select("bun270")
    app.run(timeout=30)

    assert any("检测到低重合点对" in warning.value for warning in app.warning)
    coarse = app.selectbox(key="coarse")
    assert len(coarse.options) == 7                    # 低重合时提供全部 7 个粗配选项
    assert "GeoTransformer" in coarse.options[0]       # 首选推荐 GeoTransformer
