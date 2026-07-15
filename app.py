from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pointreg.cloudcompare import export_cloudcompare, launch_cloudcompare
from pointreg.fusion import register_low_overlap_pair
from pointreg.io import parse_bun_conf, read_points
from pointreg.metrics import symmetric_overlap
from pointreg.models import RegistrationConfig
from pointreg.pipeline import register_pair
from pointreg.runtime import preload_open3d
from pointreg.transforms import apply_transform, relative_transform

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "bunny" / "data"
LOW_OVERLAP_THRESHOLD = 0.50
OVERLAP_DISTANCE = 0.01
LOW_OVERLAP_METHOD = "low_overlap_geotransformer"
PRACTICAL_TRANSLATION_RATIO = 0.05
st.set_page_config(page_title="PointReg Lab", page_icon="◌", layout="wide")


@st.cache_resource(show_spinner="正在预加载 Open3D…")
def warm_up_runtime() -> None:
    preload_open3d()


@st.cache_data(show_spinner=False)
def load_cloud(name: str) -> np.ndarray:
    return read_points(DATA / f"{name}.ply")


@st.cache_data(show_spinner=False)
def load_bunny_poses() -> dict[str, np.ndarray]:
    conf = DATA / "bun.conf"
    return parse_bun_conf(conf) if conf.exists() else {}


warm_up_runtime()
st.markdown("""<style>
:root {
    --background: #FDFCF8;
    --foreground: #2C2C24;
    --primary: #5D7052;
    --primary-foreground: #F3F4F1;
    --secondary: #C18C5D;
    --secondary-foreground: #FFFFFF;
    --accent: #E6DCCD;
    --accent-foreground: #4A4A40;
    --muted: #F0EBE5;
    --muted-foreground: #78786C;
    --border: #DED8CF;
    --destructive: #A85448;
    --panel: rgba(254, 254, 250, .86);
    --shadow-soft: 0 4px 20px -2px rgba(93, 112, 82, .15);
    --shadow-float: 0 10px 40px -10px rgba(193, 140, 93, .22);
}

.stApp {
    background-color: var(--background);
    background-image:
        radial-gradient(ellipse at 12% 8%, rgba(93, 112, 82, .14) 0%, transparent 42%),
        radial-gradient(ellipse at 88% 22%, rgba(193, 140, 93, .16) 0%, transparent 38%),
        radial-gradient(ellipse at 54% 88%, rgba(230, 220, 205, .72) 0%, transparent 48%),
        linear-gradient(180deg, #FDFCF8 0%, #F8F3EB 100%);
    color: var(--foreground);
    font-family: "Nunito", "Quicksand", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    opacity: .045;
    mix-blend-mode: multiply;
    background-image:
        radial-gradient(circle at 20% 30%, #2C2C24 0 1px, transparent 1.3px),
        radial-gradient(circle at 70% 60%, #5D7052 0 .8px, transparent 1.2px);
    background-size: 17px 19px, 23px 29px;
}
[data-testid="stHeader"] {
    background: rgba(253, 252, 248, .72);
    border-bottom: 1px solid rgba(222, 216, 207, .7);
    backdrop-filter: blur(14px);
}
[data-testid="stSidebar"] {
    background:
        radial-gradient(ellipse at 18% 12%, rgba(193, 140, 93, .18) 0%, transparent 46%),
        linear-gradient(180deg, rgba(240, 235, 229, .92), rgba(253, 252, 248, .92));
    border-right: 1px solid rgba(222, 216, 207, .88);
    box-shadow: 10px 0 40px -28px rgba(93, 112, 82, .35);
    z-index: 2;
}
[data-testid="stSidebarContent"] {
    height: 100%;
    overflow-y: auto !important;
    padding-bottom: 3rem;
}
.block-container {
    color: var(--foreground);
    padding-top: 3rem;
}

h1, h2, h3 {
    color: var(--foreground);
    font-family: "Fraunces", Georgia, "Times New Roman", serif;
    font-weight: 750;
    letter-spacing: 0;
}
.subtitle {
    color: var(--muted-foreground);
    margin-top: -16px;
    margin-bottom: 22px;
    font-weight: 700;
}
p, label,
[data-testid="stMarkdownContainer"],
[data-testid="stWidgetLabel"] p {
    color: var(--foreground);
}
[data-testid="stCaptionContainer"] {
    color: var(--muted-foreground);
}

[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid rgba(222, 216, 207, .72);
    padding: 14px;
    border-radius: 32px 24px 34px 22px / 24px 34px 22px 32px;
    box-shadow: var(--shadow-soft);
    backdrop-filter: blur(10px);
}
[data-testid="stMetricLabel"] p {
    color: var(--muted-foreground);
    font-weight: 800;
}
[data-testid="stMetricValue"] {
    color: var(--primary);
    font-size: 1.45rem;
    font-weight: 800;
    white-space: nowrap;
}

[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
    background-color: rgba(255, 255, 255, .55);
    border: 1px solid var(--border);
    border-radius: 999px;
    color: var(--foreground);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, .72);
}
[data-baseweb="select"] span,
[data-baseweb="input"] input {
    color: var(--foreground);
    font-weight: 700;
}
[data-baseweb="slider"] [role="slider"] {
    background-color: var(--primary);
    border: 2px solid var(--primary-foreground);
    box-shadow: 0 6px 20px -8px rgba(93, 112, 82, .55);
}
[data-baseweb="slider"] div {
    color: var(--muted-foreground);
}

.stTabs [data-baseweb="tab"] {
    color: var(--muted-foreground);
    font-weight: 800;
}
.stTabs [aria-selected="true"] {
    color: var(--primary);
}
.stTabs [data-baseweb="tab-highlight"] {
    background-color: var(--secondary);
}

[data-testid="stButton"] > button,
button[kind="primary"],
button[kind="secondary"] {
    background: var(--primary) !important;
    border: 2px solid rgba(93, 112, 82, .22) !important;
    border-radius: 999px !important;
    color: var(--primary-foreground) !important;
    font-weight: 800 !important;
    letter-spacing: 0 !important;
    min-height: 48px;
    box-shadow: var(--shadow-soft);
    transition: transform .3s ease, box-shadow .3s ease, background-color .3s ease, border-color .3s ease;
}
[data-testid="stButton"] > button p,
button[kind="primary"] p,
button[kind="secondary"] p {
    color: var(--primary-foreground) !important;
}
[data-testid="stButton"] > button:hover,
button[kind="primary"]:hover,
button[kind="secondary"]:hover {
    background: #667A5A !important;
    border-color: rgba(93, 112, 82, .38) !important;
    color: var(--primary-foreground) !important;
    transform: translateY(-1px) scale(1.025);
    box-shadow: 0 12px 28px -14px rgba(93, 112, 82, .42);
}
[data-testid="stButton"] > button:focus,
button[kind="primary"]:focus,
button[kind="secondary"]:focus {
    outline: 0 !important;
    box-shadow: 0 0 0 3px rgba(93, 112, 82, .24), var(--shadow-soft) !important;
    color: var(--primary-foreground) !important;
}
[data-testid="stButton"]:nth-of-type(even) > button,
button[kind="secondary"] {
    background: rgba(255, 255, 255, .38) !important;
    border-color: var(--secondary) !important;
    color: var(--secondary) !important;
}
[data-testid="stButton"]:nth-of-type(even) > button p,
button[kind="secondary"] p {
    color: var(--secondary) !important;
}
[data-testid="stButton"]:nth-of-type(even) > button:hover,
button[kind="secondary"]:hover {
    background: rgba(193, 140, 93, .12) !important;
    color: #9E7048 !important;
}
[data-testid="stButton"] > button:disabled,
button[kind="primary"]:disabled,
button[kind="secondary"]:disabled {
    background: var(--muted) !important;
    border-color: var(--border) !important;
    color: var(--muted-foreground) !important;
    box-shadow: none !important;
    opacity: 1 !important;
}
[data-testid="stButton"] > button:disabled p {
    color: var(--muted-foreground) !important;
}

[data-testid="stAlert"] {
    background-color: rgba(240, 235, 229, .76);
    border: 1px solid rgba(222, 216, 207, .82);
    border-radius: 24px 36px 22px 30px / 28px 22px 34px 24px;
    color: var(--foreground);
    box-shadow: var(--shadow-soft);
}

[data-testid="stDataFrame"],
[data-testid="stJson"] {
    border-radius: 24px;
    overflow: hidden;
    box-shadow: var(--shadow-soft);
}
</style>""", unsafe_allow_html=True)
st.title("PointReg Lab")
st.markdown('<div class="subtitle">部分重合点云配准 · 低覆盖 GeoTransformer / 自研 ICP / FPFH</div>', unsafe_allow_html=True)

files = sorted(DATA.glob("*.ply"))
if not files:
    st.error(f"未找到点云数据：{DATA}")
    st.stop()
names = [p.stem for p in files]
poses = load_bunny_poses()

with st.sidebar:
    st.header("实验配置")
    source_name = st.selectbox("源点云", names, index=names.index("bun000") if "bun000" in names else 0, key="source")
    target_name = st.selectbox("目标点云", names, index=names.index("bun045") if "bun045" in names else min(1, len(names)-1), key="target")
    source, target = load_cloud(source_name), load_cloud(target_name)
    gt = None
    selected_overlap = None
    if source_name in poses and target_name in poses:
        gt = relative_transform(poses[source_name], poses[target_name])
        selected_overlap = symmetric_overlap(source, target, gt, OVERLAP_DISTANCE)
    is_low_overlap = selected_overlap is not None and selected_overlap < LOW_OVERLAP_THRESHOLD
    if selected_overlap is not None:
        st.caption(f"当前点对重合率：{selected_overlap:.3f}")
    if is_low_overlap:
        st.warning(
            f"检测到低重合点对（<{LOW_OVERLAP_THRESHOLD:.2f}）。"
            "建议使用低覆盖 GeoTransformer；单次运行约需 2–4 分钟。"
        )
        if st.button("✨ 使用推荐的低覆盖 GeoTransformer", use_container_width=True):
            st.session_state.coarse = LOW_OVERLAP_METHOD
            st.rerun()
    elif st.session_state.get("coarse") == LOW_OVERLAP_METHOD:
        st.session_state.coarse = "fpfh"

    coarse_options = ["fpfh", "fpfh_multiscale", "gcransac", "geotransformer", "pca", "none"]
    if is_low_overlap:
        coarse_options.insert(0, LOW_OVERLAP_METHOD)
    coarse_labels = {
        LOW_OVERLAP_METHOD: "低覆盖 GeoTransformer（Fusion，推荐）",
        "fpfh": "FPFH + RANSAC",
        "fpfh_multiscale": "多尺度 FPFH + 几何筛选",
        "gcransac": "FPFH + GC-RANSAC",
        "geotransformer": "GeoTransformer (ModelNet)",
        "pca": "PCA 主轴",
        "none": "无",
    }
    coarse = st.selectbox("粗配准", coarse_options, format_func=coarse_labels.get, key="coarse")
    fine = st.selectbox("精配准", ["custom_icp", "point_to_plane"], format_func={"custom_icp":"自研 Point-to-Point ICP", "point_to_plane":"Open3D Point-to-Plane"}.get, key="fine")
    voxel = st.number_input("体素尺寸", min_value=0.0001, max_value=0.02, value=0.0025, step=0.0005, format="%.4f", key="voxel")
    distance = st.number_input("最大对应距离", min_value=0.0005, max_value=0.1, value=0.01, step=0.001, format="%.4f", key="distance")
    trim = st.slider("保留对应比例", .2, 1.0, .8, .05, key="trim")
    iterations = st.slider("最大迭代次数", 5, 200, 60, 5, key="iterations")
    run = st.button("▶ 运行配准", type="primary", use_container_width=True, disabled=source_name == target_name)
    if st.button("恢复默认值", use_container_width=True):
        for key in ["source","target","coarse","fine","voxel","distance","trim","iterations","result","selection","fusion_details"]:
            st.session_state.pop(key, None)
        st.rerun()

def cloud_figure(source: np.ndarray, target: np.ndarray, transform: np.ndarray | None, title: str) -> go.Figure:
    limit = 10000
    s = source[::max(1, len(source)//limit)]
    t = target[::max(1, len(target)//limit)]
    if transform is not None:
        s = apply_transform(s, transform)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=t[:,0], y=t[:,1], z=t[:,2], mode="markers", name="Target", marker=dict(size=1.5,color="#4da3ff",opacity=.75)))
    fig.add_trace(go.Scatter3d(x=s[:,0], y=s[:,1], z=s[:,2], mode="markers", name="Source", marker=dict(size=1.5,color="#ff6b5f",opacity=.72)))
    fig.update_layout(title=title, height=510, margin=dict(l=0,r=0,t=38,b=0), paper_bgcolor="#FDFCF8", plot_bgcolor="#FDFCF8", font_color="#2C2C24",
                      legend=dict(font=dict(color="#2C2C24"), bgcolor="rgba(254,254,250,.82)", bordercolor="#DED8CF", borderwidth=1),
                      scene=dict(aspectmode="data",
                                 xaxis=dict(showbackground=True, backgroundcolor="#F8F3EB", color="#4A4A40", gridcolor="#DED8CF", zerolinecolor="#C18C5D"),
                                 yaxis=dict(showbackground=True, backgroundcolor="#F8F3EB", color="#4A4A40", gridcolor="#DED8CF", zerolinecolor="#C18C5D"),
                                 zaxis=dict(showbackground=True, backgroundcolor="#F8F3EB", color="#4A4A40", gridcolor="#DED8CF", zerolinecolor="#C18C5D")))
    return fig

if run:
    config = RegistrationConfig(
        coarse_method="none" if coarse == LOW_OVERLAP_METHOD else coarse,
        fine_method=fine,
        voxel_size=voxel,
        max_correspondence_distance=distance,
        trim_fraction=trim,
        max_iterations=iterations,
        adaptive_trim=coarse == LOW_OVERLAP_METHOD,
        success_rotation_deg=5.0,
        success_translation_ratio=PRACTICAL_TRANSLATION_RATIO,
    )
    st.session_state.overlap = selected_overlap
    if coarse == LOW_OVERLAP_METHOD:
        status = st.status("正在启动低覆盖 GeoTransformer…", expanded=True)
        progress_line = st.empty()

        def update_progress(message: str) -> None:
            progress_line.write(message)

        result, fusion_details = register_low_overlap_pair(
            source,
            target,
            config,
            ground_truth=gt,
            progress=update_progress,
        )
        st.session_state.result = result
        st.session_state.fusion_details = fusion_details
        status.update(
            label="低覆盖 GeoTransformer 配准完成" if result.status != "failed" else "低覆盖 GeoTransformer 配准失败",
            state="complete" if result.status != "failed" else "error",
            expanded=False,
        )
    else:
        with st.spinner("正在计算原有粗配准与 ICP…"):
            st.session_state.result = register_pair(source, target, config, ground_truth=gt)
            st.session_state.fusion_details = None
    st.session_state.selection = (source_name, target_name, coarse)

result = st.session_state.get("result")
if result and st.session_state.get("selection") == (source_name, target_name, coarse):
    cols = st.columns(6)
    values = [("状态", result.status), ("Fitness", f"{result.metrics.get('fitness',0):.3f}"), ("RMSE", f"{result.metrics.get('rmse',float('nan')):.6f}"),
              ("旋转误差", f"{result.metrics.get('rotation_error_deg',float('nan')):.2f}°"),
              ("平移误差比", f"{100 * result.metrics.get('translation_error_ratio',float('nan')):.2f}%"),
              ("总耗时", f"{result.timings_ms.get('total',0)/1000:.1f} s")]
    for col, (label, value) in zip(cols, values): col.metric(label, value)
    st.caption(result.message)
    overlap = st.session_state.get("overlap")
    if overlap is not None:
        st.caption(f"真值重叠率估计：{overlap:.3f}（仅用于实验分析，不参与求解）")
        if overlap < LOW_OVERLAP_THRESHOLD:
            if coarse == LOW_OVERLAP_METHOD:
                st.info("已使用低覆盖 GeoTransformer：Geo 对应 + FFT 全局搜索 + 自由空间选优。真值仅用于结果评测。")
            else:
                st.warning("这是低重合困难点对。建议在左侧切换为“低覆盖 GeoTransformer（Fusion）”后重新运行。")
        elif overlap < 0.5:
            st.info("该组合属于中低重合点对，默认保留原有配准方法。")
    if not result.success:
        st.warning("当前结果未通过旋转 <5°、平移比 <5% 的实用成功标准。")
    left, right = st.columns(2)
    left.plotly_chart(cloud_figure(source, target, None, "配准前"), use_container_width=True)
    right.plotly_chart(cloud_figure(source, target, result.transformation, "配准后"), use_container_width=True)
    tab1, tab2, tab3 = st.tabs(["收敛过程", "变换矩阵", "导出与验证"])
    with tab1:
        history = pd.DataFrame([asdict(item) for item in result.history])
        if len(history):
            metric_col, note_col = st.columns([1, 3])
            metric_col.metric("累计 ICP 迭代", len(history))
            stages = " → ".join(history["stage"].drop_duplicates().tolist())
            note_col.caption(f"迭代阶段：{stages}")
            st.line_chart(history.set_index("iteration")[["rmse"]])
            st.dataframe(history[["iteration", "stage", "rmse", "correspondences", "rotation_delta_deg", "translation_delta", "elapsed_ms"]],
                         use_container_width=True, hide_index=True,
                         column_config={"elapsed_ms": st.column_config.NumberColumn("elapsed_ms", format="%.2f ms")})
        else: st.info("当前算法未返回逐轮历史。")
    with tab2:
        st.dataframe(pd.DataFrame(result.transformation, columns=["c0","c1","c2","c3"]), use_container_width=True)
        formatted_timings = {name: f"{value:.2f} ms" for name, value in result.timings_ms.items()}
        st.json({"metrics":result.metrics,"timings_ms":formatted_timings,"message":result.message})
        fusion_details = st.session_state.get("fusion_details")
        if fusion_details and "search" in fusion_details:
            search = fusion_details["search"]
            st.subheader("低覆盖 GeoTransformer 选优摘要")
            st.json({
                "对应数量": fusion_details.get("correspondence_count"),
                "最终候选": search.get("selected_label"),
                "自由空间门控": search.get("gate_passed"),
                "fitness": search.get("fitness"),
                "violation": search.get("violation"),
                "阶段耗时_ms": search.get("timings_ms"),
                "候选生成错误": fusion_details.get("seed_errors", {}),
            })
    with tab3:
        out = ROOT / "outputs" / "ui" / f"{source_name}_to_{target_name}"
        if st.button("导出 CloudCompare 文件"):
            exported = export_cloudcompare(out, source, target, result.transformation, result.to_dict())
            st.success(f"已导出到 {out}")
            st.session_state.exported = exported
        if st.button("尝试在 CloudCompare 打开"):
            exported = st.session_state.get("exported") or export_cloudcompare(out, source, target, result.transformation, result.to_dict())
            ok, message = launch_cloudcompare([exported["target"], exported["aligned"]])
            (st.success if ok else st.warning)(message)
else:
    st.plotly_chart(cloud_figure(source, target, None, "原始点云预览"), use_container_width=True)
    st.info("在左侧确认参数后点击“运行配准”。")
