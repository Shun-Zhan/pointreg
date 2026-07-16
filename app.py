"""PointReg Lab —— 基于 Streamlit 的点云配准交互界面。

提供选点对、调参数、运行直接/桥接配准，并用 Plotly 3D 可视化配准前后效果、
展示收敛过程与变换矩阵、导出到 CloudCompare。核心算法都在 pointreg 包内，
本文件只负责界面与调度。
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pointreg.cloudcompare import export_cloudcompare, launch_cloudcompare
from pointreg.io import parse_bun_conf, read_points
from pointreg.metrics import symmetric_overlap
from pointreg.models import RegistrationConfig
from pointreg.pipeline import register_pair
from pointreg.dataset import build_bunny_graph, register_dataset_pair
from pointreg.runtime import preload_open3d
from pointreg.transforms import apply_transform, relative_transform

ROOT = Path(__file__).resolve().parent   # 项目根目录
DATA = ROOT / "bunny" / "data"           # Bunny 点云数据目录
OVERLAP_DISTANCE = 0.01                   # 计算重合率时判定对应点的距离阈值
st.set_page_config(page_title="PointReg Lab", page_icon="◌", layout="wide")


@st.cache_resource(show_spinner="正在预加载 Open3D 并构建稳健配准缓存…")
def warm_up_runtime() -> None:
    """应用启动时预热：加载 Open3D 并预构建桥接图，避免首次运行时卡顿。

    用 @st.cache_resource 保证整个会话只执行一次（结果被缓存复用）。
    """
    preload_open3d()
    build_bunny_graph(str(DATA.resolve()), .0025, .01, .8, 60, 42)


warm_up_runtime()
# —— 自定义页面样式（CSS）：统一配色、圆角、按钮/控件外观，纯视觉无逻辑 ——
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

[data-testid="stNumberInputContainer"] {
    background: rgba(255, 255, 255, .55) !important;
    border: 1px solid var(--border) !important;
    border-radius: 999px !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, .72);
    overflow: hidden;
}
[data-testid="stNumberInputContainer"] [data-baseweb="input"],
[data-testid="stNumberInputContainer"] [data-baseweb="base-input"],
[data-testid="stNumberInputField"] {
    background: transparent !important;
    border: 0 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    color: var(--foreground) !important;
    font-weight: 700;
}
[data-testid="stNumberInputStepDown"],
[data-testid="stNumberInputStepUp"] {
    background: transparent !important;
    border: 0 !important;
    border-left: 1px solid var(--border) !important;
    color: var(--primary) !important;
}
[data-testid="stNumberInputStepDown"]:hover,
[data-testid="stNumberInputStepUp"]:hover {
    background: rgba(93, 112, 82, .10) !important;
    color: #4E6246 !important;
}
[data-testid="stNumberInputContainer"]:focus-within {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(93, 112, 82, .18) !important;
}

[data-testid="stCheckbox"] [data-baseweb="checkbox"] > div:first-child {
    background-color: var(--accent) !important;
    border: 1px solid var(--border) !important;
    box-shadow: inset 0 1px 2px rgba(44, 44, 36, .10);
}
[data-testid="stCheckbox"] [data-baseweb="checkbox"] > div:first-child > div {
    background-color: var(--primary) !important;
}
[data-testid="stCheckbox"] [data-baseweb="checkbox"]:has(input[aria-checked="true"]) > div:first-child {
    background-color: var(--primary) !important;
    border-color: var(--primary) !important;
}
[data-testid="stCheckbox"] [data-baseweb="checkbox"]:has(input[aria-checked="true"]) > div:first-child > div {
    background-color: var(--primary-foreground) !important;
}
[data-testid="stCheckbox"] [data-testid="stTooltipIcon"] {
    color: var(--muted-foreground) !important;
}
[data-testid="stCheckbox"] [data-testid="stTooltipIcon"] svg {
    stroke: currentColor !important;
}
[data-testid="stCheckbox"] [data-baseweb="checkbox"]:has(input:disabled) {
    opacity: .55;
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
st.markdown('<div class="subtitle">部分重合点云配准 · ICP / FPFH / CloudCompare</div>', unsafe_allow_html=True)

# 扫描数据目录下所有 PLY，作为可选点云；缺数据则直接停下并报错
files = sorted(DATA.glob("*.ply"))
if not files:
    st.error(f"未找到点云数据：{DATA}")
    st.stop()
names = [p.stem for p in files]
poses = parse_bun_conf(DATA / "bun.conf") if (DATA / "bun.conf").exists() else {}

# —— 左侧边栏：实验配置（选点对、调参数、选策略）——
with st.sidebar:
    st.header("实验配置")
    # 源/目标点云默认选 bun000→bun045（若存在）
    source_name = st.selectbox("源点云", names, index=names.index("bun000") if "bun000" in names else 0, key="source")
    target_name = st.selectbox("目标点云", names, index=names.index("bun045") if "bun045" in names else min(1, len(names)-1), key="target")
    source_path, target_path = DATA / f"{source_name}.ply", DATA / f"{target_name}.ply"
    source, target = read_points(source_path), read_points(target_path)
    gt = None
    pair_overlap = None
    # 有真值时展示该点对的重合率（仅供参考，不参与求解）
    if source_name in poses and target_name in poses:
        gt = relative_transform(poses[source_name], poses[target_name])
        pair_overlap = symmetric_overlap(source, target, gt, OVERLAP_DISTANCE)
        st.caption(f"当前点对重合率：{pair_overlap:.3f}")
        st.caption("基于真值位姿，仅用于实验展示，不参与配准求解。")
    else:
        st.caption("当前点对重合率：不可用")
    # 算法与参数选择控件
    coarse = st.selectbox("粗配准", ["fpfh", "pca", "none"], format_func={"fpfh":"FPFH + RANSAC", "pca":"PCA 主轴", "none":"无"}.get, key="coarse")
    fine = st.selectbox("精配准", ["custom_icp", "point_to_plane"], format_func={"custom_icp":"Point-to-Point ICP", "point_to_plane":"Open3D Point-to-Plane"}.get, key="fine")
    voxel = st.number_input("体素尺寸", min_value=0.0001, max_value=0.02, value=0.0025, step=0.0005, format="%.4f", key="voxel")
    distance = st.number_input("最大对应距离", min_value=0.0005, max_value=0.1, value=0.01, step=0.001, format="%.4f", key="distance")
    trim = st.slider("保留对应比例", .2, 1.0, .8, .05, key="trim")
    iterations = st.slider("最大迭代次数", 5, 200, 60, 5, key="iterations")
    # 桥接法开关：仅在 FPFH 粗配下可用，其他粗配方法时禁用该开关
    bridge_help = "关闭时执行严格两帧直接配准；开启时沿高重合点云路径组合变换。"
    if coarse == "fpfh":
        use_bridge = st.toggle("使用桥接法", value=False, key="use_bridge", help=bridge_help)
    else:
        st.toggle("使用桥接法", value=False, key="use_bridge_disabled", disabled=True, help="桥接法仅适用于 FPFH 粗配准。")
        use_bridge = False
    bridge_enabled = coarse == "fpfh" and use_bridge
    strategy = "bridge" if bridge_enabled else "direct"
    run = st.button("▶ 运行配准", type="primary", use_container_width=True)
    # 恢复默认：清空相关 session_state 键并重跑页面
    if st.button("恢复默认值", use_container_width=True):
        for key in ["source","target","coarse","fine","voxel","distance","trim","iterations","use_bridge","use_bridge_disabled","result","selection","result_overlap"]:
            st.session_state.pop(key, None)
        st.rerun()

def cloud_figure(source: np.ndarray, target: np.ndarray, transform: np.ndarray | None, title: str) -> go.Figure:
    """构造源(红)/目标(蓝)两片点云的 Plotly 3D 散点图；给定 transform 则先变换源点云。"""
    # 下采样到至多约 1 万点，保证前端渲染流畅
    limit = 10000
    s = source[::max(1, len(source)//limit)]
    t = target[::max(1, len(target)//limit)]
    if transform is not None:
        s = apply_transform(s, transform)  # 展示“配准后”时对源点云施加变换
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

# 用当前参数组合作为“签名”，判断已有结果是否与当前选择一致
run_signature = (source_name, target_name, coarse, fine, voxel, distance, trim, iterations, strategy)
if run:
    config = RegistrationConfig(coarse_method=coarse, fine_method=fine, voxel_size=voxel, max_correspondence_distance=distance, trim_fraction=trim, max_iterations=iterations)
    with st.spinner("正在计算粗配准与 ICP…"):
        # 按策略选择桥接法或直接两帧配准，结果存入 session_state 以便重绘复用
        if bridge_enabled:
            st.session_state.result = register_dataset_pair(DATA, source_name, target_name, config)
        else:
            st.session_state.result = register_pair(source, target, config, ground_truth=gt)
        st.session_state.selection = run_signature
        st.session_state.result_overlap = pair_overlap

result = st.session_state.get("result")
# 仅当缓存结果对应的正是当前参数签名时才展示，避免改参后显示过时结果
if result and st.session_state.get("selection") == run_signature:
    # 顶部三个上下文指标：重合率、所用策略、成功/失败判定
    context_cols = st.columns(3)
    result_overlap = st.session_state.get("result_overlap")
    context_cols[0].metric("点对重合率", f"{result_overlap:.3f}" if result_overlap is not None else "不可用")
    context_cols[1].metric("配准策略", "桥接法" if bridge_enabled else "直接两帧")
    context_cols[2].metric("实验判定", "成功" if result.success else "失败")
    # 五个核心量化指标
    cols = st.columns(5)
    values = [("状态", result.status), ("Fitness", f"{result.metrics.get('fitness',0):.3f}"), ("RMSE", f"{result.metrics.get('rmse',float('nan')):.6f}"),
              ("旋转误差", f"{result.metrics.get('rotation_error_deg',float('nan')):.2f}°"), ("总耗时", f"{result.timings_ms.get('total',0):.2f} ms")]
    for col, (label, value) in zip(cols, values): col.metric(label, value)
    st.caption(result.message)
    # 左右并排展示配准前/后的 3D 点云
    left, right = st.columns(2)
    left.plotly_chart(cloud_figure(source, target, None, "配准前"), use_container_width=True)
    right.plotly_chart(cloud_figure(source, target, result.transformation, "配准后"), use_container_width=True)
    # 把逐轮 ICP 历史转成 DataFrame 供后续图表使用
    history = pd.DataFrame([asdict(item) for item in result.history])
    # 结果分多个标签页展示；桥接法额外多一个“桥接过程”页
    tab_names = ["收敛过程", "变换矩阵", "导出与验证"]
    if bridge_enabled:
        tab_names.append("桥接过程")
    tabs = st.tabs(tab_names)
    tab1, tab2, tab3 = tabs[:3]
    bridge_tab = tabs[3] if bridge_enabled else None
    # —— 标签页 1：收敛过程（RMSE 曲线 + 逐轮明细表）——
    with tab1:
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
    # —— 标签页 2：变换矩阵与指标/耗时 JSON ——
    with tab2:
        st.dataframe(pd.DataFrame(result.transformation, columns=["c0","c1","c2","c3"]), use_container_width=True)
        formatted_timings = {name: f"{value:.2f} ms" for name, value in result.timings_ms.items()}
        st.json({"metrics":result.metrics,"timings_ms":formatted_timings,"message":result.message})
    # —— 标签页 3：导出结果并可尝试用 CloudCompare 打开验证 ——
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
    # —— 标签页 4（仅桥接法）：展示桥接路径、逐跳配准明细与单跳可视化 ——
    if bridge_tab is not None:
        with bridge_tab:
            # 复用缓存的配准图，取出本次点对的桥接路径与途经各边
            graph = build_bunny_graph(str(DATA.resolve()), voxel, distance, trim, iterations, 42)
            _, bridge_path = graph.transform(source_name, target_name)
            bridge_edges = list(zip(bridge_path, bridge_path[1:]))
            stage_labels = [f"{edge_source}→{edge_target}" for edge_source, edge_target in bridge_edges]

            path_col, hops_col = st.columns([4, 1])
            path_col.markdown(f"**桥接路径：** {' → '.join(bridge_path)}")
            hops_col.metric("桥接跳数", len(bridge_edges))

            # 汇总每一跳的 ICP 迭代数、最终 RMSE、对应点数与耗时
            summary_rows = []
            for step, stage in enumerate(stage_labels, start=1):
                stage_history = history[history["stage"] == stage] if len(history) else pd.DataFrame()
                final_record = stage_history.iloc[-1] if len(stage_history) else None
                summary_rows.append({
                    "步骤": step,
                    "桥接边": stage,
                    "ICP迭代数": len(stage_history),
                    "最终RMSE": final_record["rmse"] if final_record is not None else float("nan"),
                    "最终对应点数": int(final_record["correspondences"]) if final_record is not None else 0,
                    "ICP耗时_ms": stage_history["elapsed_ms"].sum() if len(stage_history) else 0.0,
                })
            st.dataframe(
                pd.DataFrame(summary_rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "最终RMSE": st.column_config.NumberColumn(format="%.6f"),
                    "ICP耗时_ms": st.column_config.NumberColumn(format="%.2f ms"),
                },
            )

            # 选择某一跳，单独查看该边的配准前后与收敛曲线
            selected_stage = st.selectbox(
                "查看桥接步骤",
                stage_labels,
                key=f"bridge_step_{source_name}_{target_name}",
            )
            selected_index = stage_labels.index(selected_stage)
            step_source_name, step_target_name = bridge_edges[selected_index]
            step_source = read_points(DATA / f"{step_source_name}.ply")
            step_target = read_points(DATA / f"{step_target_name}.ply")
            step_transform = graph.adjacency[step_source_name][step_target_name]  # 该边已估计好的变换

            before_col, after_col = st.columns(2)
            before_col.plotly_chart(
                cloud_figure(step_source, step_target, None, f"第 {selected_index + 1} 跳配准前：{selected_stage}"),
                use_container_width=True,
            )
            after_col.plotly_chart(
                cloud_figure(step_source, step_target, step_transform, f"第 {selected_index + 1} 跳配准后：{selected_stage}"),
                use_container_width=True,
            )

            # 该跳对应的逐轮 RMSE 曲线
            selected_history = history[history["stage"] == selected_stage] if len(history) else pd.DataFrame()
            if len(selected_history):
                st.subheader(f"{selected_stage} 收敛过程")
                st.line_chart(selected_history.set_index("iteration")[["rmse"]])
            else:
                st.info("该桥接步骤没有返回逐轮 ICP 历史。")
else:
    # 尚未运行（或参数已改）时：仅预览原始点云并提示操作
    st.plotly_chart(cloud_figure(source, target, None, "原始点云预览"), use_container_width=True)
    st.info("在左侧确认参数后点击“运行配准”。")
