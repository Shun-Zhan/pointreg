# PointReg Lab —— 基于 Streamlit 的点云配准交互式演示 Web 应用。
# 左侧栏选择源/目标点云与算法参数，主区展示配准前后的三维点云、
# 指标卡片、收敛曲线、变换矩阵以及导出到 CloudCompare 的功能。
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go   # 三维散点可视化
import streamlit as st              # Web UI 框架

# 从 pointreg 包导入各功能模块：
from pointreg.cloudcompare import export_cloudcompare, launch_cloudcompare  # 导出/打开 CloudCompare
from pointreg.fusion import register_low_overlap_pair   # 低重叠专用融合配准（GeoTransformer + 全局搜索）
from pointreg.io import parse_bun_conf, read_points     # 读位姿 / 读点云
from pointreg.metrics import symmetric_overlap          # 估计两片点云的对称重叠率
from pointreg.models import RegistrationConfig          # 配准参数配置
from pointreg.pipeline import register_pair             # 常规粗+精配准流水线
from pointreg.runtime import preload_open3d             # 预热 Open3D，减少首次运行卡顿
from pointreg.transforms import apply_transform, relative_transform  # 施加变换 / 求相对真值

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "bunny" / "data"          # 兔子点云数据目录
LOW_OVERLAP_THRESHOLD = 0.50            # 低于该重叠率即判定为“低重叠困难点对”
OVERLAP_DISTANCE = 0.01                 # 计算重叠率时判定“对应点”的距离阈值
LOW_OVERLAP_METHOD = "low_overlap_geotransformer"  # 低重叠专用方法的内部标识
PRACTICAL_TRANSLATION_RATIO = 0.05      # 实用成功标准：平移误差比 < 5%
# 配置页面标题、图标与宽屏布局
st.set_page_config(page_title="PointReg Lab", page_icon="◌", layout="wide")


@st.cache_resource(show_spinner="正在预加载 Open3D…")
def warm_up_runtime() -> None:
    """预加载 Open3D 运行时。用 cache_resource 缓存，整个会话只执行一次。"""
    preload_open3d()


@st.cache_data(show_spinner=False)
def load_cloud(name: str) -> np.ndarray:
    """按名字读取点云为 numpy 数组；cache_data 缓存结果，重复选择同一片不再重复读盘。"""
    return read_points(DATA / f"{name}.ply")


@st.cache_data(show_spinner=False)
def load_bunny_poses() -> dict[str, np.ndarray]:
    """读取 bun.conf 里的真值位姿字典；文件不存在时返回空字典（此时无真值评测）。"""
    conf = DATA / "bun.conf"
    return parse_bun_conf(conf) if conf.exists() else {}


warm_up_runtime()  # 应用启动即预热 Open3D
# 注入自定义 CSS：定义配色变量、背景纹理、侧栏、指标卡、按钮、
# 选择框、标签页等控件的视觉样式，把默认 Streamlit 界面改造成
# 温暖的米色/橄榄绿主题（unsafe_allow_html=True 允许注入原始 HTML/CSS）。
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
st.title("PointReg Lab")  # 页面主标题
st.markdown('<div class="subtitle">部分重合点云配准 · 低覆盖 GeoTransformer / 自研 ICP / FPFH</div>', unsafe_allow_html=True)  # 副标题

# 扫描数据目录下所有 .ply 点云文件；若一个都没有则报错并停止渲染
files = sorted(DATA.glob("*.ply"))
if not files:
    st.error(f"未找到点云数据：{DATA}")
    st.stop()
names = [p.stem for p in files]  # 去掉扩展名，作为下拉框可选项
poses = load_bunny_poses()       # 真值位姿（可能为空）

# ---------- 左侧栏：实验参数配置 ----------
with st.sidebar:
    st.header("实验配置")
    # 源/目标点云选择框：默认尽量选 bun000 / bun045，找不到就退化到首个/次个
    source_name = st.selectbox("源点云", names, index=names.index("bun000") if "bun000" in names else 0, key="source")
    target_name = st.selectbox("目标点云", names, index=names.index("bun045") if "bun045" in names else min(1, len(names)-1), key="target")
    source, target = load_cloud(source_name), load_cloud(target_name)  # 加载所选两片点云
    gt = None
    selected_overlap = None
    # 若两帧都有真值位姿，则算出相对真值变换，并据此估计当前点对的重叠率
    if source_name in poses and target_name in poses:
        gt = relative_transform(poses[source_name], poses[target_name])
        selected_overlap = symmetric_overlap(source, target, gt, OVERLAP_DISTANCE)
    # 判定是否属于低重叠困难点对
    is_low_overlap = selected_overlap is not None and selected_overlap < LOW_OVERLAP_THRESHOLD
    if selected_overlap is not None:
        st.caption(f"当前点对重合率：{selected_overlap:.3f}")  # 展示重叠率供参考
    if is_low_overlap:
        # 低重叠时给出提示，并提供一键切换到推荐方法的按钮
        st.warning(
            f"检测到低重合点对（<{LOW_OVERLAP_THRESHOLD:.2f}）。"
            "建议使用低覆盖 GeoTransformer；单次运行约需 2–4 分钟。"
        )
        if st.button("✨ 使用推荐的低覆盖 GeoTransformer", use_container_width=True):
            # 写入 session_state 并 rerun，让下面的下拉框默认选中推荐方法
            st.session_state.coarse = LOW_OVERLAP_METHOD
            st.rerun()
    elif st.session_state.get("coarse") == LOW_OVERLAP_METHOD:
        # 若切到了非低重叠点对但残留着低覆盖方法的选择，则回退到普通 fpfh
        st.session_state.coarse = "fpfh"

    # 粗配准可选方法列表；只有当前是低重叠点对时，才把“低覆盖 GeoTransformer”置顶插入
    coarse_options = ["fpfh", "fpfh_multiscale", "gcransac", "geotransformer", "pca", "none"]
    if is_low_overlap:
        coarse_options.insert(0, LOW_OVERLAP_METHOD)
    # 下拉框显示的中文标签映射
    coarse_labels = {
        LOW_OVERLAP_METHOD: "低覆盖 GeoTransformer（Fusion，推荐）",
        "fpfh": "FPFH + RANSAC",
        "fpfh_multiscale": "多尺度 FPFH + 几何筛选",
        "gcransac": "FPFH + GC-RANSAC",
        "geotransformer": "GeoTransformer (ModelNet)",
        "pca": "PCA 主轴",
        "none": "无",
    }
    # 粗配准方法下拉框（用上面的中文标签映射来展示）
    coarse = st.selectbox("粗配准", coarse_options, format_func=coarse_labels.get, key="coarse")
    # 精配准方法：自研点到点 ICP 或 Open3D 点到面 ICP
    fine = st.selectbox("精配准", ["custom_icp", "point_to_plane"], format_func={"custom_icp":"自研 Point-to-Point ICP", "point_to_plane":"Open3D Point-to-Plane"}.get, key="fine")
    # 体素尺寸：下采样粒度，直接影响特征分辨率与速度
    voxel = st.number_input("体素尺寸", min_value=0.0001, max_value=0.02, value=0.0025, step=0.0005, format="%.4f", key="voxel")
    # 最大对应距离：ICP 里认定为有效对应点对的距离上限
    distance = st.number_input("最大对应距离", min_value=0.0005, max_value=0.1, value=0.01, step=0.001, format="%.4f", key="distance")
    # 保留对应比例（trimmed ICP）：每轮只用误差最小的这部分对应，抗离群点
    trim = st.slider("保留对应比例", .2, 1.0, .8, .05, key="trim")
    # ICP 最大迭代次数
    iterations = st.slider("最大迭代次数", 5, 200, 60, 5, key="iterations")
    # 主运行按钮；源与目标相同时禁用（无意义）
    run = st.button("▶ 运行配准", type="primary", use_container_width=True, disabled=source_name == target_name)
    if st.button("恢复默认值", use_container_width=True):
        # 清空相关 session_state 键并 rerun，使所有控件回到默认值
        for key in ["source","target","coarse","fine","voxel","distance","trim","iterations","result","selection","fusion_details"]:
            st.session_state.pop(key, None)
        st.rerun()

def cloud_figure(source: np.ndarray, target: np.ndarray, transform: np.ndarray | None, title: str) -> go.Figure:
    """构造源/目标点云的三维散点 Plotly 图。

    为了流畅渲染，先把点抽稀到约 limit 个；若给了 transform，则先把源点云
    施加该变换（用于展示“配准后”对齐效果）。目标蓝色、源红色。
    """
    limit = 10000  # 每片点云最多绘制的点数
    # 等间隔抽稀：按 len//limit 的步长取点，保证点数不超过 limit
    s = source[::max(1, len(source)//limit)]
    t = target[::max(1, len(target)//limit)]
    if transform is not None:
        s = apply_transform(s, transform)  # 展示配准后：把源点云对齐到目标坐标系
    fig = go.Figure()
    # 目标点云（蓝）与源点云（红）分别作为两条散点轨迹叠加显示
    fig.add_trace(go.Scatter3d(x=t[:,0], y=t[:,1], z=t[:,2], mode="markers", name="Target", marker=dict(size=1.5,color="#4da3ff",opacity=.75)))
    fig.add_trace(go.Scatter3d(x=s[:,0], y=s[:,1], z=s[:,2], mode="markers", name="Source", marker=dict(size=1.5,color="#ff6b5f",opacity=.72)))
    # 统一图表布局与配色（与页面主题保持一致），aspectmode="data" 保证 xyz 等比例不变形
    fig.update_layout(title=title, height=510, margin=dict(l=0,r=0,t=38,b=0), paper_bgcolor="#FDFCF8", plot_bgcolor="#FDFCF8", font_color="#2C2C24",
                      legend=dict(font=dict(color="#2C2C24"), bgcolor="rgba(254,254,250,.82)", bordercolor="#DED8CF", borderwidth=1),
                      scene=dict(aspectmode="data",
                                 xaxis=dict(showbackground=True, backgroundcolor="#F8F3EB", color="#4A4A40", gridcolor="#DED8CF", zerolinecolor="#C18C5D"),
                                 yaxis=dict(showbackground=True, backgroundcolor="#F8F3EB", color="#4A4A40", gridcolor="#DED8CF", zerolinecolor="#C18C5D"),
                                 zaxis=dict(showbackground=True, backgroundcolor="#F8F3EB", color="#4A4A40", gridcolor="#DED8CF", zerolinecolor="#C18C5D")))
    return fig

# ---------- 点击“运行配准”后的执行分支 ----------
if run:
    # 根据侧栏参数组装配准配置对象。
    # 注意：低覆盖方法在流水线里粗配准记为 "none"（因为它有独立求解路径），
    # 并开启 adaptive_trim；成功判据设为旋转<5°、平移比<5%。
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
    st.session_state.overlap = selected_overlap  # 记住本次重叠率供结果区展示
    if coarse == LOW_OVERLAP_METHOD:
        # 低覆盖 GeoTransformer 分支：耗时较长，用 status 容器 + 进度行实时反馈
        status = st.status("正在启动低覆盖 GeoTransformer…", expanded=True)
        progress_line = st.empty()  # 占位组件，用于滚动更新进度文字

        def update_progress(message: str) -> None:
            """进度回调：把融合配准内部各阶段的消息写到进度行。"""
            progress_line.write(message)

        # 调用低重叠融合配准（Geo 对应 + FFT 全局搜索 + 自由空间选优），
        # 返回配准结果与包含选优摘要的 fusion_details。
        result, fusion_details = register_low_overlap_pair(
            source,
            target,
            config,
            ground_truth=gt,
            progress=update_progress,
        )
        st.session_state.result = result
        st.session_state.fusion_details = fusion_details
        # 根据结果状态更新 status 容器的标签与完成/错误状态
        status.update(
            label="低覆盖 GeoTransformer 配准完成" if result.status != "failed" else "低覆盖 GeoTransformer 配准失败",
            state="complete" if result.status != "failed" else "error",
            expanded=False,
        )
    else:
        # 常规分支：走普通粗+精配准流水线，用 spinner 提示计算中
        with st.spinner("正在计算原有粗配准与 ICP…"):
            st.session_state.result = register_pair(source, target, config, ground_truth=gt)
            st.session_state.fusion_details = None  # 常规分支没有融合选优摘要
    # 记录本次运行对应的选择（源、目标、粗配准方法），用于后面校验结果与当前选择是否一致
    st.session_state.selection = (source_name, target_name, coarse)

# ---------- 结果展示区 ----------
# 只有当已有结果、且该结果正对应当前的(源,目标,粗配准)选择时才展示，
# 避免切换参数后仍显示旧结果。
result = st.session_state.get("result")
if result and st.session_state.get("selection") == (source_name, target_name, coarse):
    # 六列指标卡片：状态、拟合度、RMSE、旋转误差、平移误差比、总耗时
    cols = st.columns(6)
    values = [("状态", result.status), ("Fitness", f"{result.metrics.get('fitness',0):.3f}"), ("RMSE", f"{result.metrics.get('rmse',float('nan')):.6f}"),
              ("旋转误差", f"{result.metrics.get('rotation_error_deg',float('nan')):.2f}°"),
              ("平移误差比", f"{100 * result.metrics.get('translation_error_ratio',float('nan')):.2f}%"),
              ("总耗时", f"{result.timings_ms.get('total',0)/1000:.1f} s")]
    for col, (label, value) in zip(cols, values): col.metric(label, value)  # 逐列渲染 metric
    st.caption(result.message)  # 结果附带的文字说明
    overlap = st.session_state.get("overlap")
    if overlap is not None:
        # 展示真值重叠率，并强调它只用于评测，不参与求解
        st.caption(f"真值重叠率估计：{overlap:.3f}（仅用于实验分析，不参与求解）")
        if overlap < LOW_OVERLAP_THRESHOLD:
            if coarse == LOW_OVERLAP_METHOD:
                st.info("已使用低覆盖 GeoTransformer：Geo 对应 + FFT 全局搜索 + 自由空间选优。真值仅用于结果评测。")
            else:
                # 低重叠却没用推荐方法时，提示切换
                st.warning("这是低重合困难点对。建议在左侧切换为“低覆盖 GeoTransformer（Fusion）”后重新运行。")
        elif overlap < 0.5:
            st.info("该组合属于中低重合点对，默认保留原有配准方法。")
    if not result.success:
        # 未达实用成功标准时给出警告
        st.warning("当前结果未通过旋转 <5°、平移比 <5% 的实用成功标准。")
    # 左右并排展示“配准前”（源未变换）与“配准后”（源施加结果变换）
    left, right = st.columns(2)
    left.plotly_chart(cloud_figure(source, target, None, "配准前"), use_container_width=True)
    right.plotly_chart(cloud_figure(source, target, result.transformation, "配准后"), use_container_width=True)
    # 三个标签页：收敛过程 / 变换矩阵 / 导出与验证
    tab1, tab2, tab3 = st.tabs(["收敛过程", "变换矩阵", "导出与验证"])
    with tab1:  # —— 收敛过程：展示 ICP 逐轮迭代的 RMSE 曲线与明细表 ——
        # 把结果里的迭代历史（dataclass 列表）转成 DataFrame
        history = pd.DataFrame([asdict(item) for item in result.history])
        if len(history):
            metric_col, note_col = st.columns([1, 3])
            metric_col.metric("累计 ICP 迭代", len(history))  # 总迭代轮数
            # 列出经历过的各阶段名称（去重后用箭头连接）
            stages = " → ".join(history["stage"].drop_duplicates().tolist())
            note_col.caption(f"迭代阶段：{stages}")
            st.line_chart(history.set_index("iteration")[["rmse"]])  # RMSE 随迭代下降曲线
            # 明细表：每轮的阶段、RMSE、对应数、旋转/平移增量、耗时
            st.dataframe(history[["iteration", "stage", "rmse", "correspondences", "rotation_delta_deg", "translation_delta", "elapsed_ms"]],
                         use_container_width=True, hide_index=True,
                         column_config={"elapsed_ms": st.column_config.NumberColumn("elapsed_ms", format="%.2f ms")})
        else: st.info("当前算法未返回逐轮历史。")  # 例如某些粗配准无迭代历史
    with tab2:  # —— 变换矩阵：展示 4x4 结果矩阵、指标/耗时 JSON、以及融合选优摘要 ——
        # 4x4 刚体变换矩阵（列名 c0..c3）
        st.dataframe(pd.DataFrame(result.transformation, columns=["c0","c1","c2","c3"]), use_container_width=True)
        # 把耗时格式化为带单位的字符串，连同指标与消息一起以 JSON 展示
        formatted_timings = {name: f"{value:.2f} ms" for name, value in result.timings_ms.items()}
        st.json({"metrics":result.metrics,"timings_ms":formatted_timings,"message":result.message})
        fusion_details = st.session_state.get("fusion_details")
        # 若是低覆盖方法且带有全局搜索信息，额外展示其选优过程摘要
        if fusion_details and "search" in fusion_details:
            search = fusion_details["search"]
            st.subheader("低覆盖 GeoTransformer 选优摘要")
            st.json({
                "对应数量": fusion_details.get("correspondence_count"),
                "最终候选": search.get("selected_label"),      # 最终选中的候选位姿标签
                "自由空间门控": search.get("gate_passed"),      # 是否通过自由空间一致性门控
                "fitness": search.get("fitness"),
                "violation": search.get("violation"),          # 自由空间冲突量
                "阶段耗时_ms": search.get("timings_ms"),
                "候选生成错误": fusion_details.get("seed_errors", {}),  # 候选生成阶段的报错记录
            })
    with tab3:  # —— 导出与验证：把结果导出为 CloudCompare 文件，或直接在 CloudCompare 打开 ——
        out = ROOT / "outputs" / "ui" / f"{source_name}_to_{target_name}"  # 导出目录
        if st.button("导出 CloudCompare 文件"):
            # 导出目标、源、对齐后点云及结果元数据，缓存导出路径供下一个按钮复用
            exported = export_cloudcompare(out, source, target, result.transformation, result.to_dict())
            st.success(f"已导出到 {out}")
            st.session_state.exported = exported
        if st.button("尝试在 CloudCompare 打开"):
            # 若之前没导出过就先导出，再尝试启动 CloudCompare 打开目标与对齐后点云
            exported = st.session_state.get("exported") or export_cloudcompare(out, source, target, result.transformation, result.to_dict())
            ok, message = launch_cloudcompare([exported["target"], exported["aligned"]])
            (st.success if ok else st.warning)(message)  # 按成败选用 success/warning 提示
else:
    # 尚未运行（或结果与当前选择不匹配）时，仅预览原始点云并提示操作
    st.plotly_chart(cloud_figure(source, target, None, "原始点云预览"), use_container_width=True)
    st.info("在左侧确认参数后点击“运行配准”。")
