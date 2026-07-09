from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pointreg.cloudcompare import export_cloudcompare, launch_cloudcompare
from pointreg.io import parse_bun_conf, read_points
from pointreg.models import RegistrationConfig
from pointreg.pipeline import register_pair
from pointreg.runtime import preload_open3d
from pointreg.transforms import apply_transform, relative_transform

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "bunny" / "data"
st.set_page_config(page_title="PointReg Lab", page_icon="◌", layout="wide")


@st.cache_resource(show_spinner="正在预加载 Open3D…")
def warm_up_runtime() -> None:
    preload_open3d()


warm_up_runtime()
st.markdown("""<style>
.stApp {background: radial-gradient(circle at 15% 10%, #102b3d 0, #07131d 38%, #050b11 100%);}
[data-testid="stMetric"] {background:#0b1c28;border:1px solid #17384b;padding:14px;border-radius:12px;}
[data-testid="stMetricValue"] {font-size:1.45rem; white-space:nowrap;}
h1,h2,h3 {letter-spacing:-.02em;} .subtitle{color:#86a7b9;margin-top:-16px;margin-bottom:22px}
</style>""", unsafe_allow_html=True)
st.title("PointReg Lab")
st.markdown('<div class="subtitle">部分重合点云配准 · 自研 ICP / FPFH / CloudCompare</div>', unsafe_allow_html=True)

files = sorted(DATA.glob("*.ply"))
if not files:
    st.error(f"未找到点云数据：{DATA}")
    st.stop()
names = [p.stem for p in files]

with st.sidebar:
    st.header("实验配置")
    source_name = st.selectbox("源点云", names, index=names.index("bun000") if "bun000" in names else 0, key="source")
    target_name = st.selectbox("目标点云", names, index=names.index("bun045") if "bun045" in names else min(1, len(names)-1), key="target")
    coarse = st.selectbox("粗配准", ["fpfh", "pca", "none"], format_func={"fpfh":"FPFH + RANSAC", "pca":"PCA 主轴", "none":"无"}.get, key="coarse")
    fine = st.selectbox("精配准", ["custom_icp", "point_to_plane"], format_func={"custom_icp":"自研 Point-to-Point ICP", "point_to_plane":"Open3D Point-to-Plane"}.get, key="fine")
    voxel = st.number_input("体素尺寸", min_value=0.0001, max_value=0.02, value=0.0025, step=0.0005, format="%.4f", key="voxel")
    distance = st.number_input("最大对应距离", min_value=0.0005, max_value=0.1, value=0.01, step=0.001, format="%.4f", key="distance")
    trim = st.slider("保留对应比例", .2, 1.0, .8, .05, key="trim")
    iterations = st.slider("最大迭代次数", 5, 200, 60, 5, key="iterations")
    run = st.button("▶ 运行配准", type="primary", use_container_width=True)
    if st.button("恢复默认值", use_container_width=True):
        for key in ["source","target","coarse","fine","voxel","distance","trim","iterations","result","selection"]:
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
    fig.update_layout(title=title, height=510, margin=dict(l=0,r=0,t=38,b=0), paper_bgcolor="#07131d", plot_bgcolor="#07131d", font_color="#d8eaf3",
                      scene=dict(aspectmode="data", xaxis=dict(showbackground=False), yaxis=dict(showbackground=False), zaxis=dict(showbackground=False)))
    return fig

source_path, target_path = DATA / f"{source_name}.ply", DATA / f"{target_name}.ply"
source, target = read_points(source_path), read_points(target_path)
if run:
    config = RegistrationConfig(coarse_method=coarse, fine_method=fine, voxel_size=voxel, max_correspondence_distance=distance, trim_fraction=trim, max_iterations=iterations)
    gt = None
    conf = DATA / "bun.conf"
    if conf.exists():
        poses = parse_bun_conf(conf)
        if source_name in poses and target_name in poses:
            gt = relative_transform(poses[source_name], poses[target_name])
    with st.spinner("正在计算粗配准与 ICP…"):
        st.session_state.result = register_pair(source, target, config, ground_truth=gt)
        st.session_state.selection = (source_name, target_name)

result = st.session_state.get("result")
if result and st.session_state.get("selection") == (source_name, target_name):
    cols = st.columns(5)
    values = [("状态", result.status), ("Fitness", f"{result.metrics.get('fitness',0):.3f}"), ("RMSE", f"{result.metrics.get('rmse',float('nan')):.6f}"),
              ("旋转误差", f"{result.metrics.get('rotation_error_deg',float('nan')):.2f}°"), ("总耗时", f"{result.timings_ms.get('total',0):.2f} ms")]
    for col, (label, value) in zip(cols, values): col.metric(label, value)
    st.caption(result.message)
    if not result.success:
        st.warning("当前两帧直接配准未通过成功阈值。该结果通常表示源/目标重叠不足、局部形状过于对称，或 FPFH/RANSAC 初值落入错误姿态；程序不会自动读取第三帧点云或使用桥接图。")
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
