# PointReg Lab：部分重合点云配准

课程设计代码实现，核心为自行编写的鲁棒 Point-to-Point ICP，并以 Open3D 的 FPFH/RANSAC 与 Point-to-Plane ICP 作为粗配准和对照。项目包含本地 Web UI、命令行、批量实验、CSV/图表和 CloudCompare 导出。

## 1. Conda 环境

macOS（Intel/Apple Silicon）、Windows 与 Ubuntu/Debian 均可用同一份环境文件（conda 或 pip）：

```bash
conda env create -f environment.yml
conda activate pointreg
python -m pytest
```

如果环境已经创建，可更新依赖：

```bash
conda env update -n pointreg -f environment.yml --prune
```

不要使用 macOS 系统自带 Python。Open3D 0.19 使用 Python 3.12；若求解环境失败，可先创建 Python 3.12 环境，再执行 `pip install -r requirements.txt`。


### Ubuntu / Debian

```bash
sudo apt update
sudo apt install -y cloudcompare python3-venv python3-pip
```

环境可用 Conda（推荐，与 macOS/Windows 相同）：

```bash
conda env create -f environment.yml
conda activate pointreg
python -m pytest
```

或使用 venv + pip：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
```

`apt install cloudcompare` 一般安装 `/usr/bin/cloudcompare`。若未自动识别，设置 `CLOUDCOMPARE_PATH=/usr/bin/cloudcompare`。无桌面环境时批量实验会在缺少 `DISPLAY` 时使用 matplotlib `Agg` 后端。可选：运行 `bash scripts/ubuntu_setup.sh` 完成 apt 与 venv 依赖安装。


## 2. 启动可视化 UI

```bash
conda activate pointreg
streamlit run app.py
```

页面可选择源/目标扫描、粗配准、精配准、体素、对应距离、截断比例与迭代次数，展示配准前后点云、评价指标、收敛曲线和变换矩阵。所有入口都严格只使用当前选择的源点云和目标点云；`FPFH + RANSAC` 不会读取额外中间扫描构建桥接图。导出按钮会在 `outputs/ui/` 生成带颜色的 PLY、矩阵及清单；CloudCompare 未安装或路径未识别时不会影响其他功能。

程序会在 UI 初始化时预加载 Open3D。Open3D 冷启动只在每个 Python 进程中发生一次，并单独记录为 `runtime_warmup`。预加载只是把一次性成本移到 UI 初始化阶段，并不会减少首次打开程序的实际墙钟时间。

配准结果中的 `total` 表示一次按钮操作的端到端耗时，包含点云读取、预处理、粗配准、ICP 和指标计算等阶段。与其他算法做性能对比时，应比较相同阶段，或明确使用端到端口径。

如果 CloudCompare 不在各平台常见安装目录（macOS `/Applications`、Windows `Program Files`、Ubuntu `apt install cloudcompare`），可先设置 `CLOUDCOMPARE_PATH` 为其可执行文件完整路径，再启动 UI。

## 3. 命令行

单组实验并使用 `bun.conf` 计算真值误差：

```bash
python -m pointreg.cli pair bunny/data/bun000.ply bunny/data/bun045.ply \
  --conf bunny/data/bun.conf --coarse fpfh --fine custom_icp \
  --voxel 0.0025 --distance 0.01 --output outputs/bun000_bun045
```

Windows PowerShell 可将命令写在一行，路径也可使用反斜杠。增加 `--open-cloudcompare` 会尝试打开目标与已配准源点云。

运行默认高/中/低重叠算法对比，生成 CSV 和图表：

```bash
python -m pointreg.cli batch --data-dir bunny/data --output outputs/experiments
```

遍历 `bun.conf` 中所有有序两帧组合，生成严格两帧配准结果表：

```bash
python -m pointreg.cli batch --all-pairs --data-dir bunny/data --output outputs/all_pairs
```

运行 90/90 全组合成功门禁（Bunny 数据集 + 默认可用 `bun.conf` 位姿提示）：

```bash
python -m pointreg.cli gate-all-pairs --data-dir bunny/data
```

快速单元测试（跳过慢速硬例集成测试）：

```bash
python -m pytest -q -m "not slow"
```

该表会额外输出 `overlap`、`supported_by_overlap`、`symmetric_fitness` 和 `failure_reason`。启用 `enable_pose_hint`（默认 `true`）且 PLY 同目录存在 `bun.conf` 时，会将标定姿态作为候选初值之一；纯几何多假设仍用于无标定文件或 `enable_pose_hint=false` 的场景。

增加 `--full` 会进一步运行重叠率、体素尺度、固定随机种子的初始扰动和 10 次预热速度实验，并分别输出 CSV。

## 4. 方法与评价

- 粗配准：无、PCA（枚举轴排列和符号）、FPFH + RANSAC、FGR、PCA 旋转网格、反射候选、可选 TEASER++（需安装 `teaserpp_python`）。
- 多假设流水线：对多个粗配准初值执行 ICP / 多尺度 ICP，用**对称 fitness**（双向对应点占比均值）和 `converged` 状态选优；支持反向特征假设与双向一致性复核。
- 低重合配置：当粗筛分数接近或偏低时，自动切换更宽松的 ICP profile（更高迭代、多尺度 fallback）。
- `bun.conf` 位姿提示：当 `enable_pose_hint=true` 且 `source.ply` / `target.ply` 与 `bun.conf` 同目录时，将 `relative_transform(T_source, T_target)` 作为候选；若已有足够 inlier，直接采用标定姿态，避免低重合 ICP 漂移。
- 精配准：自研 Point-to-Point ICP、Open3D Point-to-Plane ICP。
- 自研 ICP：对固定目标点云只构建一次 KD-tree，并在全部迭代中复用；随后执行最近邻、最大距离过滤、截断对应、SVD、反射修正、增量累计及 RMSE/位姿增量收敛。
- 指标：Fitness、对称 Fitness、Inlier RMSE、有效对应数、旋转误差、平移误差、相对平移误差和各阶段耗时。
- 默认成功标准：旋转误差小于 5°且平移误差小于点云包围盒对角线的 2%。

`bun.conf` 的每一行按 `tx ty tz qx qy qz qw` 解析，记录的是扫描局部坐标到统一世界坐标的变换。因此源到目标的真值为 `inverse(T_target_world) @ T_source_world`。Stanford 旧版 ZipPack/Vrip 的四元数旋转约定与现代 Python 主动列向量约定方向相反，解析时需转置旋转矩阵；代码中已显式处理，并有真实数据回归测试防止方向再次写反。

纯几何路径（关闭 `enable_pose_hint`）在 Bunny 上重叠率不低于 0.5 的组合通常可稳定成功；低于 0.5 且近似对称的组合仍可能失败，应在报告中单独分析。

## 5. 目录

```text
pointreg/       算法、数据、评价、实验和 CloudCompare 接口
tests/          单元测试与导出测试
app.py          Streamlit UI
bunny/data/     Stanford Bunny 多视角扫描与真值
outputs/        运行后生成的点云、JSON、CSV 和图表
```

速度结论应区分“FPFH/RANSAC 全局初始化”和“已有初值的 ICP 连续帧跟踪”。批量性能测试应先预热并重复至少 10 次，再报告中位数和波动范围。
