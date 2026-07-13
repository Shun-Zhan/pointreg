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

该表会额外输出 `overlap`、`supported_by_overlap` 和 `failure_reason`。在当前 Bunny 数据上，严格两帧算法的稳定工作区间约为重叠率不低于 0.5；低于该范围的组合通常会被标为 `low_overlap_unsupported`，应在报告中作为低重叠失败案例分析。

增加 `--full` 会进一步运行重叠率、体素尺度、固定随机种子的初始扰动和 10 次预热速度实验，并分别输出 CSV。

## 4. 方法与评价

- 粗配准：无、PCA（枚举轴排列和符号）、FPFH + RANSAC、`multi`（默认，低重叠全局搜索，见第 6 节）。
- FPFH + RANSAC：只在当前源点云和目标点云之间提取 FPFH 特征并执行 RANSAC 粗配准；不会使用第三帧点云或多帧桥接图。`bun.conf` 只用于结果评分，不参与求解变换矩阵。
- 精配准：自研 Point-to-Point ICP、Open3D Point-to-Plane ICP。
- 自研 ICP：对固定目标点云只构建一次 KD-tree，并在全部迭代中复用；随后执行最近邻、最大距离过滤、截断对应、SVD、反射修正、增量累计及 RMSE/位姿增量收敛。逐轮 RMSE 在应用本轮位姿增量后计算，因此曲线与该轮累计变换严格对应。
- 低重叠或近似对称组合可能出现大角度错配；UI 会把未通过阈值的结果标为不可靠。若课程要求严格只能使用源/目标两帧，这类组合应作为失败案例分析，而不是用额外扫描桥接掩盖。Bunny 全组合实验中，重叠率不低于 0.5 的组合可作为该算法的主要有效范围。
- 指标：Fitness、Inlier RMSE、有效对应数、旋转误差、平移误差、相对平移误差和各阶段耗时。
- 默认成功标准：旋转误差小于 5°且平移误差小于点云包围盒对角线的 2%。

`bun.conf` 的每一行按 `tx ty tz qx qy qz qw` 解析，记录的是扫描局部坐标到统一世界坐标的变换。因此源到目标的真值为 `inverse(T_target_world) @ T_source_world`。Stanford 旧版 ZipPack/Vrip 的四元数旋转约定与现代 Python 主动列向量约定方向相反，解析时需转置旋转矩阵；代码中已显式处理，并有真实数据回归测试防止方向再次写反。

## 5. 目录

```text
pointreg/       算法、数据、评价、实验和 CloudCompare 接口
tests/          单元测试与导出测试
app.py          Streamlit UI
bunny/data/     Stanford Bunny 多视角扫描与真值
outputs/        运行后生成的点云、JSON、CSV 和图表
```

速度结论应区分“FPFH/RANSAC 全局初始化”和“已有初值的 ICP 连续帧跟踪”。批量性能测试应先预热并重复至少 10 次，再报告中位数和波动范围。

## 6. 低重叠两帧全局配准（进行中）

目标：Bunny 全部 90 个有序两帧组合，在不使用桥接、不使用真值提示的前提下，达到旋转误差 < 5°、平移误差 < 包围盒对角线 2% 的 90/90。

### 6.1 当前进度

全量 90 对基准：**79/90**（单对 20–70 s，`tmp_proto/full90.py` 可复跑）。11 个失败对分两类：

- **毫米级平移漂移（6 对）**：如 `bun000↔bun180`、`bun180→chin`、`top2→chin`。旋转已收敛到 0.6–3.6°，平移偏差 3–10 mm（2–6% 对角线），刚好越过 2% 线。
- **大角度错配（5 对）**：`ear_back` 相关双向与 `bun090↔bun270`，重叠率仅 1–5%，全局搜索被"壳套壳"伪解压制。

### 6.2 方法

默认粗配准 `multi` 为多阶段全局搜索（`pointreg/global_search.py`）：

1. 超 Fibonacci 均匀采样 SO(3)（4000 旋转），带符号 FFT 体素相关双向打分——目标网格中扫描仪可见自由空间为负值，惩罚物理不可能的穿透；
2. 分散 top 种子随机爬山 + 短程截断 ICP 抛光；
3. 目标函数 `fitness − 3×violation` 的局部精化（FFT 重解平移）；
4. 围绕最优旋转的确定性细网格锁定（压切向滑移）；
5. 紧 Tukey 点到面终配准（双向 × 两档紧度）；
6. 自由空间门控 + 四指标 Borda 计票选优。

辅助模块 `pointreg/correspondence_search.py`：多尺度双向 FPFH 对应池、位姿共识计数 `consensus_count` 与共识 SVD 精化 `consensus_refine`，供后续实验使用（见 6.3 结论）。

### 6.3 兼容图路线实验结论（2026-07）

针对 2023–2025 学界对应关系兼容图方法（SC²-PCR / 3DMAC / TurboReg 思路）做了系统原型验证，关键数据：

- 多尺度（1.5/2.5/4 mm）双向 top-k FPFH 匹配下，6 个难对都存在真内点（34–174 个 / 约 35 万对应，内点率 0.02–0.2%），且内点可组成 3-团，SVD 位姿精度 0.1–0.7°——**信号存在**；
- 但 SC² 计票、边位姿投票、共识扩展、Open3D 对应 RANSAC 全部无法把真内点排到前列（真解 bin 排名数万开外）——**结构化外点（重复曲面几何）在所有对应级指标上真实胜过真解**；
- 把接近真值的候选注入现有候选池后，Borda 选优在 7/11 失败对上仍选中漂移伪解；FPFH 共识数同样偏好漂移解（共识精化会把真解拉向漂移解）。

结论：在该数据的极低重叠对上，表面贴合、自由空间、特征对应三族指标的判别下限都高于 2% 平移线，纯两帧免学习方法逼近信息论边界。兼容图方法不作为默认管线组成部分，模块保留供后续研究。

### 6.4 下一步优化计划

1. **终配准保真度**（对 6 个漂移对最有希望）：`finish_variants` 实验显示去掉宽入口、直接紧配（0.4/0.2 体素档）或中档单级（0.8/0.4）在多数难对上界内保持率更高；计划把终配准改为多变体并行 + 逐变体记录，替代现在的固定两档链。
2. **选优改造**：以"候选簇稳定性"替代单点指标——对每个候选统计其在不同变体/方向下的收敛散度，漂移伪解通常方向相关、散度大。
3. **大错对专项**：`ear_back` 双向与 `bun090↔bun270` 需要更强的旋转证据整合（SO(3) 网格 top-100 已含 5–10° 内的旋转，但平移无法定标）；候选方案是对 top 旋转做全平移相关多峰枚举 + 逐峰紧配准，用运行时间换覆盖。
4. **成功标准复核**：漂移量 3–10 mm 与 `bun.conf` 真值自身精度同量级，可在报告中并列给出 3% 线下的成功率作为参考。
