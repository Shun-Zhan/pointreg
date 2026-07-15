# PointReg Lab：低重合两帧点云配准

PointReg Lab 是一个严格使用源、目标两帧点云的刚体配准项目。项目同时提供传统 FPFH/ICP 基线和面向低重合点云的 GeoTransformer Fusion 后端，并包含 Streamlit WebUI、命令行、批量评测、CSV/JSON 结果以及 CloudCompare 导出。

本 README 以 `geotransformer` 分支为准，目标是让新的实验者能够从零配置环境并复现当前结果。

## 1. 当前方案说明

项目包含两条配准路径：

- 重合率 `>= 0.50`：WebUI 默认保留原有 FPFH/RANSAC、GC-RANSAC、PCA 和 ICP 方法。
- 重合率 `< 0.50`：WebUI 提示使用“低覆盖 GeoTransformer（Fusion，推荐）”。

低重合 Fusion 后端不是一个重新训练过的 Transformer。当前正式评测使用仓库内的原始 3DMatch 预训练权重：

```text
checkpoints/geotransformer-3dmatch.pth.tar
```

改进发生在推理和几何后端：

1. GeoTransformer 输出稠密对应、置信度和 LGR 位姿；
2. 使用阈值 `0.008/0.010/0.012` 和随机种子 `0/1/42` 生成九组 GC-RANSAC 候选；
3. 加入 LGR 与 FPFH 候选；
4. 独立采样 4000 个 SO(3) 旋转，并用带符号 FFT 搜索平移；
5. 使用自由空间 violation 门控排除穿透伪解；
6. 对优胜旋转执行局部旋转锁定；
7. 使用紧 Tukey 点到面精配；
8. 根据 fitness、violation、双向 RMSE 等指标做 Borda 选优。

候选生成和选优只使用当前两帧点云。`bun.conf` 真值仅用于计算重合率、误差和成功标志，不参与求解。

## 2. 仓库内已包含的复现材料

克隆仓库后应存在：

```text
bunny/data/                                      Bunny 点云与 bun.conf 真值
checkpoints/geotransformer-3dmatch.pth.tar       正式使用的 3DMatch 权重
checkpoints/geotransformer-bunny-3dmatch-ft.pth.tar  保留的实验性微调权重
third_party/GeoTransformer-main/                 运行所需 GeoTransformer 源码
pointreg/                                        配准实现
finetune_kit/evaluate_3dmatch.py                  低重合评测入口
app.py                                           Streamlit WebUI
tests/                                           自动测试
```

正式权重校验信息：

| 文件 | 大小（字节） | SHA256 |
|---|---:|---|
| `geotransformer-3dmatch.pth.tar` | 39,433,572 | `5C5FFE352BADDD83A12A8077451650235BB68A401367D7061344CD9C4AA3595C` |
| `geotransformer-bunny-3dmatch-ft.pth.tar` | 39,459,468 | `82D3377DCBEEB293C30226492BC4B509A6A02A3C26DDAF81505399C4D3F25D53` |

Windows PowerShell 校验命令：

```powershell
Get-FileHash .\checkpoints\geotransformer-3dmatch.pth.tar -Algorithm SHA256
```

如果权重文件不存在、尺寸明显不符或哈希不一致，请先重新完整克隆仓库，不要使用损坏文件继续评测。

## 3. 已验证环境

完整 Fusion 实测环境：

| 项目 | 已验证配置 |
|---|---|
| 操作系统 | Windows |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu128 |
| PyTorch CUDA runtime | 12.8 |
| Open3D | 0.19.0 |
| NumPy | 1.26.0 |
| pygcransac | 0.1.1 |
| einops | 0.8.2 |

传统 FPFH/ICP 可以在 CPU 上运行。GeoTransformer 代码也提供 CPU 回退，但完整 4000 旋转 Fusion 在 CPU 上会非常慢；复现实验强烈建议使用 NVIDIA GPU。

不同 GPU、CUDA 和线程调度可能造成耗时及最后几位浮点数不同。验收应比较成功率和误差范围，而不是要求 JSON 每一位完全一致。

## 4. Windows 从零配置（推荐）

### 4.1 前置条件

安装以下软件：

- Git
- Anaconda 或 Miniconda
- 支持 CUDA 的 NVIDIA 显卡驱动

在 PowerShell 或 Anaconda Prompt 中检查：

```powershell
git --version
conda --version
nvidia-smi
```

### 4.2 克隆并切换分支

```powershell
git clone <仓库地址> pointreg
cd pointreg
git switch geotransformer
```

### 4.3 创建 Conda 环境

```powershell
conda env create -f environment.yml
conda activate pointreg
```

如果环境已经存在：

```powershell
conda env update -n pointreg -f environment.yml --prune
conda activate pointreg
```

也可以在 Anaconda Prompt 中运行一键脚本：

```bat
finetune_kit\setup_env_windows.bat
```

该脚本安装 CUDA 12.1 版本 PyTorch。若需要尽量贴近本项目 RTX 4050 的实测环境，并且驱动支持 CUDA 12.8，可在激活环境后安装已验证版本：

```powershell
python -m pip install --upgrade torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

如果显卡驱动不支持 CUDA 12.8，保留一键脚本安装的 CUDA 12.1 PyTorch 即可。不要同时安装多个 Conda/Pip PyTorch 包。

### 4.4 环境自检

```powershell
python -c "import torch,open3d,pygcransac,streamlit; print('Torch:',torch.__version__); print('CUDA:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('Open3D:',open3d.__version__)"
```

GPU 复现应看到：

```text
CUDA: True
Open3D: 0.19.0
```

然后运行测试：

```powershell
python -m pytest -q
```

当前分支预期结果为：

```text
37 passed, 1 skipped
```

跳过项是可选平台能力，不影响核心配准。

## 5. Linux/macOS 配置

基础算法可使用同一份环境文件：

```bash
conda env create -f environment.yml
conda activate pointreg
python -m pytest -q
```

Ubuntu/Debian 也提供：

```bash
bash scripts/ubuntu_setup.sh
source .venv/bin/activate
```

注意：完整 Fusion 的正式结果目前只在 Windows + RTX 4050 上验证。macOS 或纯 CPU 环境可以运行基础算法，但不建议用来复现本文的耗时结论。

## 6. 启动 WebUI

```powershell
conda activate pointreg
python -m streamlit run app.py
```

浏览器默认打开：

```text
http://localhost:8501
```

操作步骤：

1. 在左侧选择源点云和目标点云；
2. WebUI 使用 `bun.conf` 真值、1 cm 最近邻阈值计算双向重合率；
3. 当重合率 `<0.50` 时，页面显示低重合提示；
4. 点击“使用推荐的低覆盖 GeoTransformer”；
5. 点击运行配准，并等待阶段进度完成。

单组完整 Fusion 在 RTX 4050 上通常约需 2–4 分钟。运行期间页面显示 GeoTransformer、候选生成、4000 旋转 FFT 搜索和自由空间选优等阶段。

WebUI 中的自动低重合判断仅适用于带有 `bun.conf` 真值的仓库内 Bunny 数据。对于用户自行上传且没有真值的点云，程序不能提前知道真实重合率，需要由用户手动选择方法或另行实现无真值重合率估计。

## 7. 复现42组低重合评测

当前数据包含10帧点云，共有 `10 × 9 = 90` 个有方向点对。按原始点云、1 cm双向覆盖率和 `<0.50` 阈值筛选后，共得到42组。

在项目根目录运行：

```powershell
python finetune_kit\evaluate_3dmatch.py `
  --checkpoint checkpoints\geotransformer-3dmatch.pth.tar `
  --backend fusion `
  --all-pairs `
  --overlap-max 0.5 `
  --resume `
  --output outputs\eval_C_low05.json
```

Linux/macOS 将反引号换为反斜杠，路径分隔符换为 `/`。

`--resume` 会跳过 JSON 中已经成功完成的点对。长时间运行中断后，执行同一条命令即可续跑。

输出文件：

```text
outputs/eval_C_low05.json          完整候选、位姿、诊断与阶段耗时
outputs/eval_C_low05.csv           每个方向点对一行的扁平结果
outputs/eval_C_low05.summary.json  成功率汇总
```

如需保存实时日志：

```powershell
python -u finetune_kit\evaluate_3dmatch.py --checkpoint checkpoints\geotransformer-3dmatch.pth.tar --backend fusion --all-pairs --overlap-max 0.5 --resume --output outputs\eval_C_low05.json *> outputs\eval_C_low05.log
```

另开一个 PowerShell 查看进度：

```powershell
Get-Content .\outputs\eval_C_low05.log -Wait -Tail 20
```

## 8. CSV 字段

`eval_C_low05.csv` 使用 UTF-8 BOM 编码，可直接用 Excel 打开。

| 字段 | 含义 |
|---|---|
| `pair/source/target` | 有方向点对及源、目标名称 |
| `backend/status/error` | 后端、运行状态和异常信息 |
| `overlap` | 真值对齐后的1 cm双向重合率，仅用于筛选与评测 |
| `correspondence_count` | GeoTransformer 稠密对应数量 |
| `gt_inliers_0.005` | 真值下残差不超过5 mm的对应数量 |
| `gt_inliers_0.010` | 真值下残差不超过10 mm的对应数量 |
| `gcransac_inliers` | GC-RANSAC 内点数；Fusion详细种子内点也保存在JSON中 |
| `selected_candidate` | 最终入选候选来源 |
| `coarse_rot_deg` | 粗配旋转误差，单位为度 |
| `coarse_tr_ratio` | 粗配平移误差/点云包围盒对角线 |
| `final_rot_deg` | 最终旋转误差，单位为度 |
| `final_tr_ratio` | 最终相对平移误差 |
| `fitness/global_fitness` | 最终对齐与全局搜索 fitness |
| `violation/fine_violation` | 粗、细自由空间冲突比例 |
| `gate_passed` | 是否通过自由空间门控 |
| `runtime_seconds` | 单组 Fusion 总耗时 |
| `success_2pct` | 旋转 `<5°` 且平移比 `<0.02` |
| `success_3pct` | 旋转 `<5°` 且平移比 `<0.03` |
| `success_practical_5pct` | 旋转 `<5°` 且平移比 `<0.05` |

真值相关字段用于实验诊断，不能作为候选生成或选优输入。

## 9. 当前参考结果

RTX 4050 上42组低重合有方向点对的参考结果：

| 标准 | 成功数 | 成功率 |
|---|---:|---:|
| 5° / 2% | 31/42 | 73.81% |
| 5° / 3% | 34/42 | 80.95% |
| 5° / 5% 实用标准 | 36/42 | 85.71% |

运行异常为 `0`。平均单组耗时约188.9秒，42组逐对耗时之和约2小时12分钟。

按5%实用标准仍失败的方向点对：

```text
bun000 -> ear_back
ear_back -> bun000
bun090 -> bun270
bun270 -> bun090
bun315 -> ear_back
ear_back -> bun315
```

参考结果用于判断环境和实现是否明显异常。由于GPU并行、GC-RANSAC和浮点计算差异，个别候选标签或最后几位数值可能不同。

## 10. 基础命令行配准

单组 FPFH + 自研 ICP：

```powershell
python -m pointreg.cli pair bunny\data\bun000.ply bunny\data\bun045.ply --conf bunny\data\bun.conf --coarse fpfh --fine custom_icp --voxel 0.0025 --distance 0.01 --output outputs\bun000_bun045
```

基础算法遍历全部有方向点对：

```powershell
python -m pointreg.cli batch --all-pairs --data-dir bunny\data --output outputs\all_pairs
```

该命令是传统两帧基线，不等同于完整 Fusion 评测。完整 Fusion 必须使用第7节的 `evaluate_3dmatch.py --backend fusion`。

## 11. 成功标准与真值约定

本项目同时报告三档标准：

- 严格：旋转误差 `<5°`，平移误差比 `<2%`；
- 辅助：旋转误差 `<5°`，平移误差比 `<3%`；
- 实用：旋转误差 `<5°`，平移误差比 `<5%`。

平移误差比定义为平移误差除以源、目标点云联合包围盒对角线。

`bun.conf` 每行按 `tx ty tz qx qy qz qw` 解析，记录扫描局部坐标到统一世界坐标的变换。源到目标真值为：

```text
inverse(T_target_world) @ T_source_world
```

Stanford Bunny 旧版 ZipPack/Vrip 四元数约定与现代 Python 主动列向量约定方向相反，代码已转置旋转矩阵，并由回归测试保护。

## 12. CloudCompare（可选）

CloudCompare 不影响算法运行。若需要从 WebUI 或 CLI 打开结果，可安装后设置：

```powershell
$env:CLOUDCOMPARE_PATH = "C:\Program Files\CloudCompare\CloudCompare.exe"
```

Ubuntu/Debian：

```bash
sudo apt install cloudcompare
export CLOUDCOMPARE_PATH=/usr/bin/cloudcompare
```

## 13. 常见问题

### `CUDA: False`

当前环境安装了 CPU 版 PyTorch，或 NVIDIA 驱动不可用。先运行 `nvidia-smi`，再按第4节安装对应 CUDA wheel。

### 找不到 `pygcransac`

```powershell
python -m pip install pygcransac==0.1.1
```

### 找不到 GeoTransformer 配置或模型

确认 `third_party/GeoTransformer-main/` 和 `checkpoints/geotransformer-3dmatch.pth.tar` 均存在，并校验权重哈希。

### WebUI 看起来停在“4000旋转 FFT全局搜索”

该阶段是计算量最大的阶段，RTX 4050 上可能持续数分钟，并不一定是卡死。可在任务管理器或 `nvidia-smi` 中检查 Python 进程和GPU占用。

### WebUI 没有提示低重合方法

自动提示需要当前点对在 `bun.conf` 中有真值，且按1 cm双向覆盖率计算结果 `<0.50`。普通点对继续显示原有方法。

### 评测中断

不要删除已有 JSON，使用完全相同的命令和 `--resume` 继续。

## 14. 项目目录

```text
app.py                       Streamlit WebUI
pointreg/fusion.py            GeoTransformer Fusion统一入口
pointreg/global_search.py     SO(3)、FFT、自由空间与Borda全局搜索
pointreg/geotransformer.py    3DMatch/ModelNet推理适配
pointreg/coarse.py            FPFH、RANSAC与GC-RANSAC
pointreg/icp.py               ICP实现
finetune_kit/                 评测、诊断和历史微调实验
third_party/                  仓库内固定的GeoTransformer源码
checkpoints/                  可复现权重
bunny/data/                   10帧Bunny点云与真值
tests/                        单元与WebUI测试
outputs/                      运行生成结果，默认不提交Git
```

## 15. 复现提交前检查清单

向其他实验者或老师提交前，建议确认：

```powershell
git status
Get-FileHash .\checkpoints\geotransformer-3dmatch.pth.tar -Algorithm SHA256
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pytest -q
python -m streamlit run app.py
```

需要复现完整科学结果时，再运行第7节的42组 Fusion 评测并保存 JSON、CSV、summary和日志。
