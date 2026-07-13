# 交接给 Codex:配置环境 + 微调 GeoTransformer(低重合点云配准)

## 背景(一句话)
项目 `pointreg--` 用 GeoTransformer 做 Stanford Bunny 点云配准。传统方法在低重合对上失败;
已验证 GeoTransformer 的 **3DMatch 权重**能把 `bun000->bun180` 从 177°(完全错)救到 17.7°(接近),
但仍未过 5° 成功线。现在要**微调 3DMatch 权重**以适配 Bunny 低重合分布,争取过线。

所有脚本已写好并在 CPU 上验证可运行,放在 `finetune_kit\`。Codex 需要做的是:
在本机(Windows + NVIDIA GPU)把环境的 torch 换成 GPU 版,然后运行微调与评测。

## 当前环境现状(已确认)
- conda 环境路径:`D:\pointreg--\.conda\pointreg`(用 `conda activate D:\pointreg--\.conda\pointreg` 激活)
- 现装的是 **CPU 版 torch 2.5.1+cpu**(`torch.cuda.is_available()` 为 False)——这是唯一要改的地方。
- 已有权重文件:`checkpoints\geotransformer-3dmatch.pth.tar`(39MB,微调的起点,已就位)。

## 任务 1:把 torch 换成 GPU 版
1. 激活环境:`conda activate D:\pointreg--\.conda\pointreg`
2. 查驱动支持的 CUDA 上限:运行 `nvidia-smi`,看右上角 `CUDA Version`。
3. 卸载 CPU 版、装 GPU 版(按驱动选 cu121 或 cu118):
   - CUDA Version ≥ 12.1:
     ```
     pip uninstall -y torch
     pip install torch --index-url https://download.pytorch.org/whl/cu121
     ```
   - CUDA Version 在 11.8~12.0:
     ```
     pip uninstall -y torch
     pip install torch --index-url https://download.pytorch.org/whl/cu118
     ```
4. 装 GeoTransformer 依赖(GeoTransformer 的 C++/CUDA 扩展在本项目有纯 Python CPU 回退
   `geotransformer/ext.py`,**无需编译**,不用装 VS Build Tools):
   ```
   pip install einops coloredlogs easydict scikit-learn ipython tqdm
   ```
5. 自检,必须输出 True:
   ```
   python -c "import torch;print(torch.cuda.is_available())"
   ```

## 任务 2:运行微调 + 前后评测
在项目根目录 `D:\pointreg--` 下,环境已激活:

1. 微调前基线评测(原版 3DMatch 权重):
   ```
   python finetune_kit\evaluate_3dmatch.py --checkpoint checkpoints\geotransformer-3dmatch.pth.tar --output outputs\eval_before.json
   ```
2. 微调(默认 400 步;GPU 上通常几分钟~十几分钟。可加大到 800/1500):
   ```
   python finetune_kit\finetune_3dmatch_bunny.py --steps 400 --lr 1e-4
   ```
   产出:`checkpoints\geotransformer-bunny-3dmatch-ft.pth.tar`
3. 微调后评测:
   ```
   python finetune_kit\evaluate_3dmatch.py --checkpoint checkpoints\geotransformer-bunny-3dmatch-ft.pth.tar --output outputs\eval_after.json
   ```

## 验收标准
对比 `outputs\eval_before.json` 与 `outputs\eval_after.json` 中每对的 `final_rot`(旋转误差,越小越好)
和 `success`。重点看 `bun000->bun180`:
- 微调前基线约 `final_rot=17.7, success=false`。
- 若微调后该行 `final_rot < 5` 且 `success=true`,即达成目标。
- `chin->top2`(重合区几乎为零)预期仍失败,属数据本身两帧不可解,可不追求。

## 脚本关键实现点(供 Codex 理解/排错,不用改)
- 微调脚本 `finetune_kit\finetune_3dmatch_bunny.py`:
  * 起点权重是 3DMatch(`checkpoints\geotransformer-3dmatch.pth.tar`),不是 ModelNet。
  * 用随机平面裁剪(保留 55%~85%)制造低重合训练样本;真值来自 `bun.conf`。
  * **尺度对齐**:把 Bunny(米级 ~0.15m)按 `scale = init_voxel_size / voxel = 0.025 / 0.0025 = 10` 放大,
    匹配 3DMatch 的工作尺度;平移也随之 ×10。这一步是 3DMatch 权重能生效的关键,勿动。
  * 3DMatch backbone 参数:`num_stages=4`, `init_voxel_size=0.025`, neighbor_limits=[38,36,36,38]。
  * 评测/训练的 5 个扫描(bun000/bun180/chin/top2/ear_back)被排除出训练集,防止数据泄漏。
  * 有 CPU 兜底补丁(`torch.Tensor.cuda` no-op),GPU 环境下不生效,不影响训练。
- 评测脚本 `finetune_kit\evaluate_3dmatch.py`:GeoTransformer 粗配准 + 项目自研 ICP 精配,
  报告 overlap / coarse_rot / final_rot / final_tr_ratio / fitness / success。

## 常见问题
- 自检仍 False:多半 cu121 与驱动不匹配,改用 cu118 重装 torch。
- 显存不足(OOM):本脚本每步只用 1 对点云,一般不会;若报错,给评测/微调加 `--voxel 0.003`(下采样更狠、点更少)。
- `No module named geotransformer` / `config` / `model`:必须在项目根目录 `D:\pointreg--` 下运行,
  脚本会把 `third_party\GeoTransformer-main` 及其 3dmatch experiment 目录加入 sys.path。
