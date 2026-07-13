# GeoTransformer 低重合微调 · 傻瓜包

这个包用来在**你自己的电脑(有 NVIDIA 显卡)**上,把 GeoTransformer 的
3DMatch 权重微调到你的 Bunny 数据上,尝试把低重合对(尤其 bun000→bun180)
从"接近成功"推过成功线。

> 为什么要在你电脑上跑:准备这个包的 AI 运行在一个**只有 CPU 的云端沙箱**里,
> 训练需要显卡,而显卡在你的电脑里。所以脚本我都写好并验证过了,你只需要运行。

---

## 你只需要做三步

### 第 0 步:确认前提
- 已经按项目 README 建好 conda 环境,并且**装的是 GPU 版 PyTorch**
  (在项目根目录运行 `python -c "import torch;print(torch.cuda.is_available())"`,
   要输出 `True`。如果是 `False`,先装 GPU 版 torch,见文末)。
- `checkpoints\geotransformer-3dmatch.pth.tar` 这个文件存在(约 39MB)。
  这个包里应该已经带了;若没有,从下面链接下载放进 `checkpoints\`:
  https://github.com/qinzheng93/GeoTransformer/releases/download/1.0.0/geotransformer-3dmatch.pth.tar

### 第 1 步:激活环境
打开"Anaconda Prompt"或 PowerShell,进入项目根目录:
```
conda activate pointreg
cd 你的项目路径\pointreg
```

### 第 2 步:一键运行
```
finetune_kit\run_finetune_windows.bat
```
它会自动:①检查 GPU → ②评测微调前的基线 → ③微调 400 步 → ④评测微调后。
GPU 上整个过程通常几分钟到十几分钟。

### 第 3 步:看结果
对比这两个文件里每行末尾的 `final_rot`(旋转误差,越小越好)和 `OK/FAIL`:
- `outputs\eval_before.json` — 微调前
- `outputs\eval_after.json`  — 微调后

**判断标准**:`bun000->bun180` 那行的 `final_rot` 如果从 ~17° 降到 **5° 以内**、
且显示 `OK`,就是成功了。

---

## 想自己调参数(可选)

微调步数/学习率可以自己改,一般先试默认值。想手动跑:
```
python finetune_kit\finetune_3dmatch_bunny.py --steps 800 --lr 1e-4
python finetune_kit\evaluate_3dmatch.py --checkpoint checkpoints\geotransformer-bunny-3dmatch-ft.pth.tar
```
常见调节:
- 效果不明显 → 把 `--steps` 加大(如 800、1500),或试 `--lr 2e-4`。
- 训练报显存不足(out of memory)→ 这个脚本每步只用 1 对点云,一般不会;
  若仍报错,联系我把点云下采样体素调大(`--voxel 0.003`)。

---

## 重要说明(诚实交代)

1. **不保证一定成功**。我们已验证 3DMatch 权重能把 bun000→bun180 从"完全错(177°)"
   救到"接近(17°)",微调是想再进一步。但 `chin->top2` 这类重合区几乎为零的组合,
   **微调大概率也救不回**——那是数据本身两帧不可解,不是方法问题。
2. 评测用的 5 个扫描(bun000/bun180/chin/top2/ear_back)被**排除在训练之外**,
   保证对比公平、不作弊。
3. 训练是在"随机平面裁剪"制造的低重合样本上做的领域自适应,让模型见过
   Bunny 这种物体级、这种采样密度的低重合分布。

---

## 附:如果 torch.cuda.is_available() 是 False

说明装成了 CPU 版 torch。在激活的环境里重装 GPU 版(以 CUDA 12.1 为例):
```
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
装完再运行 `python -c "import torch;print(torch.cuda.is_available())"` 确认为 `True`。
