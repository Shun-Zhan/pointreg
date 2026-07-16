#!/usr/bin/env bash
# Ubuntu/Debian 一键环境搭建脚本：安装 CloudCompare，创建虚拟环境并装好依赖。
set -euo pipefail   # 出错即停、未定义变量报错、管道任一环节失败即失败
# 无论从哪里调用，都切换到项目根目录（脚本所在目录的上一级）
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 本脚本依赖 apt-get，仅适用于 Debian/Ubuntu 系发行版
if ! command -v apt-get >/dev/null 2>&1; then
  echo "This script is intended for Debian/Ubuntu (apt-get)." >&2
  exit 1
fi

# 安装 CloudCompare（可视化）与 Python 虚拟环境/包管理工具
sudo apt-get update
sudo apt-get install -y cloudcompare python3-venv python3-pip

# 若虚拟环境尚不存在则创建
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate           # 激活虚拟环境
pip install --upgrade pip
pip install -r requirements.txt     # 安装项目依赖

echo "Done. Activate with: source $ROOT/.venv/bin/activate"
echo "Run tests: python -m pytest"
if command -v cloudcompare >/dev/null 2>&1; then
  echo "CloudCompare: $(command -v cloudcompare)"
elif command -v CloudCompare >/dev/null 2>&1; then
  echo "CloudCompare: $(command -v CloudCompare)"
else
  echo "CloudCompare not in PATH; set CLOUDCOMPARE_PATH if needed."
fi
