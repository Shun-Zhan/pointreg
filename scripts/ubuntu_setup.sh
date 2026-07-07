#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v apt-get >/dev/null 2>&1; then
  echo "This script is intended for Debian/Ubuntu (apt-get)." >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y cloudcompare python3-venv python3-pip

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Done. Activate with: source $ROOT/.venv/bin/activate"
echo "Run tests: python -m pytest"
if command -v cloudcompare >/dev/null 2>&1; then
  echo "CloudCompare: $(command -v cloudcompare)"
elif command -v CloudCompare >/dev/null 2>&1; then
  echo "CloudCompare: $(command -v CloudCompare)"
else
  echo "CloudCompare not in PATH; set CLOUDCOMPARE_PATH if needed."
fi
