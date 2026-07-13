@echo off
REM ============================================================
REM  Create and configure the pointreg conda environment
REM  (includes GPU PyTorch and finetune dependencies)
REM  Run this from an "Anaconda Prompt", in the project root:
REM      finetune_kit\setup_env_windows.bat
REM ============================================================
setlocal
cd /d "%~dp0.."

echo.
echo [1/4] Creating pointreg environment from environment.yml ...
call conda env create -f environment.yml
if errorlevel 1 (
  echo [note] Create failed or env exists, trying to update ...
  call conda env update -n pointreg -f environment.yml --prune
)

echo.
echo [2/4] Activating pointreg ...
call conda activate pointreg
if errorlevel 1 (
  echo [error] Could not activate pointreg. Use "Anaconda Prompt" and retry.
  pause & exit /b 1
)

echo.
echo [3/4] Installing GPU PyTorch (CUDA 12.1) and deps (~2GB download) ...
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install einops coloredlogs easydict scikit-learn ipython tqdm

echo.
echo [4/4] Self-check: is CUDA available?
python -c "import torch;print('CUDA available:',torch.cuda.is_available())"

echo.
echo ============================================================
echo  Done. If it prints "CUDA available: True", you are ready.
echo  Next: run  finetune_kit\run_finetune_windows.bat
echo ============================================================
pause
endlocal
