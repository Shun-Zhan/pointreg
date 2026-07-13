@echo off
REM ============================================================
REM  One-click: finetune GeoTransformer (3DMatch) + before/after eval
REM  Run from an "Anaconda Prompt", in the project root:
REM      finetune_kit\run_finetune_windows.bat
REM  Requires: conda activate pointreg, and a working NVIDIA GPU
REM ============================================================
setlocal
cd /d "%~dp0.."

echo.
echo [1/4] Checking GPU ...
python -c "import torch;print('CUDA available:',torch.cuda.is_available());print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
if errorlevel 1 (
  echo [error] Cannot run python/torch. Run "conda activate pointreg" first.
  pause & exit /b 1
)

echo.
echo [2/4] Before finetune: evaluate base 3DMatch weights (baseline) ...
python finetune_kit\evaluate_3dmatch.py --checkpoint checkpoints\geotransformer-3dmatch.pth.tar --output outputs\eval_before.json

echo.
echo [3/4] Finetuning (default 400 steps, a few minutes on GPU) ...
python finetune_kit\finetune_3dmatch_bunny.py --steps 400 --lr 1e-4

echo.
echo [4/4] After finetune: evaluate the finetuned weights ...
python finetune_kit\evaluate_3dmatch.py --checkpoint checkpoints\geotransformer-bunny-3dmatch-ft.pth.tar --output outputs\eval_after.json

echo.
echo ============================================================
echo  Done. Compare:
echo    before: outputs\eval_before.json
echo    after : outputs\eval_after.json
echo  Look at final_rot (smaller is better) and OK/FAIL per line.
echo ============================================================
pause
endlocal
