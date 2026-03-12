@echo off
setlocal

:: 1. Paths (Auto-detection is handled by the python script)
set PROJECT_ROOT=d:/Final Project Viton/virtual-tryon
set OUTPUT_ROOT=%PROJECT_ROOT%/outputs
set CONDA_PATH=C:/Users/hp/anaconda3/condabin/conda.bat

:: 2. Options (Change as needed: none, half, full)
set GARMENT_TYPE=flat
set SLEEVE_TYPE=full
set PRESERVE_ARMS=--preserve_arms

echo --- STARTING MASTER VITON PIPELINE (AUTO-DETECT) ---
echo Searching for files in: %PROJECT_ROOT%/inputs/
echo ---------------------------------------------------

python "%PROJECT_ROOT%/src/master_pipeline.py" ^
    --type %GARMENT_TYPE% ^
    --sleeve_type %SLEEVE_TYPE% ^
    --conda_path "%CONDA_PATH%" ^
    --project_root "%PROJECT_ROOT%" ^
    --output_root "%OUTPUT_ROOT%" ^
    %PRESERVE_ARMS%

echo.
echo Pipeline execution attempt finished.
pause
