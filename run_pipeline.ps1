# Master VITON Orchestration Script

# 1. Paths (Auto-detection is handled by the python script)
$PROJECT_ROOT = "d:/Final Project Viton/virtual-tryon"
$OUTPUT_ROOT = "$PROJECT_ROOT/outputs"
$CONDA_PATH = "C:/Users/hp/anaconda3/condabin/conda.bat"

# 2. Options (Edit these as needed)
$GARMENT_TYPE = "flat"  # Options: flat, worn
$SLEEVE_TYPE = "full"   # Options: none, half, full
$PRESERVE_ARMS = $true  # Set to $true if you want to keep original hands/arms

$PARAMS = @(
    "--type", $GARMENT_TYPE,
    "--sleeve_type", $SLEEVE_TYPE,
    "--conda_path", $CONDA_PATH,
    "--project_root", $PROJECT_ROOT,
    "--output_root", $OUTPUT_ROOT
)

if ($PRESERVE_ARMS) {
    $PARAMS += "--preserve_arms"
}

Write-Host "--- STARTING MASTER VITON PIPELINE (AUTO-DETECT) ---" -ForegroundColor Cyan
Write-Host "Searching for files in: $PROJECT_ROOT/inputs/"

python "$PROJECT_ROOT/src/master_pipeline.py" $PARAMS

Write-Host "`nPipeline execution attempt finished." -ForegroundColor Yellow
pause
