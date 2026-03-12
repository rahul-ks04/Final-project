# run_stylevton.ps1
# This script runs the final StyleVTON generation pass.

$CONDA_ENV = "stylevton"
$PROJECT_ROOT = "d:/Final Project Viton/virtual-tryon"

# Check if environment is already active, if not suggest activation
if ($env:CONDA_DEFAULT_ENV -ne $CONDA_ENV) {
    Write-Host "Warning: You are not in the $CONDA_ENV environment." -ForegroundColor Yellow
    Write-Host "Please run: conda activate $CONDA_ENV"
    # We continue anyway as the user might be using a global python or manual setup
}

$AGNOSTIC_IMG = "$PROJECT_ROOT/outputs/agnostic/img/person.png"
$AGNOSTIC_MASK = "$PROJECT_ROOT/outputs/agnostic/mask/person.png"
$ORIGINAL_IMG = "$PROJECT_ROOT/outputs/rembg/person.png"
$WARPED_CLOTH = "$PROJECT_ROOT/outputs/flow_renderer/warped_garment.png"
$WARPED_MASK = "$PROJECT_ROOT/outputs/flow_renderer/projected_mask.png"
$CHECKPOINT = "$PROJECT_ROOT/Flow-Style-VTON/checkpoints/ckp/non_aug/PFAFN_gen_epoch_101.pth"
$OUTPUT_PATH = "$PROJECT_ROOT/outputs/final/tryon_result.png"

$PYTHON_EXE = "C:/Users/hp/anaconda3/envs/stylevton/python.exe"

Write-Host "--- Running StyleVTON Refined Generation ---" -ForegroundColor Cyan
Write-Host "Input Agnostic: $AGNOSTIC_IMG"
Write-Host "Input Warped Cloth: $WARPED_CLOTH"

& $PYTHON_EXE "$PROJECT_ROOT/src/run_stylevton.py" `
    --agnostic $AGNOSTIC_IMG `
    --original $ORIGINAL_IMG `
    --agnostic_mask $AGNOSTIC_MASK `
    --warped_cloth $WARPED_CLOTH `
    --warped_mask $WARPED_MASK `
    --checkpoint $CHECKPOINT `
    --output_path $OUTPUT_PATH

if ($LASTEXITCODE -eq 0) {
    Write-Host "Success! Result saved to $OUTPUT_PATH" -ForegroundColor Green
} else {
    Write-Host "Error: StyleVTON generation failed." -ForegroundColor Red
}
