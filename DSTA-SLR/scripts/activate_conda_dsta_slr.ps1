$condaRoot = "C:\Users\haoha\miniconda3"
$condaHook = Join-Path $condaRoot "shell\condabin\conda-hook.ps1"

if (!(Test-Path $condaHook)) {
    throw "Cannot find conda hook at $condaHook"
}

. $condaHook
conda activate dsta-slr
Write-Host "Activated conda environment: dsta-slr"
