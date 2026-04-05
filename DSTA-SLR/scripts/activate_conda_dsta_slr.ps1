. (Join-Path $PSScriptRoot "common\runtime_helpers.ps1")

Initialize-CondaShell
conda activate dsta-slr
Write-Host "Activated conda environment: dsta-slr"
