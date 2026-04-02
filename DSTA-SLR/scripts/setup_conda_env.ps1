$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$condaRoot = "C:\Users\haoha\miniconda3"
$condaExe = Join-Path $condaRoot "Scripts\conda.exe"
$pythonExe = Join-Path $condaRoot "envs\dsta-slr\python.exe"

if (!(Test-Path $condaExe)) {
    throw "Cannot find conda executable at $condaExe"
}

& $condaExe env create -f (Join-Path $root "environment.yml") 2>$null
if ($LASTEXITCODE -ne 0) {
    & $condaExe env update -n dsta-slr -f (Join-Path $root "environment.yml")
}

& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.10.0+cu128 torchvision==0.25.0
& $pythonExe -m pip install -r (Join-Path $root "requirements-conda.txt")

Write-Host "Conda environment dsta-slr is ready."
