$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "common\runtime_helpers.ps1")

$root = Get-DstaSlrRoot -StartPath $PSScriptRoot
$condaExe = Resolve-CondaExecutable

& $condaExe env create -f (Join-Path $root "environment.yml") 2>$null
if ($LASTEXITCODE -ne 0) {
    & $condaExe env update -n dsta-slr -f (Join-Path $root "environment.yml")
}

$pythonExe = Join-Path (Get-CondaBase -CondaExe $condaExe) "envs\dsta-slr\python.exe"
if (!(Test-Path $pythonExe)) {
    throw "Cannot find the dsta-slr environment Python at $pythonExe"
}
& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.10.0+cu128 torchvision==0.25.0
& $pythonExe -m pip install -r (Join-Path $root "requirements-conda.txt")

Write-Host "Conda environment dsta-slr is ready."
