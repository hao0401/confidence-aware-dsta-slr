$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "..\common\runtime_helpers.ps1")

$root = Get-DstaSlrRoot -StartPath $PSScriptRoot
$python = Resolve-DstaSlrPython -Root $root
$candidateDatasets = @(
    "WLASL2000",
    "MSASL100",
    "MSASL200",
    "MSASL500",
    "MSASL1000",
    "SLR500",
    "NMFs-CSL"
)

Set-Location $root
$datasets = & $python scripts\list_valid_datasets.py --datasets $candidateDatasets

function Get-TrainingProcess {
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -like "python*" -and
            $_.CommandLine -like "*main.py*" -and
            $_.CommandLine -like "*conf_*"
        }
}

while ($true) {
    $active = Get-TrainingProcess
    if ($null -eq $active) {
        break
    }
    Write-Host "Active training process detected, waiting 300 seconds..."
    Start-Sleep -Seconds 300
}

foreach ($dataset in $datasets) {
    Write-Host "Running benchmark suite for $dataset"
    & $python -u scripts\run_confidence_suite.py --dataset $dataset --device 0 --num-worker 0 --num-epoch 100
}

Write-Host "Benchmark queue finished."
