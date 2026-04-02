$ErrorActionPreference = "Stop"

$root = "C:\Users\haoha\Documents\New project\DSTA-SLR"
$python = "C:\Users\haoha\miniconda3\envs\dsta-slr\python.exe"

function Invoke-QuickPilot {
    param(
        [string]$DatasetName
    )

    Write-Host "Running quick pilot for $DatasetName"
    & $python -u "scripts\run_confidence_suite.py" `
        --dataset $DatasetName `
        --device 0 `
        --num-worker 0 `
        --num-epoch 20 `
        --overwrite-work-dir

    if ($LASTEXITCODE -ne 0) {
        throw "Quick pilot failed for $DatasetName"
    }
}

Invoke-QuickPilot -DatasetName "SLR500"
Invoke-QuickPilot -DatasetName "MSASL500"
Invoke-QuickPilot -DatasetName "MSASL1000"

Write-Host "SLR500 / MSASL500 / MSASL1000 quick queue finished."
