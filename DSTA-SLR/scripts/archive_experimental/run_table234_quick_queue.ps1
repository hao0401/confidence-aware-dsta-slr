$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "..\common\runtime_helpers.ps1")

$root = Get-DstaSlrRoot -StartPath $PSScriptRoot
$python = Resolve-DstaSlrPython -Root $root

function Invoke-QuickPilot {
    param(
        [string]$DatasetName,
        [int]$NumEpoch
    )

    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Running quick pilot for $DatasetName ($NumEpoch epochs)"
    & $python -u "scripts\run_confidence_suite.py" `
        --dataset $DatasetName `
        --device 0 `
        --num-worker 0 `
        --num-epoch $NumEpoch `
        --overwrite-work-dir

    if ($LASTEXITCODE -ne 0) {
        throw "Quick pilot failed for $DatasetName"
    }
}

Set-Location $root

Invoke-QuickPilot -DatasetName "MSASL200" -NumEpoch 5
Invoke-QuickPilot -DatasetName "MSASL500" -NumEpoch 5
Invoke-QuickPilot -DatasetName "MSASL1000" -NumEpoch 5
Invoke-QuickPilot -DatasetName "SLR500" -NumEpoch 2
Invoke-QuickPilot -DatasetName "NMFs-CSL" -NumEpoch 3

Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Table 2/3/4 quick queue finished."
