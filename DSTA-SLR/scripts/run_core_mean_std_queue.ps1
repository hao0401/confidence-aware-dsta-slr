$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "common\runtime_helpers.ps1")

$root = Get-DstaSlrRoot -StartPath $PSScriptRoot
$python = Resolve-DstaSlrPython -Root $root

Set-Location $root

& $python scripts\run_wlasl100_repeats.py --variant baseline all_modules --epochs 250 --num-worker 0 --overwrite-work-dir
& $python scripts\summarize_mean_std.py

Write-Host "Core mean±std queue finished."
