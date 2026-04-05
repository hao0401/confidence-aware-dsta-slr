$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "common\runtime_helpers.ps1")

$root = Get-DstaSlrRoot -StartPath $PSScriptRoot
$python = Resolve-DstaSlrPython -Root $root

Set-Location $root

& $python scripts\run_wlasl100_repeats.py --variant baseline all_modules --epochs 250 --num-worker 0 --seed-start 1 --seed-end 3
& $python scripts\summarize_mean_std.py

Write-Host "Core 3-seed mean±std queue finished."
