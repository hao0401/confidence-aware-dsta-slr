$ErrorActionPreference = "Stop"

$root = "C:\Users\haoha\Documents\New project\DSTA-SLR"
$python = "C:\Users\haoha\miniconda3\envs\dsta-slr\python.exe"

Set-Location $root

& $python scripts\run_wlasl100_repeats.py --variant baseline all_modules --epochs 250 --num-worker 0 --seed-start 1 --seed-end 3
& $python scripts\summarize_mean_std.py

Write-Host "Core 3-seed mean±std queue finished."
