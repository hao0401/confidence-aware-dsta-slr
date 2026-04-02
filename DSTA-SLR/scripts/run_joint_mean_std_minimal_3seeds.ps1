$ErrorActionPreference = "Stop"

$root = "C:\Users\haoha\Documents\New project\DSTA-SLR"
$python = "C:\Users\haoha\miniconda3\envs\dsta-slr\python.exe"

Set-Location $root

& $python scripts\run_wlasl100_joint_repeats_minimal.py --variant baseline all_modules --epochs 5 --num-worker 0 --seed-start 1 --seed-end 3

Write-Host "Joint-only 3-seed mean±std queue finished."
