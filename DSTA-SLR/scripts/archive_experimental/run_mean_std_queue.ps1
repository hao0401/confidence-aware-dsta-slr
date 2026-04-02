$ErrorActionPreference = "Stop"

$root = "C:\Users\haoha\Documents\New project\DSTA-SLR"
$python = "C:\Users\haoha\miniconda3\envs\dsta-slr\python.exe"
$variants = @(
    "baseline",
    "node_encoding_only",
    "graph_only",
    "temporal_only",
    "all_modules"
)

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

foreach ($variant in $variants) {
    Write-Host "Running repeated WLASL100 experiments for $variant"
    & $python -u scripts\run_wlasl100_repeats.py --variant $variant --epochs 100
}

& $python scripts\summarize_mean_std.py

Write-Host "Mean/std queue finished."
