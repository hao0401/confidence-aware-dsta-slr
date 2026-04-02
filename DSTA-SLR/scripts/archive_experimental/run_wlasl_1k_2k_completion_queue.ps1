$ErrorActionPreference = "Stop"

$root = "C:\Users\haoha\Documents\New project\DSTA-SLR"
$python = "C:\Users\haoha\miniconda3\envs\dsta-slr\python.exe"
$logDir = Join-Path $root "work_dir\queue_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Wait-ForTrainingToFinish {
    param(
        [string]$Pattern
    )
    while ($true) {
        $active = Get-CimInstance Win32_Process |
            Where-Object {
                $_.Name -like "python*" -and
                $_.CommandLine -like "*main.py*" -and
                $_.CommandLine -like "*$Pattern*"
            }
        if ($null -eq $active) {
            break
        }
        Write-Host "Waiting for active training matching $Pattern ..."
        Start-Sleep -Seconds 120
    }
}

function Wait-ForArtifacts {
    param(
        [string[]]$Paths
    )
    while ($true) {
        $missing = @()
        foreach ($path in $Paths) {
            if (-not (Test-Path $path)) {
                $missing += $path
            }
        }
        if ($missing.Count -eq 0) {
            break
        }
        Write-Host "Waiting for artifacts:"
        $missing | ForEach-Object { Write-Host ("  " + $_) }
        Start-Sleep -Seconds 60
    }
}

function Copy-FusionInputs {
    param(
        [string]$DatasetKey
    )
    $fusionInputDir = Join-Path $root ("work_dir\conf_" + $DatasetKey + "_fusion_inputs")
    New-Item -ItemType Directory -Force -Path $fusionInputDir | Out-Null
    foreach ($stream in @("joint", "bone", "joint_motion", "bone_motion")) {
        $workDir = Join-Path $root ("work_dir\conf_" + $DatasetKey + "_" + $stream)
        $scorePath = Join-Path $workDir "eval_results\best_acc.pkl"
        $modelPath = Join-Path $workDir "save_models\best_model.pt"
        if (-not (Test-Path $scorePath)) {
            throw "Missing score file for ${DatasetKey} ${stream}: ${scorePath}"
        }
        if (-not (Test-Path $modelPath)) {
            throw "Missing model file for ${DatasetKey} ${stream}: ${modelPath}"
        }
        Copy-Item -Force $scorePath (Join-Path $fusionInputDir ("best_acc_" + $stream + ".pkl"))
        Copy-Item -Force $modelPath (Join-Path $fusionInputDir ("best_model_" + $stream + ".pt"))
    }
}

function Run-Fusion {
    param(
        [string]$DatasetName,
        [string]$DatasetKey
    )
    $fusionInputDir = Join-Path $root ("work_dir\conf_" + $DatasetKey + "_fusion_inputs")
    $fusionOutputDir = Join-Path $root ("work_dir\conf_" + $DatasetKey + "_fusion_results")
    New-Item -ItemType Directory -Force -Path $fusionOutputDir | Out-Null
    & $python "ensemble\fuse_streams.py" `
        --label-path ("data\" + $DatasetName + "\val_label.pkl") `
        --data-path ("data\" + $DatasetName + "\val_data_joint.npy") `
        --joint (Join-Path $fusionInputDir "best_acc_joint.pkl") `
        --bone (Join-Path $fusionInputDir "best_acc_bone.pkl") `
        --joint-motion (Join-Path $fusionInputDir "best_acc_joint_motion.pkl") `
        --bone-motion (Join-Path $fusionInputDir "best_acc_bone_motion.pkl") `
        --window-size 120 `
        --out-dir $fusionOutputDir
}

function Run-StreamIfNeeded {
    param(
        [string]$ConfigPath,
        [string]$DatasetKey,
        [string]$StreamName,
        [int]$NumEpoch = 100
    )
    $workDir = Join-Path $root ("work_dir\conf_" + $DatasetKey + "_" + $StreamName)
    $metricsPath = Join-Path $workDir "eval_results\best_metrics.json"
    if (Test-Path $metricsPath) {
        Write-Host "Existing metrics found for conf_$DatasetKey`_$StreamName, skipping training."
        return
    }
    Write-Host "Running missing stream conf_$DatasetKey`_$StreamName"
    & $python -u "main.py" `
        --config $ConfigPath `
        --device 0 `
        --num-worker 0 `
        --num-epoch $NumEpoch `
        --overwrite-work-dir true
}

Wait-ForTrainingToFinish -Pattern "conf_wlasl1000_bone_motion"
Wait-ForArtifacts -Paths @(
    (Join-Path $root "work_dir\conf_wlasl1000_bone_motion\eval_results\best_acc.pkl"),
    (Join-Path $root "work_dir\conf_wlasl1000_bone_motion\save_models\best_model.pt")
)
Copy-FusionInputs -DatasetKey "wlasl1000"
Run-Fusion -DatasetName "WLASL1000" -DatasetKey "wlasl1000"

Run-StreamIfNeeded -ConfigPath "config\confidence\wlasl2000_bone_motion.yaml" -DatasetKey "wlasl2000" -StreamName "bone_motion"
Wait-ForArtifacts -Paths @(
    (Join-Path $root "work_dir\conf_wlasl2000_bone_motion\eval_results\best_acc.pkl"),
    (Join-Path $root "work_dir\conf_wlasl2000_bone_motion\save_models\best_model.pt")
)
Copy-FusionInputs -DatasetKey "wlasl2000"
Run-Fusion -DatasetName "WLASL2000" -DatasetKey "wlasl2000"

Write-Host "WLASL1000/WLASL2000 completion queue finished."
