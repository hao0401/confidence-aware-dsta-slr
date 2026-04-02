$ErrorActionPreference = "Stop"

$root = "C:\Users\haoha\Documents\New project\DSTA-SLR"
$python = "C:\Users\haoha\miniconda3\envs\dsta-slr\python.exe"
$fusionDir = Join-Path $root "work_dir\wlasl100_fusion_inputs"
if (!(Test-Path $fusionDir)) {
    New-Item -ItemType Directory -Path $fusionDir | Out-Null
}

function Run-Stream {
    param(
        [string]$ConfigPath,
        [string]$ExperimentName,
        [string]$ScoreOutName,
        [string]$ModelOutName
    )

    & $python -u main.py --config $ConfigPath -Experiment_name $ExperimentName --device 0 --num-worker 0 --num-epoch 100

    $evalDir = Join-Path $root ("work_dir\" + $ExperimentName + "\eval_results")
    $modelDir = Join-Path $root ("work_dir\" + $ExperimentName + "\save_models")
    Copy-Item (Join-Path $evalDir "best_acc.pkl") (Join-Path $fusionDir $ScoreOutName) -Force
    Copy-Item (Join-Path $modelDir "best_model.pt") (Join-Path $fusionDir $ModelOutName) -Force
}

# Joint and bone were already trained before this script starts.
Run-Stream -ConfigPath "config/train_wlasl100_joint_motion.yaml" -ExperimentName "wlasl100_joint_motion_100" -ScoreOutName "best_acc_joint_motion.pkl" -ModelOutName "best_model_joint_motion.pt"
Run-Stream -ConfigPath "config/train_wlasl100_bone_motion.yaml" -ExperimentName "wlasl100_bone_motion_100" -ScoreOutName "best_acc_bone_motion.pkl" -ModelOutName "best_model_bone_motion.pt"

& $python ensemble\fuse_streams.py `
    --label-path data\WLASL100\val_label.pkl `
    --joint (Join-Path $fusionDir "best_acc_joint.pkl") `
    --bone (Join-Path $fusionDir "best_acc_bone.pkl") `
    --joint-motion (Join-Path $fusionDir "best_acc_joint_motion.pkl") `
    --bone-motion (Join-Path $fusionDir "best_acc_bone_motion.pkl") `
    --alpha 1.7 1.2 0.2 0.3 `
    --out-dir (Join-Path $root "work_dir\wlasl100_fusion_results")
