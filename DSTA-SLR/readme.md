# Dynamic Spatial-Temporal Aggregation for Skeleton-Aware Sign Language Recognition

This codebase is built on the original [DSTA-SLR](https://github.com/hulianyuyy/DSTA-SLR) project for skeleton-aware sign language recognition and keeps the same overall training and evaluation workflow, while extending it with confidence-aware modeling, reliability-aware consistency training, robustness evaluation, and reproducible experiment utilities.

Skeleton-based sign language recognition uses sequences of body, hand, and facial keypoints as input instead of RGB frames. Compared with RGB-based pipelines, it is lighter to train and evaluate, while still providing a strong recognition baseline. The original DSTA-SLR work introduced a dynamic spatial-temporal aggregation network for this setting. This repository keeps that foundation and adds a confidence-aware research track for studying how pose-confidence signals affect recognition quality and robustness.

## Confidence-Aware Extension

Compared with the upstream DSTA-SLR release, this version adds:

- confidence-aware data handling in `feeders/feeder.py`
  - support for `original`, `constant`, and `shuffle` confidence modes
  - confidence remapping options such as `square`, `sqrt`, `power`, `rank`, and `binary`
  - missing-joint and coordinate-noise perturbations for robustness studies
- confidence-aware model updates in `model/fstgan.py`
  - confidence encoding
  - confidence-guided graph aggregation
  - temporal rectification from confidence maps
- reliability-aware consistency training in `main.py`
  - prediction-level consistency loss
  - feature-level consistency loss
  - optional confidence-derived weighting for supervision
- experiment and reporting support in `scripts/`
  - confidence configuration generation
  - four-stream confidence-aware training and fusion
  - repeat runs, mean/std summaries, and paper-table export helpers
  - robustness and confidence-distribution-shift evaluation workflows

## Data Preparation

The preprocessed skeleton data for NMFs-CSL, SLR500, MSASL, and WLASL are referenced from the original project release [here](https://mega.nz/folder/EvkEzIAC#gq_nWLbbWoj9WVnJGxnGaA). Please follow the dataset licenses, rules, and usage agreements before downloading or using any data.

For datasets used to train or test the model, place each dataset under `data/<DATASET_NAME>/`. For example, for `WLASL2000`:

```bash
mkdir -p data
ln -s /path/to/WLASL2000 ./data/WLASL2000
```

If you are working on Windows, you can also copy the dataset directory directly into `data/` or create a junction instead of a symlink.

Supported dataset keys in the configs include:

- `WLASL100`
- `WLASL300`
- `WLASL1000`
- `WLASL2000`
- `MSASL100`
- `MSASL200`
- `MSASL500`
- `MSASL1000`
- `SLR500`
- `NMFs-CSL`

## Pretrained Models

Optional pretrained checkpoints can be placed under `pretrained_models/`.

For a public release, large checkpoints are better shared through GitHub Releases or external storage instead of normal Git history. This repository does not bundle large model weights or training outputs.

## Installation

Using a dedicated Conda environment is the recommended setup, especially on Windows.

### Conda

```bash
conda env create -f environment.yml
conda activate dsta-slr
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.10.0+cu128 torchvision==0.25.0
python -m pip install -r requirements-conda.txt
```

PowerShell helpers are also provided:

```powershell
.\scripts\setup_conda_env.ps1
.\scripts\activate_conda_dsta_slr.ps1
```

### Legacy pip setup

```bash
pip install -r requirements.txt
```

## Training and Testing

The baseline DSTA-SLR workflow remains the main entry point.

### Training

```bash
python -u main.py --config config/train.yaml --device 0
```

### Testing

```bash
python -u main.py --config config/test.yaml --device 0
```

If you want to evaluate a trained checkpoint, set the weight path in the evaluation config or use the evaluation workflow that passes `--weights` explicitly.

## Confidence-Aware Workflows

In addition to the baseline training and testing setup, this repository provides confidence-aware experiment paths.

### Confidence-aware single-stream training

```bash
python -u main.py --config config/confidence/wlasl100_joint.yaml --device 0
```

### Confidence-aware four-stream training and fusion

```bash
python scripts/run_confidence_suite.py --dataset WLASL100 --device 0 --num-worker 0
```

### Reliability-aware consistency training

```bash
python scripts/run_wlasl100_consistency_suite.py --device 0 --num-worker 0 --num-epoch 100 --run-robustness
```

### Robustness evaluation

```bash
python scripts/run_robustness_suite.py --config config/confidence/wlasl100_joint.yaml --weights work_dir/conf_wlasl100_joint/save_models/best_model.pt --device 0 --num-worker 0 --out-file work_dir/conf_wlasl100_joint/robustness.csv
```

### Confidence distribution shift evaluation

```bash
python scripts/run_confidence_distribution_shift.py --device 0 --num-worker 0
```

## Ensembling

To reproduce multi-stream fusion results, train the required streams separately, then fuse the saved prediction files from `work_dir/`.

This repository keeps the upstream-style multi-stream setup while also providing a more explicit fusion utility:

```bash
python ensemble/fuse_streams.py \
  --label-path data/WLASL100/val_label.pkl \
  --data-path data/WLASL100/val_data_joint.npy \
  --joint work_dir/conf_wlasl100_fusion_inputs/best_acc_joint.pkl \
  --bone work_dir/conf_wlasl100_fusion_inputs/best_acc_bone.pkl \
  --joint-motion work_dir/conf_wlasl100_fusion_inputs/best_acc_joint_motion.pkl \
  --bone-motion work_dir/conf_wlasl100_fusion_inputs/best_acc_bone_motion.pkl \
  --out-dir work_dir/conf_wlasl100_fusion_results
```

For confidence-aware workflows, `scripts/run_confidence_suite.py` is the easiest entry point because it automates the four-stream training and fusion process.

## Scripts

The `scripts/` directory keeps stable top-level entry points, while the main implementations are grouped under:

- `scripts/experiments/`: experiment runner implementations
- `scripts/data_tools/`: dataset preparation and config generation helpers
- `scripts/reporting/`: summary and table export helpers
- `scripts/common/`: shared runtime, path, and artifact helpers

For more detail, see:

- `scripts/README.md`
- `scripts/EXPERIMENT_RUNBOOK.md`

## Notes for Public Use

- Keep datasets under `data/`.
- Keep experiment outputs under `work_dir/`.
- Keep large checkpoints out of normal Git history.
- Prefer the provided YAML configs and script entry points when reproducing experiments.

## Acknowledgements

This code is based on [SAM-SLR-v2](https://github.com/jackyjsy/SAM-SLR-v2), [SLGTformer](https://github.com/neilsong/SLGTformer), and the original [DSTA-SLR](https://github.com/hulianyuyy/DSTA-SLR). Many thanks to the original authors for open-sourcing their code.
