# DSTA-SLR: Confidence-Aware Skeleton-Based Sign Language Recognition

This directory contains the main research code for the confidence-aware DSTA-SLR release. It extends the original DSTA-SLR pipeline with confidence-aware modeling, reliability-aware consistency training, robustness evaluation, and reproducible experiment workflows for skeleton-based sign language recognition.

The codebase is designed for two practical goals:

- reproduce the baseline DSTA-SLR training and evaluation pipeline
- run controlled studies on how pose-confidence signals affect recognition accuracy and robustness

## What This Release Adds

- Confidence-aware data handling in `feeders/feeder.py`
  - sanitizes pose-confidence values
  - supports `original`, `constant`, and `shuffle` confidence modes
  - simulates missing joints and coordinate noise for robustness studies
- Confidence-aware model updates in `model/fstgan.py`
  - confidence encoding
  - confidence-guided graph aggregation
  - temporal rectification from confidence maps
- Reliability-aware consistency training in `main.py`
  - prediction-level consistency loss
  - feature-level consistency loss
  - optional confidence-derived weighting for consistency supervision
- Reproducible experiment tooling in `scripts/`
  - dataset-stream config generation
  - four-stream training and fusion
  - WLASL100 ablation suites
  - robustness evaluation and reporting helpers
- Fusion utilities in `ensemble/`
  - confidence-aware fusion
  - uniform-fusion control baselines

## Repository Structure

- `config/`: baseline and confidence-aware YAML experiment configurations
- `feeders/`, `graph/`, `model/`: data pipeline, graph definition, and network implementation
- `ensemble/`: stream fusion utilities and search helpers
- `scripts/`: experiment runners, data-preparation tools, and reporting scripts
- `pretrained_models/`: optional checkpoints and notes for external model distribution
- `main.py`: training and evaluation entry point

## Supported Datasets

The configs and experiment scripts currently support:

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

Preprocessed skeleton data are not bundled with this repository. The original project release references the prepared data package [here](https://mega.nz/folder/EvkEzIAC#gq_nWLbbWoj9WVnJGxnGaA). Please follow the relevant dataset licenses and usage terms before downloading or using any data.

Place each dataset under `data/<DATASET_NAME>/`. For example, for `WLASL2000`:

```bash
mkdir -p data
ln -s /path/to/WLASL2000 ./data/WLASL2000
```

On Windows, you can also copy the dataset folder directly under `data/` or create a junction instead of a symlink.

## Installation

Using a dedicated Conda environment is the recommended setup, especially on Windows.

### Conda

```bash
conda env create -f environment.yml
conda activate dsta-slr
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.10.0+cu128 torchvision==0.25.0
python -m pip install -r requirements-conda.txt
```

PowerShell helpers are also available:

```powershell
.\scripts\setup_conda_env.ps1
.\scripts\activate_conda_dsta_slr.ps1
```

### Legacy pip or venv setup

```bash
pip install -r requirements.txt
```

## Common Workflows

### Baseline training

```bash
python -u main.py --config config/train.yaml --device 0
```

### Baseline evaluation

```bash
python -u main.py --config config/test.yaml --device 0
```

### Confidence-aware single-stream training

```bash
python -u main.py --config config/confidence/wlasl100_joint.yaml --device 0
```

### Four-stream confidence-aware training and fusion

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

### Manual four-stream fusion

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

## Scripts and Runbook

For a high-level map of active scripts, see [`scripts/README.md`](scripts/README.md).

For the recommended execution order of the main experiments, see [`scripts/EXPERIMENT_RUNBOOK.md`](scripts/EXPERIMENT_RUNBOOK.md).

## Outputs and Artifact Policy

Datasets, local environments, logs, training outputs, and large checkpoints are intentionally excluded from version control.

- keep datasets under `data/`
- keep experiment outputs under `work_dir/`
- distribute large checkpoints through GitHub Releases or external storage instead of normal Git history

Optional pretrained checkpoints can be placed under `pretrained_models/`, but large model files are better shared through release assets or external storage.

## Acknowledgements

This code builds on [SAM-SLR-v2](https://github.com/jackyjsy/SAM-SLR-v2) and [SLGTformer](https://github.com/neilsong/SLGTformer). Thanks to the original authors for open-sourcing their work.
