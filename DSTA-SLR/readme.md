# DSTA-SLR: Confidence-Aware Skeleton-Based Sign Language Recognition

This repository is based on the paper [Dynamic Spatial-Temporal Aggregation for Skeleton-Aware Sign Language Recognition](https://arxiv.org/pdf/2403.12519.pdf) and extends the original DSTA-SLR codebase with confidence-aware modeling, reliability-aware consistency training, robustness evaluation, and reproducible experiment scripts.

The goal of this version is not only to reproduce the baseline model, but also to support controlled ablations and robustness studies on skeleton confidence signals for sign language recognition.

## Key additions in this version

- Confidence-aware data handling in `feeders/feeder.py`
  - sanitizes pose confidence values
  - supports `original`, `constant`, and `shuffle` confidence modes
  - simulates missing joints and coordinate noise for robustness studies
- Confidence-aware model extensions in `model/fstgan.py`
  - confidence encoding
  - confidence-guided graph aggregation
  - temporal rectification based on confidence maps
- Reliability-aware consistency training in `main.py`
  - prediction-level consistency loss
  - feature-level consistency loss
  - optional confidence-derived weighting for consistency supervision
- Reproducible experiment utilities in `scripts/`
  - confidence configuration generation
  - four-stream training and fusion
  - WLASL100 ablation suites
  - robustness evaluation and reporting helpers
- Confidence-aware and uniform fusion tools in `ensemble/`

## Repository layout

- `config/`: baseline and confidence-aware YAML experiment configs
- `feeders/`, `graph/`, `model/`: data pipeline and network implementation
- `ensemble/`: score fusion utilities
- `scripts/`: experiment runners, data preparation helpers, and reporting tools
- `pretrained_models/`: optional pretrained checkpoints

## Data preparation

Datasets and training outputs are intentionally not included in this repository.

The preprocessed skeleton data for NMFs-CSL, SLR500, MSASL, and WLASL are referenced in the original project release [here](https://mega.nz/folder/EvkEzIAC#gq_nWLbbWoj9WVnJGxnGaA). Please follow the dataset licenses and usage agreements before downloading or using any data.

Place each dataset under `data/<DATASET_NAME>/`. For example, for WLASL2000:

```bash
mkdir -p data
ln -s /path/to/WLASL2000 ./data/WLASL2000
```

If you are working on Windows, you can also place the folder directly under `data/` or create a junction instead.

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

## Installation

Using a dedicated Conda environment is the recommended setup, especially on Windows.

### Conda

```bash
conda env create -f environment.yml
conda activate dsta-slr
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.10.0+cu128 torchvision==0.25.0
python -m pip install -r requirements-conda.txt
```

PowerShell helpers are also included:

```powershell
.\scripts\setup_conda_env.ps1
.\scripts\activate_conda_dsta_slr.ps1
```

### Legacy pip / venv setup

```bash
pip install -r requirements.txt
```

## Quick start

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

Additional workflow notes are documented in `scripts/README.md` and `scripts/EXPERIMENT_RUNBOOK.md`.

## Pretrained models

Optional pretrained checkpoints can be placed under `pretrained_models/`. For a public GitHub repository, it is usually better to distribute large model files through GitHub Releases or external storage instead of regular Git history.

## Notes for public release

- Keep `data/`, `work_dir/`, local environments, and logs out of version control.
- Large checkpoints are better distributed through GitHub Releases or external storage than through regular Git history.

## Acknowledgements

This code builds on [SAM-SLR-v2](https://github.com/jackyjsy/SAM-SLR-v2) and [SLGTformer](https://github.com/neilsong/SLGTformer). Thanks to the original authors for open-sourcing their work.
