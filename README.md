# Confidence-Aware DSTA-SLR

This repository packages a confidence-aware extension of DSTA-SLR for skeleton-based sign language recognition together with the supporting utilities used for experiment management, robustness analysis, and paper revision.

The main codebase extends the original DSTA-SLR pipeline with confidence-aware modeling, reliability-aware consistency training, reproducible experiment runners, and robustness evaluation workflows across multiple sign language datasets.

## Highlights

- Confidence-aware data handling and graph-temporal modeling for skeleton sequences
- Reliability-aware consistency losses for training and ablation studies
- Robustness evaluation under missing-joint and noisy-pose perturbations
- Reproducible experiment scripts for WLASL, MSASL, SLR500, and NMFs-CSL workflows
- Paper-revision utilities and submission support documents kept alongside the main research code

## Repository Layout

- `DSTA-SLR/`: main training code, configs, models, fusion tools, and experiment runners
- `docs/wacv_revision/`: revision notes, supplementary submission material, and checklists
- `tools/paper_revision/`: DOCX cleanup and reviewer-alignment utilities
- `output/`: generated outputs and exports, ignored by Git
- `tmp/`: scratch files and local backups, ignored by Git

## Quick Start

The main training and evaluation workflow lives in `DSTA-SLR/`.

```bash
cd DSTA-SLR
conda env create -f environment.yml
conda activate dsta-slr
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.10.0+cu128 torchvision==0.25.0
python -m pip install -r requirements-conda.txt
```

Common entry points:

```bash
python -u main.py --config config/train.yaml --device 0
python scripts/run_confidence_suite.py --dataset WLASL100 --device 0 --num-worker 0
python scripts/run_robustness_suite.py --config config/confidence/wlasl100_joint.yaml --weights work_dir/conf_wlasl100_joint/save_models/best_model.pt --device 0 --num-worker 0
```

For full setup details and experiment notes, start with:

- [DSTA-SLR/readme.md](DSTA-SLR/readme.md)
- [DSTA-SLR/scripts/EXPERIMENT_RUNBOOK.md](DSTA-SLR/scripts/EXPERIMENT_RUNBOOK.md)
- [tools/paper_revision/README.md](tools/paper_revision/README.md)

## Data and Artifact Policy

Datasets, training outputs, logs, exported predictions, and large checkpoints are intentionally excluded from version control.

- Place datasets under `DSTA-SLR/data/<DATASET_NAME>/`
- Keep training outputs under `DSTA-SLR/work_dir/` or other ignored local directories
- Distribute large checkpoints through GitHub Releases or external storage instead of regular Git history

## License

The main code release includes the license file at [DSTA-SLR/LICENSE](DSTA-SLR/LICENSE).
