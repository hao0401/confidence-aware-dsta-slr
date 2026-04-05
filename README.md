# Confidence-Aware DSTA-SLR

Confidence-Aware DSTA-SLR is a research-oriented extension of DSTA-SLR for skeleton-based sign language recognition. This release focuses on making pose-confidence signals usable throughout the training and evaluation pipeline, with support for confidence-aware modeling, reliability-aware consistency learning, robustness analysis, and reproducible experiment workflows.

The public repository root is intentionally lightweight. It serves as the entry point to the main codebase, experiment scripts, and release-facing documentation, while large artifacts stay out of version control.

## Overview

This repository is centered on one practical question:

How much do skeleton confidence signals help sign language recognition, and how can they be used in a controlled, reproducible way?

To answer that, the codebase extends the original DSTA-SLR pipeline with:

- confidence-aware input handling and confidence perturbation controls
- model-side confidence encoding and aggregation changes
- reliability-aware consistency losses for training
- experiment runners for ablation, robustness, and repeat studies
- utilities for stream fusion and result reporting

## What You Will Find Here

- a clean public entry point for the project release
- the main research code under [`DSTA-SLR/`](DSTA-SLR/)
- runnable experiment workflows for baseline, confidence-aware, and robustness settings
- documentation for scripts, run order, and reproduction boundaries

## Start Here

- Main codebase: [`DSTA-SLR/`](DSTA-SLR/)
- Main usage guide: [`DSTA-SLR/readme.md`](DSTA-SLR/readme.md)
- Script index: [`DSTA-SLR/scripts/README.md`](DSTA-SLR/scripts/README.md)
- Experiment runbook: [`DSTA-SLR/scripts/EXPERIMENT_RUNBOOK.md`](DSTA-SLR/scripts/EXPERIMENT_RUNBOOK.md)
- License: [`DSTA-SLR/LICENSE`](DSTA-SLR/LICENSE)

## Recommended Reading Path

If you are opening this repository for the first time, the fastest path is:

1. Read [`DSTA-SLR/readme.md`](DSTA-SLR/readme.md) for the project overview and setup.
2. Check [`DSTA-SLR/scripts/README.md`](DSTA-SLR/scripts/README.md) to understand the script layout.
3. Use [`DSTA-SLR/scripts/EXPERIMENT_RUNBOOK.md`](DSTA-SLR/scripts/EXPERIMENT_RUNBOOK.md) to follow the intended experiment order.

## Repository Scope

The public release includes the code and documentation needed to understand and reproduce the confidence-aware DSTA-SLR workflows:

- training and evaluation code
- model modifications
- experiment orchestration scripts
- fusion and reporting utilities
- public-facing documentation

The following are intentionally excluded from version control:

- datasets
- checkpoints and large pretrained weights
- training outputs and logs
- temporary artifacts

## Notes for Public Use

- Data should be placed locally under `DSTA-SLR/data/`.
- Experiment outputs should be kept under `DSTA-SLR/work_dir/` or other ignored local directories.
- Large checkpoints are better distributed through GitHub Releases or external storage than through normal Git history.

## Acknowledgements

This release builds on the original DSTA-SLR line of work and related open-source sign language recognition projects referenced from the main codebase documentation.
