# Scripts Overview

`scripts/` keeps stable top-level entry points, but the real implementations now live in grouped subdirectories.

## Internal layout

- `experiments/`: active experiment runner implementations.
- `data_tools/`: dataset preparation, validation, and config generation utilities.
- `reporting/`: paper-table export and summary helpers.
- `common/`: shared metadata and runtime helpers.
- Top-level `run_*.py` and utility files are compatibility wrappers so existing commands still work.

## Active experiment entry points

- `run_confidence_suite.py`: four-stream training plus fusion for a selected dataset.
- `run_confidence_signal_ablation.py`: confidence signal validity ablations on WLASL100.
- `run_wlasl100_consistency_suite.py`: consistency-training runs and optional robustness evaluation.
- `run_wlasl100_43.py`: WLASL100 section 4.3 ablation and hyperparameter experiments.
- `run_wlasl100_repeats.py`: repeated four-stream WLASL100 runs for mean/std reporting.
- `run_wlasl100_joint_repeats_minimal.py`: minimal joint-only repeat runs.
- `run_robustness_suite.py`: missing-joint and noise robustness evaluation for one checkpoint.
- `run_ablation_suite.py`: generic ablation and hyperparameter sweep runner.
- `run_baseline_vs_ours_robustness.py`: side-by-side robustness comparison.
- `run_uniform_fusion_controls.py`: uniform-fusion control experiments.

## Support utilities

- `generate_confidence_configs.py`: generate dataset-stream YAML configs.
- `experiment_specs.py`: shared dataset and stream metadata.
- `python_locator.py`: locate the intended Python interpreter.
- `script_utils.py`: shared runtime, YAML/JSON/CSV, and checkpoint helpers.
- `validate_preprocessed_data.py`, `download_preprocessed_data.py`, `repair_corrupt_joint_npy.py`: data preparation helpers.
- `export_paper_tables.py`, `summarize_mean_std.py`: reporting helpers.
- `activate_conda_dsta_slr.ps1`, `setup_conda_env.ps1`: environment setup helpers.

## Archived scripts

- `archive_experimental/` keeps old queue runners and quick-pilot utilities that are not part of the main workflow.
- These scripts now resolve the repository root through `script_utils.py`, so nested archived runners stay runnable.
