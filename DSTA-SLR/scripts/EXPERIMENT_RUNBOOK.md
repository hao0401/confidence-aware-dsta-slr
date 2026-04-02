# Experiment Runbook

## 1. Reliability-Aware Consistency Training

Purpose:

- Train the four-stream confidence-guided model with prediction consistency and reliability-weighted feature consistency.

Main entry:

- `scripts/run_wlasl100_consistency_suite.py`

Default outputs:

- `work_dir/conf_wlasl100_consistency_<stream>/...`
- `work_dir/conf_wlasl100_consistency_fusion_results/metrics.json`
- `work_dir/conf_wlasl100_consistency_summary.json`

Example:

```powershell
python scripts/run_wlasl100_consistency_suite.py --device 0 --num-worker 0 --num-epoch 100 --run-robustness
```

## 2. Confidence Signal Validity Ablation

Purpose:

- Compare `original / constant / shuffle / no_confidence_signal` under the same code path.

Main entry:

- `scripts/run_confidence_signal_ablation.py`

Default outputs:

- `work_dir/conf_signal_wlasl100/confidence_signal_ablation.csv`
- `work_dir/conf_signal_wlasl100/confidence_signal_ablation.json`

Example:

```powershell
python scripts/run_confidence_signal_ablation.py --device 0 --num-worker 0 --num-epoch 100 --overwrite-work-dir
```

## 3. Comparative Robustness Evaluation

Purpose:

- Evaluate one trained checkpoint under missing-joint and coordinate-noise perturbations.

Main entry:

- `scripts/run_robustness_suite.py`

Example:

```powershell
python scripts/run_robustness_suite.py --config config/confidence/wlasl100_joint.yaml --weights work_dir/conf_wlasl100_joint/save_models/best_model.pt --device 0 --num-worker 0 --out-file work_dir/conf_wlasl100_joint/robustness.csv
```

## 4. Recommended Execution Order

1. Run confidence signal validity ablation.
2. Run reliability-aware consistency training.
3. Run robustness evaluation for:
   - baseline
   - structure-only confidence model
   - consistency-enhanced model
4. Fill paper tables:
   - Table 6: consistency training ablation
   - Table 7: confidence signal validity
   - Comparative robustness table if added later
