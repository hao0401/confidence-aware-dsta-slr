import argparse
from pathlib import Path

from script_utils import (
    build_comparison_fieldnames,
    build_metric_comparison_rows,
    CONFIDENCE_SHIFT_STREAM_FIELDS,
    experiment_artifact_paths,
    extract_metric_fields,
    find_repo_root,
    read_json,
    read_yaml,
    run_fusion as run_fusion_job,
    write_csv,
    write_json,
)

from python_locator import resolve_python

try:
    from .run_robustness_suite import run_eval
except ImportError:
    from run_robustness_suite import run_eval


ROOT = find_repo_root(__file__)
PYTHON = resolve_python(ROOT)
STREAM_ORDER = ("joint", "bone", "joint_motion", "bone_motion")


def latest_score_file(eval_dir):
    candidates = sorted(
        eval_dir.glob("epoch_*.pkl"),
        key=lambda item: item.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def build_scenarios(binary_threshold):
    return [
        {
            "scenario": "clean",
            "description": "original confidence distribution",
            "feeder_overrides": {
                "confidence_transform": "identity",
            },
        },
        {
            "scenario": "square",
            "description": "square remapping that suppresses mid/low scores",
            "feeder_overrides": {
                "confidence_transform": "square",
            },
        },
        {
            "scenario": "sqrt",
            "description": "sqrt remapping that lifts mid/low scores",
            "feeder_overrides": {
                "confidence_transform": "sqrt",
            },
        },
        {
            "scenario": "rank",
            "description": "rank-based remapping to an approximately uniform distribution",
            "feeder_overrides": {
                "confidence_transform": "rank",
            },
        },
        {
            "scenario": f"binary_{str(binary_threshold).replace('.', '_')}",
            "description": f"binary thresholding at {binary_threshold:g}",
            "feeder_overrides": {
                "confidence_transform": "binary",
                "confidence_transform_threshold": binary_threshold,
            },
        },
    ]


def resolve_artifact_paths(prefix, stream_name):
    artifacts = experiment_artifact_paths(ROOT, f"{prefix}_{stream_name}")
    config_path = artifacts["config_path"]
    weights_path = artifacts["model_path"]
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config for {prefix}_{stream_name}: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint for {prefix}_{stream_name}: {weights_path}"
        )
    return config_path, weights_path


def evaluate_stream(
    prefix,
    stream_name,
    scenario_name,
    feeder_overrides,
    device,
    num_worker,
    overwrite,
):
    config_path, weights_path = resolve_artifact_paths(prefix, stream_name)
    base_config = read_yaml(config_path)
    eval_name = f"conf_shift_{prefix}_{stream_name}_{scenario_name}"
    eval_dir = ROOT / "work_dir" / eval_name / "eval_results"
    metrics_path = eval_dir / "last_metrics.json"
    score_path = latest_score_file(eval_dir)
    if overwrite or not metrics_path.exists() or score_path is None:
        metrics, score_path = run_eval(
            base_config=base_config,
            weights_path=weights_path,
            experiment_name=eval_name,
            feeder_overrides=feeder_overrides,
            device=device,
            num_worker=num_worker,
        )
    else:
        metrics = read_json(metrics_path)
    return base_config, metrics, score_path


def run_fusion(
    prefix,
    dataset_name,
    window_size,
    scenario_name,
    feeder_overrides,
    score_paths,
    overwrite,
):
    out_dir = ROOT / "work_dir" / "confidence_distribution_shift" / f"{prefix}_{scenario_name}_fusion"
    metrics_path = out_dir / "metrics.json"
    if overwrite or not metrics_path.exists():
        extra_args = [
            "--confidence-transform",
            feeder_overrides.get("confidence_transform", "identity"),
            "--confidence-transform-power",
            str(feeder_overrides.get("confidence_transform_power", 2.0)),
            "--confidence-transform-threshold",
            str(feeder_overrides.get("confidence_transform_threshold", 0.5)),
        ]
        if "confidence_mode" in feeder_overrides:
            extra_args.extend(["--confidence-mode", str(feeder_overrides["confidence_mode"])])
        if "confidence_constant_value" in feeder_overrides:
            extra_args.extend(
                [
                    "--confidence-constant-value",
                    str(feeder_overrides["confidence_constant_value"]),
                ]
            )
        run_fusion_job(
            ROOT,
            PYTHON,
            label_path=ROOT / "data" / dataset_name / "val_label.pkl",
            data_path=ROOT / "data" / dataset_name / "val_data_joint.npy",
            score_paths=score_paths,
            out_dir=out_dir,
            window_size=window_size,
            extra_args=extra_args,
        )
    return read_json(metrics_path)


def build_comparison_rows(method_results, metric_key):
    baseline_results = {
        scenario_name: result[metric_key]
        for scenario_name, result in method_results["baseline"].items()
    }
    ours_results = {
        scenario_name: result[metric_key]
        for scenario_name, result in method_results["ours"].items()
    }
    return build_metric_comparison_rows(
        baseline_results,
        ours_results,
        left_label="baseline",
        right_label="ours",
        retention_field_names={
            "baseline": "baseline_retention",
            "ours": "ours_retention",
        },
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline vs ours under confidence distribution shifts."
    )
    parser.add_argument("--baseline-prefix", default="conf_wlasl100_43_baseline")
    parser.add_argument("--ours-prefix", default="conf_wlasl100")
    parser.add_argument("--device", default="0")
    parser.add_argument("--num-worker", type=int, default=0)
    parser.add_argument("--binary-threshold", type=float, default=0.5)
    parser.add_argument("--skip-fusion", action="store_true")
    parser.add_argument("--overwrite-work-dir", action="store_true")
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "work_dir" / "confidence_distribution_shift"),
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    scenarios = build_scenarios(args.binary_threshold)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    method_prefixes = {
        "baseline": args.baseline_prefix,
        "ours": args.ours_prefix,
    }
    method_results = {key: {} for key in method_prefixes}
    stream_rows = []

    for method_name, prefix in method_prefixes.items():
        for scenario in scenarios:
            stream_metrics = {}
            score_paths = {}
            dataset_name = None
            window_size = None
            for stream_name in STREAM_ORDER:
                base_config, metrics, score_path = evaluate_stream(
                    prefix=prefix,
                    stream_name=stream_name,
                    scenario_name=scenario["scenario"],
                    feeder_overrides=scenario["feeder_overrides"],
                    device=args.device,
                    num_worker=args.num_worker,
                    overwrite=args.overwrite_work_dir,
                )
                dataset_name = dataset_name or base_config["dataset"]
                window_size = window_size or base_config["test_feeder_args"]["window_size"]
                stream_metrics[stream_name] = metrics
                score_paths[stream_name] = score_path
                stream_rows.append(
                    {
                        "method": method_name,
                        "prefix": prefix,
                        "scenario": scenario["scenario"],
                        "scenario_description": scenario["description"],
                        "stream": stream_name,
                        **extract_metric_fields(metrics),
                        "mean_loss": metrics["mean_loss"],
                    }
                )
            fusion_metrics = None
            if not args.skip_fusion:
                fusion_metrics = run_fusion(
                    prefix=prefix,
                    dataset_name=dataset_name,
                    window_size=window_size,
                    scenario_name=scenario["scenario"],
                    feeder_overrides=scenario["feeder_overrides"],
                    score_paths=score_paths,
                    overwrite=args.overwrite_work_dir,
                )
            method_results[method_name][scenario["scenario"]] = {
                "description": scenario["description"],
                "joint": stream_metrics["joint"],
                "bone": stream_metrics["bone"],
                "joint_motion": stream_metrics["joint_motion"],
                "bone_motion": stream_metrics["bone_motion"],
                "fusion": fusion_metrics,
            }

    write_csv(
        out_dir / "stream_metrics.csv",
        stream_rows,
        CONFIDENCE_SHIFT_STREAM_FIELDS,
    )

    joint_comparison_rows = build_comparison_rows(method_results, "joint")
    write_csv(
        out_dir / "joint_comparison.csv",
        joint_comparison_rows,
        build_comparison_fieldnames(
            "baseline",
            "ours",
            retention_field_names={
                "baseline": "baseline_retention",
                "ours": "ours_retention",
            },
        ),
    )

    if not args.skip_fusion:
        fusion_comparison_rows = build_comparison_rows(method_results, "fusion")
        write_csv(
            out_dir / "fusion_comparison.csv",
            fusion_comparison_rows,
            build_comparison_fieldnames(
                "baseline",
                "ours",
                retention_field_names={
                    "baseline": "baseline_retention",
                    "ours": "ours_retention",
                },
            ),
        )
    else:
        fusion_comparison_rows = None

    payload = {
        "baseline_prefix": args.baseline_prefix,
        "ours_prefix": args.ours_prefix,
        "scenarios": scenarios,
        "method_results": method_results,
        "joint_comparison_rows": joint_comparison_rows,
        "fusion_comparison_rows": fusion_comparison_rows,
    }
    write_json(out_dir / "summary.json", payload)

    print(out_dir / "stream_metrics.csv")
    print(out_dir / "joint_comparison.csv")
    if fusion_comparison_rows is not None:
        print(out_dir / "fusion_comparison.csv")
    print(out_dir / "summary.json")


if __name__ == "__main__":
    main()
