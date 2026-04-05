import argparse
import shutil
from pathlib import Path

from generate_confidence_configs import build_config
from python_locator import resolve_python
from script_utils import (
    build_main_command,
    build_stream_fusion_fieldnames,
    experiment_artifact_paths,
    extract_metric_fields,
    find_fusion_metrics,
    find_repo_root,
    has_best_artifacts,
    load_best_artifacts,
    maybe_append_resume_args,
    read_json,
    run_command,
    run_fusion as run_fusion_job,
    write_csv,
    write_yaml,
)


ROOT = find_repo_root(__file__)
PYTHON = resolve_python(ROOT)
TMP_CONFIG_DIR = ROOT / "work_dir" / "tmp_configs_43"
PAPER_TABLE_DIR = ROOT / "work_dir" / "paper_tables_wlasl100"
STREAM_ORDER = ("joint", "bone", "joint_motion", "bone_motion")

VARIANT_SPECS = {
    "baseline": {
        "model_args": {
            "use_confidence_encoding": False,
            "use_confidence_graph": False,
            "use_temporal_rectification": False,
        },
        "consistency_loss_weight": 0.0,
    },
    "node_encoding_only": {
        "model_args": {
            "use_confidence_encoding": True,
            "use_confidence_graph": False,
            "use_temporal_rectification": False,
        },
        "consistency_loss_weight": 0.0,
    },
    "graph_only": {
        "model_args": {
            "use_confidence_encoding": False,
            "use_confidence_graph": True,
            "use_temporal_rectification": False,
        },
        "consistency_loss_weight": 0.0,
    },
    "temporal_only": {
        "model_args": {
            "use_confidence_encoding": False,
            "use_confidence_graph": False,
            "use_temporal_rectification": True,
        },
        "consistency_loss_weight": 0.0,
    },
    "all_modules": {
        "model_args": {
            "use_confidence_encoding": True,
            "use_confidence_graph": True,
            "use_temporal_rectification": True,
        },
        "consistency_loss_weight": 0.1,
        "reuse_prefix": "conf_wlasl100",
    },
}

HYPERPARAM_SWEEPS = {
    "confidence_graph_lambda": [0.5, 1.0, 1.5],
    "temporal_window_size": [3, 5, 7],
    "consistency_loss_weight": [0.0, 0.05, 0.1],
}
DEFAULT_HPARAM_VALUES = {
    "confidence_graph_lambda": 1.0,
    "temporal_window_size": 5,
    "consistency_loss_weight": 0.1,
}


def make_slug(value):
    return str(value).replace(".", "_").replace("-", "_")


def run_stream(experiment_name, stream_name, config, device, num_worker, overwrite):
    artifacts = experiment_artifact_paths(ROOT, experiment_name)
    work_dir = artifacts["work_dir"]
    save_dir = artifacts["save_dir"]

    if not overwrite and has_best_artifacts(artifacts):
        return load_best_artifacts(ROOT, experiment_name)

    config_path = TMP_CONFIG_DIR / f"{experiment_name}.yaml"
    write_yaml(config_path, config)
    command = build_main_command(
        PYTHON,
        config_path=config_path,
        device=device,
        num_worker=num_worker,
        num_epoch=config["num_epoch"],
        overwrite_work_dir=overwrite,
    )
    if not overwrite:
        maybe_append_resume_args(command, save_dir)
    run_command(command, cwd=ROOT)
    return load_best_artifacts(ROOT, experiment_name)


def run_fusion(prefix, collected, window_size, overwrite):
    fusion_dir = ROOT / "work_dir" / f"{prefix}_fusion"
    metrics_path = fusion_dir / "metrics.json"
    if overwrite and fusion_dir.exists():
        shutil.rmtree(fusion_dir)
    fusion_dir.mkdir(parents=True, exist_ok=True)
    if overwrite or not metrics_path.exists():
        run_fusion_job(
            ROOT,
            PYTHON,
            label_path=ROOT / "data" / "WLASL100" / "val_label.pkl",
            data_path=ROOT / "data" / "WLASL100" / "val_data_joint.npy",
            score_paths={
                stream_name: collected[stream_name]["score_path"]
                for stream_name in STREAM_ORDER
            },
            out_dir=fusion_dir,
            window_size=window_size,
        )
    return read_json(metrics_path)


def build_row(prefix, collected, fusion_metrics, row_key, row_value):
    row = {row_key: row_value}
    for stream_name in STREAM_ORDER:
        row.update(
            extract_metric_fields(
                collected[stream_name]["metrics"],
                fields=("top1",),
                prefix=f"{stream_name}_",
            )
        )
    row.update(extract_metric_fields(fusion_metrics, prefix="fusion_"))
    row["prefix"] = prefix
    return row


def run_variant(prefix, overrides, epochs, device, num_worker, overwrite):
    reuse_prefix = overrides.get("reuse_prefix")
    if reuse_prefix:
        collected = {}
        for stream_name in STREAM_ORDER:
            collected[stream_name] = load_best_artifacts(
                ROOT,
                f"{reuse_prefix}_{stream_name}",
            )
        fusion_metrics = find_fusion_metrics(ROOT, reuse_prefix)
        if fusion_metrics is None:
            raise FileNotFoundError(f"Missing fusion metrics for {reuse_prefix}")
        return collected, fusion_metrics

    collected = {}
    for stream_name in STREAM_ORDER:
        config = build_config("WLASL100", stream_name)
        config["Experiment_name"] = f"{prefix}_{stream_name}"
        config["num_epoch"] = epochs
        config["wandb_name"] = config["Experiment_name"]
        config["consistency_loss_weight"] = overrides.get(
            "consistency_loss_weight", config.get("consistency_loss_weight", 0.0)
        )
        config["model_args"].update(overrides.get("model_args", {}))
        collected[stream_name] = run_stream(
            config["Experiment_name"],
            stream_name,
            config,
            device,
            num_worker,
            overwrite,
        )
    fusion_metrics = run_fusion(
        prefix, collected, window_size=120, overwrite=overwrite
    )
    return collected, fusion_metrics


def main():
    parser = argparse.ArgumentParser(description="Run WLASL100 section 4.3 experiments.")
    parser.add_argument(
        "--section",
        choices=["ablation", "hyperparams", "all"],
        default="all",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="0")
    parser.add_argument("--num-worker", type=int, default=0)
    parser.add_argument("--overwrite-work-dir", action="store_true")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Subset of variants or parameter names to run.",
    )
    args = parser.parse_args()

    TMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    if args.section in {"ablation", "all"}:
        ablation_rows = []
        selected_variants = (
            [name for name in VARIANT_SPECS if name in args.only]
            if args.only
            else list(VARIANT_SPECS.keys())
        )
        for variant_name in selected_variants:
            prefix = f"conf_wlasl100_43_{variant_name}"
            collected, fusion_metrics = run_variant(
                prefix,
                VARIANT_SPECS[variant_name],
                args.epochs,
                args.device,
                args.num_worker,
                args.overwrite_work_dir,
            )
            row = build_row(prefix, collected, fusion_metrics, "variant", variant_name)
            ablation_rows.append(row)
            write_csv(
                PAPER_TABLE_DIR / "table_4_3_ablation.csv",
                ablation_rows,
                build_stream_fusion_fieldnames(("variant",), include_prefix=True),
            )

    if args.section in {"hyperparams", "all"}:
        hyperparam_rows = []
        selected_parameters = (
            [name for name in HYPERPARAM_SWEEPS if name in args.only]
            if args.only
            else list(HYPERPARAM_SWEEPS.keys())
        )
        for parameter_name in selected_parameters:
            for value in HYPERPARAM_SWEEPS[parameter_name]:
                overrides = {
                    "model_args": dict(VARIANT_SPECS["all_modules"]["model_args"]),
                    "consistency_loss_weight": VARIANT_SPECS["all_modules"][
                        "consistency_loss_weight"
                    ],
                }
                if value == DEFAULT_HPARAM_VALUES[parameter_name]:
                    overrides["reuse_prefix"] = "conf_wlasl100"
                if parameter_name in overrides["model_args"]:
                    overrides["model_args"][parameter_name] = value
                else:
                    overrides[parameter_name] = value
                    if parameter_name == "consistency_loss_weight":
                        overrides["consistency_loss_weight"] = value
                prefix = (
                    f"conf_wlasl100_43_hparam_{parameter_name}_{make_slug(value)}"
                )
                collected, fusion_metrics = run_variant(
                    prefix,
                    overrides,
                    args.epochs,
                    args.device,
                    args.num_worker,
                    args.overwrite_work_dir,
                )
                row = build_row(
                    prefix, collected, fusion_metrics, "parameter", parameter_name
                )
                row["value"] = value
                hyperparam_rows.append(row)
                write_csv(
                    PAPER_TABLE_DIR / "table_4_3_hyperparams.csv",
                    hyperparam_rows,
                    build_stream_fusion_fieldnames(
                        ("parameter", "value"),
                        include_prefix=True,
                    ),
                )


if __name__ == "__main__":
    main()
