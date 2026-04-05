import argparse
from pathlib import Path

from python_locator import resolve_python
from script_utils import (
    build_main_command,
    build_fieldnames_from_rows,
    find_repo_root,
    load_best_artifacts,
    read_json,
    read_yaml,
    run_command,
    write_csv,
    write_yaml,
)


ROOT = find_repo_root(__file__)
PYTHON = resolve_python(ROOT)


def run_training(config, config_path, device, num_worker, num_epoch, overwrite):
    write_yaml(config_path, config)
    command = build_main_command(
        PYTHON,
        config_path=config_path,
        device=device,
        num_worker=num_worker,
        num_epoch=num_epoch,
        overwrite_work_dir=overwrite,
    )
    run_command(command, cwd=ROOT)
    return load_best_artifacts(ROOT, config["Experiment_name"])["metrics"]


def main():
    parser = argparse.ArgumentParser(description="Run ablation and hyperparameter sweeps.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--num-worker", type=int, default=0)
    parser.add_argument("--num-epoch", type=int, default=100)
    parser.add_argument("--overwrite-work-dir", action="store_true")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fusion-metrics")
    args = parser.parse_args()

    base_config = read_yaml(Path(args.config).resolve())
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_config_dir = ROOT / "work_dir" / "tmp_configs"
    tmp_config_dir.mkdir(parents=True, exist_ok=True)

    ablation_variants = [
        ("baseline", False, False, False, 0.0),
        ("node_encoding_only", True, False, False, 0.0),
        ("graph_only", False, True, False, 0.0),
        ("temporal_only", False, False, True, 0.0),
        ("all_modules", True, True, True, base_config.get("consistency_loss_weight", 0.1)),
    ]
    ablation_rows = []
    for name, use_encoding, use_graph, use_temporal, consistency_weight in ablation_variants:
        config = dict(base_config)
        config["model_args"] = dict(base_config["model_args"])
        config["Experiment_name"] = f"{base_config['Experiment_name']}_{name}"
        config["model_args"]["use_confidence_encoding"] = use_encoding
        config["model_args"]["use_confidence_graph"] = use_graph
        config["model_args"]["use_temporal_rectification"] = use_temporal
        config["consistency_loss_weight"] = consistency_weight
        metrics = run_training(
            config,
            tmp_config_dir / f"{config['Experiment_name']}.yaml",
            args.device,
            args.num_worker,
            args.num_epoch,
            args.overwrite_work_dir,
        )
        ablation_rows.append({"variant": name, **metrics})

    if args.fusion_metrics and Path(args.fusion_metrics).exists():
        fusion_metrics = read_json(Path(args.fusion_metrics).resolve())
        ablation_rows.append({"variant": "fusion_only", **fusion_metrics})

    write_csv(
        out_dir / "ablation.csv",
        ablation_rows,
        build_fieldnames_from_rows(ablation_rows, leading_fields=("variant",)),
    )

    hyperparam_rows = []
    sweeps = {
        "confidence_graph_lambda": [0.5, 1.0, 1.5],
        "temporal_window_size": [3, 5, 7],
        "consistency_loss_weight": [0.0, 0.05, 0.1],
    }
    for key, values in sweeps.items():
        for value in values:
            config = dict(base_config)
            config["model_args"] = dict(base_config["model_args"])
            config["Experiment_name"] = f"{base_config['Experiment_name']}_{key}_{str(value).replace('.', '_')}"
            if key in config["model_args"]:
                config["model_args"][key] = value
            else:
                config[key] = value
            metrics = run_training(
                config,
                tmp_config_dir / f"{config['Experiment_name']}.yaml",
                args.device,
                args.num_worker,
                args.num_epoch,
                args.overwrite_work_dir,
            )
            hyperparam_rows.append({"parameter": key, "value": value, **metrics})

    write_csv(
        out_dir / "hyperparams.csv",
        hyperparam_rows,
        build_fieldnames_from_rows(
            hyperparam_rows,
            leading_fields=("parameter", "value"),
        ),
    )


if __name__ == "__main__":
    main()
