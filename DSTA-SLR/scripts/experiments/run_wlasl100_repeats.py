import argparse
from pathlib import Path

from generate_confidence_configs import build_config
from python_locator import resolve_python
from script_utils import (
    build_main_command,
    experiment_artifact_paths,
    extract_metric_fields,
    find_repo_root,
    load_best_artifacts,
    maybe_append_resume_args,
    read_json,
    run_command,
    run_fusion,
    WLASL100_REPEAT_FIELDS,
    write_csv,
    write_yaml,
)


ROOT = find_repo_root(__file__)
PYTHON = resolve_python(ROOT)
TMP_CONFIG_DIR = ROOT / "work_dir" / "tmp_configs_repeats"
STREAMS = ("joint", "bone", "joint_motion", "bone_motion")

VARIANTS = {
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
    },
}
def run_seed_variant(variant_name, seed, epochs, overwrite, num_worker):
    overrides = VARIANTS[variant_name]
    collected = {}
    TMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for stream_name in STREAMS:
        config = build_config("WLASL100", stream_name)
        config["Experiment_name"] = (
            f"conf_wlasl100_43_{variant_name}_seed{seed}_{stream_name}"
        )
        config["seed"] = seed
        config["num_epoch"] = epochs
        config["num_worker"] = num_worker
        config["wandb_name"] = config["Experiment_name"]
        config["consistency_loss_weight"] = overrides["consistency_loss_weight"]
        config["model_args"].update(overrides["model_args"])

        config_path = TMP_CONFIG_DIR / f"{config['Experiment_name']}.yaml"
        write_yaml(config_path, config)
        command = build_main_command(
            PYTHON,
            config_path=config_path,
            num_worker=num_worker,
            num_epoch=epochs,
            overwrite_work_dir=overwrite,
            extra_args=["--seed", str(seed)],
        )
        if not overwrite:
            save_dir = experiment_artifact_paths(ROOT, config["Experiment_name"])["save_dir"]
            maybe_append_resume_args(command, save_dir)
        run_command(command, cwd=ROOT)

        collected[stream_name] = load_best_artifacts(ROOT, config["Experiment_name"])

    fusion_dir = ROOT / "work_dir" / f"conf_wlasl100_43_{variant_name}_seed{seed}_fusion"
    run_fusion(
        ROOT,
        PYTHON,
        label_path=ROOT / "data" / "WLASL100" / "val_label.pkl",
        data_path=ROOT / "data" / "WLASL100" / "val_data_joint.npy",
        score_paths={
            stream_name: collected[stream_name]["score_path"]
            for stream_name in STREAMS
        },
        out_dir=fusion_dir,
        window_size=120,
    )
    fusion_metrics = read_json(fusion_dir / "metrics.json")
    return {
        "variant": variant_name,
        "seed": seed,
        **extract_metric_fields(
            collected["joint"]["metrics"],
            fields=("top1",),
            prefix="joint_",
        ),
        **extract_metric_fields(
            fusion_metrics,
            field_map={
                "top1": "fusion_top1",
                "top1_per_class": "fusion_pc",
            },
            fields=("top1", "top1_per_class"),
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run repeated WLASL100 experiments for mean +/- std."
    )
    parser.add_argument("--variant", nargs="+", choices=list(VARIANTS.keys()), required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-worker", type=int, default=0)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=5)
    parser.add_argument("--overwrite-work-dir", action="store_true")
    parser.add_argument(
        "--out-file",
        default=str(ROOT / "work_dir" / "paper_tables_wlasl100" / "wlasl100_repeat_runs.csv"),
    )
    args = parser.parse_args()
    if args.seed_start > args.seed_end:
        raise ValueError("--seed-start must be <= --seed-end")

    rows = []
    for variant_name in args.variant:
        for seed in range(args.seed_start, args.seed_end + 1):
            rows.append(
                run_seed_variant(
                    variant_name,
                    seed,
                    args.epochs,
                    args.overwrite_work_dir,
                    args.num_worker,
                )
            )

    out_path = Path(args.out_file).resolve()
    write_csv(
        out_path,
        rows,
        WLASL100_REPEAT_FIELDS,
    )


if __name__ == "__main__":
    main()
