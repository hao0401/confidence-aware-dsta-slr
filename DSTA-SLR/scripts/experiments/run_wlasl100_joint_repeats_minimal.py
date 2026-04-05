import argparse
import pickle
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
    run_command,
    summarize_metric_series,
    WLASL100_JOINT_MEAN_STD_FIELDS,
    WLASL100_JOINT_REPEAT_FIELDS,
    write_csv,
    write_yaml,
)


ROOT = find_repo_root(__file__)
PYTHON = resolve_python(ROOT)
TMP_CONFIG_DIR = ROOT / "work_dir" / "tmp_configs_joint_repeats"

VARIANTS = {
    "baseline": {
        "model_args": {
            "use_confidence_encoding": False,
            "use_confidence_graph": False,
            "use_temporal_rectification": False,
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


def per_class_top_k(score_matrix, labels, num_class, k):
    rank = score_matrix.argsort()
    hit_top_k = [label in rank[i, -k:] for i, label in enumerate(labels)]
    acc = [0.0 for _ in range(num_class)]
    for c in range(num_class):
        hit_label = [label == c for label in labels]
        class_count = max(labels.count(c), 1)
        acc[c] = sum(
            float(hit) * float(is_label)
            for hit, is_label in zip(hit_top_k, hit_label)
        ) / class_count
    return float(sum(acc) / len(acc))


def find_score_file_for_epoch(eval_dir: Path, epoch_index: int):
    candidates = sorted(eval_dir.glob(f"epoch_{epoch_index}_*.pkl"))
    return candidates[-1] if candidates else None


def metrics_from_score_file(score_path: Path, label_path: Path):
    with score_path.open("rb") as handle:
        score_dict = pickle.load(handle)
    with label_path.open("rb") as handle:
        sample_names, label_list = pickle.load(handle, encoding="latin1")
    labels = [int(label) for label in label_list]

    import numpy as np

    score_matrix = np.stack([score_dict[name] for name in sample_names])
    top1 = float(
        sum(int(int(score.argmax()) == label) for score, label in zip(score_matrix, labels))
        / len(labels)
    )
    top5 = float(
        sum(int(label in score.argsort()[-5:]) for score, label in zip(score_matrix, labels))
        / len(labels)
    )
    num_class = int(max(labels)) + 1
    top1_per_class = per_class_top_k(score_matrix, labels, num_class, 1)
    return {"top1": top1, "top5": top5, "top1_per_class": top1_per_class}


def run_seed_variant(variant_name, seed, epochs, num_worker):
    overrides = VARIANTS[variant_name]
    TMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    config = build_config("WLASL100", "joint")
    config["Experiment_name"] = f"conf_wlasl100_43_{variant_name}_seed{seed}_joint"
    config["seed"] = seed
    config["num_epoch"] = epochs
    config["num_worker"] = num_worker
    config["wandb_name"] = config["Experiment_name"]
    config["consistency_loss_weight"] = overrides["consistency_loss_weight"]
    config["model_args"].update(overrides["model_args"])

    work_dir = ROOT / "work_dir" / config["Experiment_name"]
    eval_dir = work_dir / "eval_results"
    score_path = find_score_file_for_epoch(eval_dir, epochs - 1) if eval_dir.exists() else None
    if score_path is not None:
        metrics = metrics_from_score_file(
            score_path=score_path,
            label_path=ROOT / "data" / "WLASL100" / "val_label.pkl",
        )
        return {
            "variant": variant_name,
            "seed": seed,
            **extract_metric_fields(
                metrics,
                field_map={
                    "top1": "joint_top1",
                    "top5": "joint_top5",
                    "top1_per_class": "joint_pc",
                },
                fields=("top1", "top5", "top1_per_class"),
            ),
        }

    config_path = TMP_CONFIG_DIR / f"{config['Experiment_name']}.yaml"
    write_yaml(config_path, config)

    command = build_main_command(
        PYTHON,
        config_path=config_path,
        num_worker=num_worker,
        num_epoch=epochs,
        extra_args=["--seed", str(seed)],
    )

    save_dir = experiment_artifact_paths(ROOT, config["Experiment_name"])["save_dir"]
    maybe_append_resume_args(command, save_dir, max_epoch=epochs)

    run_command(command, cwd=ROOT)

    metrics = load_best_artifacts(ROOT, config["Experiment_name"])["metrics"]
    return {
        "variant": variant_name,
        "seed": seed,
        **extract_metric_fields(
            metrics,
            field_map={
                "top1": "joint_top1",
                "top5": "joint_top5",
                "top1_per_class": "joint_pc",
            },
            fields=("top1", "top5", "top1_per_class"),
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run minimal joint-stream repeats for baseline and all_modules."
    )
    parser.add_argument(
        "--variant",
        nargs="+",
        choices=list(VARIANTS.keys()),
        default=["baseline", "all_modules"],
    )
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--num-worker", type=int, default=0)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=3)
    parser.add_argument(
        "--out-file",
        default=str(ROOT / "work_dir" / "paper_tables_wlasl100" / "wlasl100_joint_repeat_runs.csv"),
    )
    parser.add_argument(
        "--summary-file",
        default=str(ROOT / "work_dir" / "paper_tables_wlasl100" / "wlasl100_joint_mean_std.csv"),
    )
    args = parser.parse_args()

    rows = []
    summary_rows = []
    for variant_name in args.variant:
        variant_rows = []
        for seed in range(args.seed_start, args.seed_end + 1):
            row = run_seed_variant(variant_name, seed, args.epochs, args.num_worker)
            rows.append(row)
            variant_rows.append(row)
        summary_row = {"variant": variant_name}
        summary_row.update(
            summarize_metric_series(
                {
                    "joint_top1_mean_std": [row["joint_top1"] for row in variant_rows],
                    "joint_top5_mean_std": [row["joint_top5"] for row in variant_rows],
                    "joint_pc_mean_std": [row["joint_pc"] for row in variant_rows],
                },
                separator=" +/- ",
                zero_std_for_single=True,
            )
        )
        summary_rows.append(summary_row)

    out_path = Path(args.out_file).resolve()
    write_csv(
        out_path,
        rows,
        WLASL100_JOINT_REPEAT_FIELDS,
    )

    summary_path = Path(args.summary_file).resolve()
    write_csv(
        summary_path,
        summary_rows,
        WLASL100_JOINT_MEAN_STD_FIELDS,
    )

    print(out_path)
    print(summary_path)


if __name__ == "__main__":
    main()
