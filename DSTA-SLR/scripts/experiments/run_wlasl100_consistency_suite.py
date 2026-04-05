import argparse
import shutil
from pathlib import Path

from common.runtime_helpers import maybe_reexec_with_python
from python_locator import resolve_python
from script_utils import (
    build_main_command,
    experiment_artifact_paths,
    find_repo_root,
    has_best_artifacts,
    load_best_artifacts,
    maybe_append_resume_args,
    prepare_fusion_workspace,
    read_json,
    read_yaml,
    run_command,
    run_fusion,
    write_json,
    write_yaml,
)

ROOT = find_repo_root(__file__)
PYTHON = resolve_python(ROOT)

CONFIG_DIR = ROOT / "config" / "confidence"
TMP_CONFIG_DIR = ROOT / "work_dir" / "tmp_consistency_configs"
STREAMS = ("joint", "bone", "joint_motion", "bone_motion")


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Unsupported boolean value: {value}")


def config_matches_existing(work_dir: Path, config_path: Path) -> bool:
    copied_config = work_dir / config_path.name
    if not copied_config.exists():
        return True
    return copied_config.read_text(encoding="utf-8") == config_path.read_text(
        encoding="utf-8"
    )


def patch_config(
    config,
    experiment_name,
    pred_weight,
    feat_weight,
    noise_std,
    missing_prob,
    use_reliability_weight,
    num_epoch,
):
    config = dict(config)
    config["Experiment_name"] = experiment_name
    config["wandb_name"] = experiment_name
    if num_epoch is not None:
        config["num_epoch"] = num_epoch

    # Keep the legacy field in sync for compatibility/logging.
    config["consistency_loss_weight"] = pred_weight
    config["consistency_pred_loss_weight"] = pred_weight
    config["consistency_feature_loss_weight"] = feat_weight
    config["consistency_use_reliability_weight"] = use_reliability_weight
    config["consistency_noise_std"] = noise_std
    config["consistency_missing_prob"] = missing_prob
    return config


def run_stream(config_path, device, num_worker, overwrite):
    config = read_yaml(config_path)
    experiment_name = config["Experiment_name"]
    artifacts = experiment_artifact_paths(ROOT, experiment_name)
    work_dir = artifacts["work_dir"]
    save_dir = artifacts["save_dir"]
    if work_dir.exists() and not overwrite and not config_matches_existing(work_dir, config_path):
        raise RuntimeError(
            f"Existing work_dir for {experiment_name} was created from a different config. "
            "Use --overwrite-work-dir or change --experiment-prefix."
        )
    if not overwrite and has_best_artifacts(artifacts):
        return load_best_artifacts(ROOT, experiment_name)
    command = build_main_command(
        PYTHON,
        config_path=config_path,
        device=device,
        num_worker=num_worker,
        num_epoch=config.get("num_epoch"),
        overwrite_work_dir=overwrite,
    )
    if not overwrite:
        maybe_append_resume_args(command, save_dir)
    run_command(command, cwd=ROOT)
    return load_best_artifacts(ROOT, experiment_name)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run WLASL100 four-stream consistency training and optional robustness evaluation."
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--num-worker", type=int, default=0)
    parser.add_argument("--num-epoch", type=int, default=None)
    parser.add_argument("--overwrite-work-dir", action="store_true")
    parser.add_argument("--experiment-prefix", default="conf_wlasl100_consistency")
    parser.add_argument("--pred-weight", type=float, default=0.5)
    parser.add_argument("--feat-weight", type=float, default=0.1)
    parser.add_argument("--noise-std", type=float, default=5.0)
    parser.add_argument("--missing-prob", type=float, default=0.1)
    parser.add_argument(
        "--use-reliability-weight",
        type=str2bool,
        default=True,
    )
    parser.add_argument("--run-robustness", action="store_true")
    return parser


def main(argv=None):
    if argv is None and maybe_reexec_with_python(ROOT, __file__, PYTHON):
        return

    args = build_parser().parse_args(argv)

    TMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    fusion_workspace = prepare_fusion_workspace(
        ROOT,
        args.experiment_prefix,
        stream_names=STREAMS,
        include_models=True,
    )
    fusion_output_dir = fusion_workspace["output_dir"]
    staged_score_paths = fusion_workspace["score_paths"]
    staged_model_paths = fusion_workspace["model_paths"]

    collected = {}
    for stream_name in STREAMS:
        base_config_path = CONFIG_DIR / f"wlasl100_{stream_name}.yaml"
        config = read_yaml(base_config_path)
        experiment_name = f"{args.experiment_prefix}_{stream_name}"
        patched = patch_config(
            config=config,
            experiment_name=experiment_name,
            pred_weight=args.pred_weight,
            feat_weight=args.feat_weight,
            noise_std=args.noise_std,
            missing_prob=args.missing_prob,
            use_reliability_weight=args.use_reliability_weight,
            num_epoch=args.num_epoch,
        )
        config_path = TMP_CONFIG_DIR / f"{experiment_name}.yaml"
        write_yaml(config_path, patched)
        collected[stream_name] = run_stream(
            config_path=config_path,
            device=args.device,
            num_worker=args.num_worker,
            overwrite=args.overwrite_work_dir,
        )
        shutil.copy2(
            collected[stream_name]["score_path"],
            staged_score_paths[stream_name],
        )
        shutil.copy2(
            collected[stream_name]["model_path"],
            staged_model_paths[stream_name],
        )

    run_fusion(
        ROOT,
        PYTHON,
        label_path=ROOT / "data" / "WLASL100" / "val_label.pkl",
        data_path=ROOT / "data" / "WLASL100" / "val_data_joint.npy",
        score_paths=staged_score_paths,
        out_dir=fusion_output_dir,
        window_size=120,
    )

    summary = {
        "experiment_prefix": args.experiment_prefix,
        "pred_weight": args.pred_weight,
        "feat_weight": args.feat_weight,
        "noise_std": args.noise_std,
        "missing_prob": args.missing_prob,
        "streams": {
            stream_name: info["metrics"]
            for stream_name, info in collected.items()
        },
        "fusion": read_json(fusion_output_dir / "metrics.json"),
    }

    if args.run_robustness:
        joint_config_path = TMP_CONFIG_DIR / f"{args.experiment_prefix}_joint.yaml"
        robustness_out = ROOT / "work_dir" / args.experiment_prefix / "robustness.csv"
        run_command(
            [
                PYTHON,
                "scripts/run_robustness_suite.py",
                "--config",
                str(joint_config_path),
                "--weights",
                str(collected["joint"]["model_path"]),
                "--device",
                str(args.device),
                "--num-worker",
                str(args.num_worker),
                "--out-file",
                str(robustness_out),
            ],
            cwd=ROOT,
        )
        summary["robustness_csv"] = str(robustness_out)

    summary_path = ROOT / "work_dir" / f"{args.experiment_prefix}_summary.json"
    write_json(summary_path, summary)
    print(summary_path)


if __name__ == "__main__":
    main()
