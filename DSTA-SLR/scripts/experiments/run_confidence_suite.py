import argparse
import shutil
from pathlib import Path

from experiment_specs import STREAM_SPECS
from python_locator import resolve_python
from script_utils import (
    build_main_command,
    experiment_artifact_paths,
    find_repo_root,
    load_best_artifacts,
    maybe_append_resume_args,
    prepare_fusion_workspace,
    run_command,
    run_fusion,
)


ROOT = find_repo_root(__file__)
CONFIG_DIR = ROOT / "config" / "confidence"
PYTHON = resolve_python(ROOT)


def main():
    parser = argparse.ArgumentParser(description="Run confidence-aware four-stream training and fusion.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--num-worker", type=int, default=0)
    parser.add_argument("--num-epoch", type=int, default=None)
    parser.add_argument("--overwrite-work-dir", action="store_true")
    parser.add_argument(
        "--config-dir",
        default=str(CONFIG_DIR),
        help="Directory containing per-stream YAML configs.",
    )
    parser.add_argument(
        "--experiment-prefix",
        default="conf",
        help="Prefix used for work_dir experiment names and fusion result directories.",
    )
    args = parser.parse_args()

    dataset_key = args.dataset.lower().replace("-", "_")
    config_dir = Path(args.config_dir)
    fusion_base = f"{args.experiment_prefix}_{dataset_key}" if args.experiment_prefix else dataset_key
    fusion_workspace = prepare_fusion_workspace(
        ROOT,
        fusion_base,
        stream_names=tuple(STREAM_SPECS),
        include_models=True,
    )
    fusion_output_dir = fusion_workspace["output_dir"]
    staged_score_paths = fusion_workspace["score_paths"]
    staged_model_paths = fusion_workspace["model_paths"]

    for stream_name in STREAM_SPECS:
        config_path = config_dir / f"{dataset_key}_{stream_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config: {config_path}")
        experiment_name = (
            f"{args.experiment_prefix}_{dataset_key}_{stream_name}"
            if args.experiment_prefix
            else f"{dataset_key}_{stream_name}"
        )
        artifacts = experiment_artifact_paths(ROOT, experiment_name)
        save_dir = artifacts["save_dir"]
        command = build_main_command(
            PYTHON,
            config_path=config_path,
            device=args.device,
            num_worker=args.num_worker,
            num_epoch=args.num_epoch,
            overwrite_work_dir=args.overwrite_work_dir,
        )
        if not args.overwrite_work_dir:
            maybe_append_resume_args(command, save_dir)
        run_command(command, cwd=ROOT)
        artifacts = load_best_artifacts(ROOT, experiment_name)
        shutil.copy2(
            artifacts["score_path"],
            staged_score_paths[stream_name],
        )
        shutil.copy2(
            artifacts["model_path"],
            staged_model_paths[stream_name],
        )

    run_fusion(
        ROOT,
        PYTHON,
        label_path=ROOT / "data" / args.dataset / "val_label.pkl",
        data_path=ROOT / "data" / args.dataset / "val_data_joint.npy",
        score_paths=staged_score_paths,
        out_dir=fusion_output_dir,
    )


if __name__ == "__main__":
    main()
