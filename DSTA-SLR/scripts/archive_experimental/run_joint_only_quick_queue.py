import argparse
import json
import re
from pathlib import Path

try:
    from ._bootstrap import load_common_module
except ImportError:
    from _bootstrap import load_common_module

resolve_python = load_common_module("python_locator").resolve_python
_SCRIPT_UTILS = load_common_module("script_utils")
append_csv_row = _SCRIPT_UTILS.append_csv_row
build_main_command = _SCRIPT_UTILS.build_main_command
extract_metric_fields = _SCRIPT_UTILS.extract_metric_fields
find_repo_root = _SCRIPT_UTILS.find_repo_root
QUICK_QUEUE_SUMMARY_FIELDS = _SCRIPT_UTILS.QUICK_QUEUE_SUMMARY_FIELDS
run_command = _SCRIPT_UTILS.run_command


ROOT = find_repo_root(__file__)
CONFIG_DIR = ROOT / "config" / "confidence"
WORK_DIR = ROOT / "work_dir"
PYTHON = resolve_python(ROOT)


def dataset_key(dataset: str) -> str:
    return dataset.lower().replace("-", "_")


def rewrite_config(source: Path, target: Path, experiment_name: str, num_epoch: int) -> None:
    text = source.read_text(encoding="utf-8")
    replacements = {
        r"^Experiment_name:\s*.*$": f"Experiment_name: {experiment_name}",
        r"^wandb_name:\s*.*$": f"wandb_name: {experiment_name}",
        r"^num_epoch:\s*.*$": f"num_epoch: {num_epoch}",
        r"^overwrite_work_dir:\s*.*$": "overwrite_work_dir: true",
    }
    for pattern, replacement in replacements.items():
        text, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
        if count == 0:
            text = text.rstrip() + "\n" + replacement + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def load_best_metrics(experiment_name: str) -> dict:
    metrics_path = WORK_DIR / experiment_name / "eval_results" / "best_metrics.json"
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def append_summary(summary_path: Path, row: dict) -> None:
    append_csv_row(
        summary_path,
        row,
        QUICK_QUEUE_SUMMARY_FIELDS,
    )


def run_dataset(dataset: str, num_epoch: int, device: str, num_worker: int, prefix: str, summary_path: Path) -> None:
    key = dataset_key(dataset)
    source_config = CONFIG_DIR / f"{key}_joint.yaml"
    experiment_name = f"{prefix}_{key}_joint"
    temp_config = WORK_DIR / "tmp_joint_only_quick_configs" / f"{experiment_name}.yaml"

    rewrite_config(source_config, temp_config, experiment_name, num_epoch)
    command = build_main_command(
        PYTHON,
        config_path=temp_config,
        device=device,
        num_worker=num_worker,
        num_epoch=num_epoch,
        overwrite_work_dir=True,
    )
    run_command(
        command,
        cwd=ROOT,
        flush=True,
    )
    metrics = load_best_metrics(experiment_name)
    row = {
        "dataset": dataset,
        "experiment_name": experiment_name,
        "epoch": metrics.get("epoch"),
        **extract_metric_fields(metrics, use_get=True),
    }
    append_summary(summary_path, row)
    print(
        f"[done] {dataset}: top1={row['top1']:.4f}, top1_per_class={row['top1_per_class']:.4f}, epoch={row['epoch']}",
        flush=True,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Run joint-only quick pilots sequentially.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["SLR500", "MSASL500", "MSASL1000"],
    )
    parser.add_argument("--num-epoch", type=int, default=5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--num-worker", type=int, default=0)
    parser.add_argument("--prefix", default="fast5")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary_path = WORK_DIR / "joint_only_quick_results" / f"{args.prefix}_summary.csv"
    for dataset in args.datasets:
        run_dataset(dataset, args.num_epoch, args.device, args.num_worker, args.prefix, summary_path)


if __name__ == "__main__":
    main()
