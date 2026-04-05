from __future__ import annotations

import csv
import json
import statistics
import subprocess
from pathlib import Path
from typing import Callable, Mapping, Sequence


STREAM_NAMES = ("joint", "bone", "joint_motion", "bone_motion")
DEFAULT_METRIC_FIELDS = ("top1", "top1_per_class", "top5", "top5_per_class")
QUICK_QUEUE_SUMMARY_FIELDS = (
    "dataset",
    "experiment_name",
    "epoch",
    "top1",
    "top1_per_class",
    "top5",
    "top5_per_class",
)
UNIFORM_FUSION_SUMMARY_FIELDS = (
    "variant",
    "uniform_fusion_top1",
    "uniform_fusion_pc",
    "uniform_fusion_top5",
    "uniform_fusion_top5_pc",
)
WLASL100_REPEAT_FIELDS = (
    "variant",
    "seed",
    "joint_top1",
    "fusion_top1",
    "fusion_pc",
)
WLASL100_JOINT_REPEAT_FIELDS = (
    "variant",
    "seed",
    "joint_top1",
    "joint_top5",
    "joint_pc",
)
WLASL100_JOINT_MEAN_STD_FIELDS = (
    "variant",
    "joint_top1_mean_std",
    "joint_top5_mean_std",
    "joint_pc_mean_std",
)
WLASL100_MEAN_STD_FIELDS = (
    "variant",
    "joint_top1_mean_std",
    "fusion_top1_mean_std",
    "fusion_pc_mean_std",
)
CONFIDENCE_SHIFT_STREAM_FIELDS = (
    "method",
    "prefix",
    "scenario",
    "scenario_description",
    "stream",
    "top1",
    "top5",
    "top1_per_class",
    "top5_per_class",
    "mean_loss",
)
CONFIDENCE_SIGNAL_ABLATION_FIELDS = (
    "variant",
    "joint_top1",
    "joint_top1_per_class",
    "fusion_top1",
    "fusion_top1_per_class",
)
PAPER_SINGLE_TOP1_FIELDS = ("method", "Top-1")


def find_repo_root(current_file: str | Path) -> Path:
    current_path = Path(current_file).resolve()
    for candidate in (current_path.parent, *current_path.parents):
        if (candidate / "main.py").exists() and (candidate / "scripts").exists():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {current_path}")


def run_command(command, cwd: Path, flush: bool = False) -> None:
    print(" ".join(str(part) for part in command), flush=flush)
    subprocess.run(command, cwd=cwd, check=True)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_json_if_exists(path: Path):
    if not path.exists():
        return None
    return read_json(path)


def read_yaml(path: Path):
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, content) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(content, handle, sort_keys=False)


def write_json(path: Path, content, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=indent)


def write_csv(path: Path, rows, fieldnames) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_csv_row(path: Path, row: Mapping[str, object], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def format_mean_std(
    values: Sequence[float],
    *,
    scale: float = 100.0,
    digits: int = 2,
    empty_value: str = "",
    separator: str = " +/- ",
    zero_std_for_single: bool = False,
) -> str:
    if not values:
        return empty_value
    scaled_values = [float(value) * scale for value in values]
    if len(scaled_values) == 1:
        if zero_std_for_single:
            return f"{scaled_values[0]:.{digits}f}{separator}{0:.{digits}f}"
        return f"{scaled_values[0]:.{digits}f}"
    return (
        f"{statistics.mean(scaled_values):.{digits}f}"
        f"{separator}"
        f"{statistics.stdev(scaled_values):.{digits}f}"
    )


def summarize_metric_series(
    metric_series: Mapping[str, Sequence[float]],
    *,
    scale: float = 100.0,
    digits: int = 2,
    empty_value: str = "",
    separator: str = " +/- ",
    zero_std_for_single: bool = False,
) -> dict[str, str]:
    return {
        field_name: format_mean_std(
            values,
            scale=scale,
            digits=digits,
            empty_value=empty_value,
            separator=separator,
            zero_std_for_single=zero_std_for_single,
        )
        for field_name, values in metric_series.items()
    }


def build_stream_fusion_fieldnames(
    primary_fields: Sequence[str],
    *,
    stream_names: Sequence[str] = STREAM_NAMES,
    stream_metrics: Sequence[str] = ("top1",),
    fusion_metrics: Sequence[str] = DEFAULT_METRIC_FIELDS,
    include_prefix: bool = False,
) -> list[str]:
    fieldnames = list(primary_fields)
    for stream_name in stream_names:
        for metric_name in stream_metrics:
            fieldnames.append(f"{stream_name}_{metric_name}")
    for metric_name in fusion_metrics:
        fieldnames.append(f"fusion_{metric_name}")
    if include_prefix:
        fieldnames.append("prefix")
    return fieldnames


def build_dataset_table_fieldnames(
    dataset_names: Sequence[str],
    *,
    include_per_class: bool = True,
) -> list[str]:
    fieldnames = ["method"]
    for dataset_name in dataset_names:
        fieldnames.append(f"{dataset_name}_P-I")
        if include_per_class:
            fieldnames.append(f"{dataset_name}_P-C")
    return fieldnames


def build_fieldnames_from_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    leading_fields: Sequence[str] = (),
    trailing_fields: Sequence[str] = (),
) -> list[str]:
    fieldnames = list(leading_fields)
    seen = set(fieldnames)
    trailing = list(trailing_fields)
    trailing_set = set(trailing)

    for row in rows:
        for field_name in row.keys():
            if field_name in seen or field_name in trailing_set:
                continue
            fieldnames.append(field_name)
            seen.add(field_name)

    for field_name in trailing:
        if field_name not in seen and any(field_name in row for row in rows):
            fieldnames.append(field_name)
            seen.add(field_name)
    return fieldnames


def extract_metric_fields(
    metrics: Mapping[str, object],
    *,
    fields: Sequence[str] = DEFAULT_METRIC_FIELDS,
    prefix: str = "",
    field_map: Mapping[str, str] | None = None,
    scale: float | None = None,
    round_digits: int | None = None,
    use_get: bool = False,
) -> dict[str, object]:
    row = {}
    for field in fields:
        value = metrics.get(field) if use_get else metrics[field]
        if value is not None and scale is not None:
            value = float(value) * scale
        if value is not None and round_digits is not None:
            value = round(float(value), round_digits)
        output_field = (
            field_map[field]
            if field_map is not None and field in field_map
            else f"{prefix}{field}"
        )
        row[output_field] = value
    return row


def experiment_artifact_paths(root: Path, experiment_name: str) -> dict[str, Path]:
    work_dir = root / "work_dir" / experiment_name
    eval_dir = work_dir / "eval_results"
    save_dir = work_dir / "save_models"
    return {
        "work_dir": work_dir,
        "eval_dir": eval_dir,
        "save_dir": save_dir,
        "config_path": work_dir / "config.yaml",
        "metrics_path": eval_dir / "best_metrics.json",
        "last_metrics_path": eval_dir / "last_metrics.json",
        "score_path": eval_dir / "best_acc.pkl",
        "model_path": save_dir / "best_model.pt",
    }


def has_best_artifacts(artifacts: Mapping[str, Path], *, require_model: bool = True) -> bool:
    required_keys = ["metrics_path", "score_path"]
    if require_model:
        required_keys.append("model_path")
    return all(Path(artifacts[key]).exists() for key in required_keys)


def load_best_artifacts(
    root: Path,
    experiment_name: str,
    *,
    require_model: bool = True,
) -> dict[str, object]:
    artifacts = experiment_artifact_paths(root, experiment_name)
    missing = [
        key.removesuffix("_path")
        for key in ("metrics_path", "score_path", "model_path")
        if (require_model or key != "model_path") and not artifacts[key].exists()
    ]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing best artifacts for {experiment_name}: {missing_list}"
        )

    result: dict[str, object] = dict(artifacts)
    result["metrics"] = read_json(artifacts["metrics_path"])
    return result


def find_latest_checkpoint(save_dir: Path):
    candidates = []
    for path in save_dir.glob("epoch-*.pt"):
        try:
            epoch = int(path.stem.split("-")[-1])
        except ValueError:
            continue
        candidates.append((epoch, path))
    if not candidates:
        return None, None
    return max(candidates, key=lambda item: item[0])


def build_main_command(
    python_executable: str,
    *,
    config_path: Path,
    device: str | int | None = None,
    num_worker: int | None = None,
    num_epoch: int | None = None,
    overwrite_work_dir: bool = False,
    extra_args: Sequence[str | Path | int | float] | None = None,
    unbuffered: bool = True,
) -> list[str]:
    command = [str(python_executable)]
    if unbuffered:
        command.append("-u")
    command.extend(["main.py", "--config", str(config_path)])
    if device is not None:
        command.extend(["--device", str(device)])
    if num_worker is not None:
        command.extend(["--num-worker", str(num_worker)])
    if num_epoch is not None:
        command.extend(["--num-epoch", str(num_epoch)])
    if overwrite_work_dir:
        command.extend(["--overwrite-work-dir", "true"])
    if extra_args:
        command.extend(str(arg) for arg in extra_args)
    return command


def maybe_append_resume_args(
    command: list[str],
    save_dir: Path,
    *,
    max_epoch: int | None = None,
):
    latest_epoch, latest_ckpt = find_latest_checkpoint(save_dir)
    if latest_ckpt is None:
        return latest_epoch, latest_ckpt

    next_epoch = latest_epoch + 1
    if max_epoch is not None and next_epoch > max_epoch:
        return latest_epoch, latest_ckpt

    command.extend(
        [
            "--weights",
            str(latest_ckpt),
            "--start-epoch",
            str(next_epoch),
        ]
    )
    return latest_epoch, latest_ckpt


def prepare_fusion_workspace(
    root: Path,
    prefix: str,
    *,
    stream_names: Sequence[str] = STREAM_NAMES,
    include_models: bool = False,
    input_suffix: str = "_fusion_inputs",
    output_suffix: str = "_fusion_results",
):
    input_dir = root / "work_dir" / f"{prefix}{input_suffix}"
    output_dir = root / "work_dir" / f"{prefix}{output_suffix}"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "score_paths": {
            stream_name: input_dir / f"best_acc_{stream_name}.pkl"
            for stream_name in stream_names
        },
    }
    if include_models:
        result["model_paths"] = {
            stream_name: input_dir / f"best_model_{stream_name}.pt"
            for stream_name in stream_names
        }
    return result


def safe_ratio(value, reference):
    if value is None or reference in (None, 0, 0.0):
        return None
    return value / reference


def build_comparison_fieldnames(
    left_label: str,
    right_label: str,
    *,
    metric_names: Sequence[str] = ("top1", "top5"),
    retention_metric: str = "top1",
    retention_field_names: Mapping[str, str] | None = None,
    scenario_key: str = "scenario",
) -> list[str]:
    retention_field_names = retention_field_names or {
        left_label: f"{left_label}_{retention_metric}_retention",
        right_label: f"{right_label}_{retention_metric}_retention",
    }
    fieldnames = [scenario_key]
    for metric_name in metric_names:
        fieldnames.extend(
            [
                f"{left_label}_{metric_name}",
                f"{right_label}_{metric_name}",
                f"{metric_name}_gain",
            ]
        )
        if metric_name == retention_metric:
            fieldnames.extend(
                [
                    retention_field_names[left_label],
                    retention_field_names[right_label],
                ]
            )
    return fieldnames


def build_metric_comparison_rows(
    left_by_scenario: Mapping[str, Mapping[str, object] | None],
    right_by_scenario: Mapping[str, Mapping[str, object] | None],
    *,
    left_label: str,
    right_label: str,
    scenario_names: Sequence[str] | None = None,
    metric_names: Sequence[str] = ("top1", "top5"),
    retention_metric: str = "top1",
    retention_field_names: Mapping[str, str] | None = None,
    value_transform: Callable[[object], object] | None = None,
    scenario_key: str = "scenario",
    clean_scenario_name: str = "clean",
) -> list[dict[str, object]]:
    def metric_value(metrics: Mapping[str, object] | None, metric_name: str):
        if metrics is None:
            return None
        value = metrics.get(metric_name)
        if value_transform is not None and value is not None:
            value = value_transform(value)
        return value

    if scenario_names is None:
        scenario_names = list(
            dict.fromkeys([*left_by_scenario.keys(), *right_by_scenario.keys()])
        )
    retention_field_names = retention_field_names or {
        left_label: f"{left_label}_{retention_metric}_retention",
        right_label: f"{right_label}_{retention_metric}_retention",
    }
    clean_left = metric_value(left_by_scenario.get(clean_scenario_name), retention_metric)
    clean_right = metric_value(
        right_by_scenario.get(clean_scenario_name), retention_metric
    )

    rows = []
    for scenario_name in scenario_names:
        left_metrics = left_by_scenario.get(scenario_name)
        right_metrics = right_by_scenario.get(scenario_name)
        row = {scenario_key: scenario_name}
        for metric_name in metric_names:
            left_value = metric_value(left_metrics, metric_name)
            right_value = metric_value(right_metrics, metric_name)
            row[f"{left_label}_{metric_name}"] = left_value
            row[f"{right_label}_{metric_name}"] = right_value
            row[f"{metric_name}_gain"] = (
                None
                if left_value is None or right_value is None
                else right_value - left_value
            )
            if metric_name == retention_metric:
                row[retention_field_names[left_label]] = safe_ratio(
                    left_value, clean_left
                )
                row[retention_field_names[right_label]] = safe_ratio(
                    right_value, clean_right
                )
        rows.append(row)
    return rows


def find_fusion_metrics(
    root: Path,
    prefix: str,
    suffixes: Sequence[str] = ("_fusion_results", "_fusion"),
):
    for suffix in suffixes:
        metrics = read_json_if_exists(root / "work_dir" / f"{prefix}{suffix}" / "metrics.json")
        if metrics is not None:
            return metrics
    return None


def run_fusion(
    root: Path,
    python_executable: str,
    *,
    label_path: Path,
    score_paths: Mapping[str, Path],
    out_dir: Path,
    data_path: Path | None = None,
    window_size: int | None = None,
    extra_args: Sequence[str | Path] | None = None,
    script_path: str = "ensemble/fuse_streams.py",
    stream_names: Sequence[str] = STREAM_NAMES,
) -> None:
    command = [str(python_executable), script_path, "--label-path", str(label_path)]
    if data_path is not None:
        command.extend(["--data-path", str(data_path)])
    for stream_name in stream_names:
        if stream_name not in score_paths:
            raise KeyError(f"Missing score path for stream {stream_name!r}")
        command.extend(
            [f"--{stream_name.replace('_', '-')}", str(score_paths[stream_name])]
        )
    if window_size is not None:
        command.extend(["--window-size", str(window_size)])
    command.extend(["--out-dir", str(out_dir)])
    if extra_args:
        command.extend(str(arg) for arg in extra_args)
    run_command(command, cwd=root)


__all__ = [
    "STREAM_NAMES",
    "DEFAULT_METRIC_FIELDS",
    "QUICK_QUEUE_SUMMARY_FIELDS",
    "UNIFORM_FUSION_SUMMARY_FIELDS",
    "WLASL100_REPEAT_FIELDS",
    "WLASL100_JOINT_REPEAT_FIELDS",
    "WLASL100_JOINT_MEAN_STD_FIELDS",
    "WLASL100_MEAN_STD_FIELDS",
    "CONFIDENCE_SHIFT_STREAM_FIELDS",
    "CONFIDENCE_SIGNAL_ABLATION_FIELDS",
    "PAPER_SINGLE_TOP1_FIELDS",
    "find_repo_root",
    "run_command",
    "read_json",
    "read_json_if_exists",
    "read_yaml",
    "write_json",
    "write_yaml",
    "write_csv",
    "append_csv_row",
    "format_mean_std",
    "summarize_metric_series",
    "build_stream_fusion_fieldnames",
    "build_dataset_table_fieldnames",
    "build_fieldnames_from_rows",
    "extract_metric_fields",
    "experiment_artifact_paths",
    "has_best_artifacts",
    "load_best_artifacts",
    "find_latest_checkpoint",
    "build_main_command",
    "maybe_append_resume_args",
    "prepare_fusion_workspace",
    "safe_ratio",
    "build_comparison_fieldnames",
    "build_metric_comparison_rows",
    "find_fusion_metrics",
    "run_fusion",
]
