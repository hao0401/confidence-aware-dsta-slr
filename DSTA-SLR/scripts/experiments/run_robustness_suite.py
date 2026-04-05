import argparse
import pickle
from functools import lru_cache
from pathlib import Path

from common.runtime_helpers import ensure_sys_path
from python_locator import resolve_python
from script_utils import (
    build_main_command,
    build_fieldnames_from_rows,
    find_repo_root,
    read_json,
    run_command,
    read_yaml,
    write_csv,
    write_json,
    write_yaml,
)


ROOT = find_repo_root(__file__)
PYTHON = resolve_python(ROOT)


def ensure_repo_imports() -> None:
    ensure_sys_path(ROOT)


@lru_cache(maxsize=1)
def get_compute_sample_quality():
    ensure_repo_imports()
    from feeders.feeder import compute_sample_quality

    return compute_sample_quality


@lru_cache(maxsize=1)
def get_valid_config_keys():
    ensure_repo_imports()
    from main import get_parser

    return set(vars(get_parser().parse_args([])).keys())


@lru_cache(maxsize=1)
def get_numpy():
    import numpy as np

    return np


@lru_cache(maxsize=1)
def get_torch_runtime():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    return torch, nn, DataLoader


def import_class(name):
    ensure_repo_imports()
    components = name.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def load_checkpoint(path):
    torch, _, _ = get_torch_runtime()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def latest_score_file(eval_dir):
    candidates = sorted(eval_dir.glob("epoch_*.pkl"), key=lambda item: item.stat().st_mtime)
    return candidates[-1] if candidates else None


def save_metrics(eval_dir, metrics, score_dict, epoch_tag="cpu"):
    eval_dir.mkdir(parents=True, exist_ok=True)
    write_json(eval_dir / "last_metrics.json", metrics)
    score_path = eval_dir / f"epoch_{epoch_tag}_{metrics['top1']}.pkl"
    with open(score_path, "wb") as handle:
        pickle.dump(score_dict, handle)
    return score_path


def run_eval_cpu(base_config, weights_path, experiment_name, feeder_overrides, num_worker):
    np = get_numpy()
    torch, nn, DataLoader = get_torch_runtime()
    config = dict(base_config)
    config["Experiment_name"] = experiment_name
    config["test_feeder_args"] = dict(config["test_feeder_args"])
    config["test_feeder_args"].update(feeder_overrides)
    config["test_feeder_args"].setdefault(
        "data_path", f"./data/{config['dataset']}/val_data_joint.npy"
    )
    config["test_feeder_args"].setdefault(
        "label_path", f"./data/{config['dataset']}/val_label.pkl"
    )

    feeder_cls = import_class(config["feeder"])
    dataset = feeder_cls(
        **config["test_feeder_args"],
        num_class=config["model_args"]["num_class"],
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=config.get("test_batch_size", 24),
        shuffle=False,
        num_workers=num_worker,
        drop_last=False,
    )

    model_cls = import_class(config["model"])
    model = model_cls(**config["model_args"]).to("cpu")
    criterion = nn.CrossEntropyLoss().to("cpu")
    ckpt = load_checkpoint(weights_path)
    weights = ckpt["weights"] if isinstance(ckpt, dict) and "weights" in ckpt else ckpt
    weights = {k.split("module.")[-1]: v.cpu() for k, v in weights.items()}
    model.load_state_dict(weights, strict=False)
    model.eval()

    score_frag = []
    loss_value = []
    with torch.no_grad():
        for data, label, _ in loader:
            data = data.float()
            label = label.long()
            output = model(data)
            if isinstance(output, tuple):
                output = output[0]
            loss = criterion(output, label)
            score_frag.append(output.cpu().numpy())
            loss_value.append(float(loss.cpu().numpy()))

    score = np.concatenate(score_frag)
    metrics = {
        "dataset": config["dataset"],
        "experiment_name": experiment_name,
        "epoch": int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1,
        "top1": float(dataset.top_k(score, 1)),
        "top5": float(dataset.top_k(score, 5)),
        "top1_per_class": float(dataset.per_class_acc_top_k(score, 1)),
        "top5_per_class": float(dataset.per_class_acc_top_k(score, 5)),
        "torch_top1": float(dataset.top_k(score, 1)),
        "auc": None,
        "mean_loss": float(np.mean(loss_value)),
    }
    score_dict = dict(zip(dataset.sample_name, score))
    eval_dir = ROOT / "work_dir" / experiment_name / "eval_results"
    score_path = save_metrics(eval_dir, metrics, score_dict)
    return metrics, score_path


def run_eval(base_config, weights_path, experiment_name, feeder_overrides, device, num_worker):
    if str(device).lower() == "cpu":
        return run_eval_cpu(
            base_config=base_config,
            weights_path=weights_path,
            experiment_name=experiment_name,
            feeder_overrides=feeder_overrides,
            num_worker=num_worker,
        )

    config = dict(base_config)
    config["phase"] = "test"
    config["weights"] = str(weights_path)
    config["Experiment_name"] = experiment_name
    config["overwrite_work_dir"] = True
    config["test_feeder_args"] = dict(config["test_feeder_args"])
    config["test_feeder_args"].update(feeder_overrides)
    config = {key: value for key, value in config.items() if key in get_valid_config_keys()}
    tmp_dir = ROOT / "work_dir" / "tmp_configs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    config_path = tmp_dir / f"{experiment_name}.yaml"
    write_yaml(config_path, config)
    command = build_main_command(
        PYTHON,
        config_path=config_path,
        device=device,
        num_worker=num_worker,
        overwrite_work_dir=True,
    )
    run_command(command, cwd=ROOT)
    eval_dir = ROOT / "work_dir" / experiment_name / "eval_results"
    metrics = read_json(eval_dir / "last_metrics.json")
    score_path = latest_score_file(eval_dir)
    return metrics, score_path


def bucket_accuracy(
    score_dict,
    label_path,
    data_path,
    window_size,
    bone_stream,
    motion_stream,
    confidence_mode="original",
    confidence_constant_value=1.0,
):
    np = get_numpy()
    compute_sample_quality = get_compute_sample_quality()
    with open(label_path, "rb") as handle:
        sample_names, labels = pickle.load(handle, encoding="latin1")
    labels = [int(label) for label in labels]
    data = np.load(data_path, mmap_mode="r")
    qualities = np.array(
        [
            compute_sample_quality(
                sample,
                bone_stream=bone_stream,
                motion_stream=motion_stream,
                window_size=window_size,
                confidence_mode=confidence_mode,
                confidence_constant_value=confidence_constant_value,
                sample_index=idx,
            )
            for idx, sample in enumerate(data)
        ]
    )
    q1, q2 = np.quantile(qualities, [1.0 / 3.0, 2.0 / 3.0])
    buckets = {
        "low": qualities <= q1,
        "mid": (qualities > q1) & (qualities <= q2),
        "high": qualities > q2,
    }
    rows = []
    for bucket_name, mask in buckets.items():
        indices = np.where(mask)[0]
        if len(indices) == 0:
            rows.append({"scenario": f"bucket_{bucket_name}", "top1": None, "top5": None})
            continue
        hit_top1 = 0
        hit_top5 = 0
        for idx in indices:
            score = np.asarray(score_dict[sample_names[idx]])
            rank5 = score.argsort()[-5:]
            pred = int(np.argmax(score))
            hit_top1 += int(pred == labels[idx])
            hit_top5 += int(labels[idx] in rank5)
        rows.append(
            {
                "scenario": f"bucket_{bucket_name}",
                "top1": hit_top1 / len(indices),
                "top5": hit_top5 / len(indices),
            }
        )
    return rows


def build_parser():
    parser = argparse.ArgumentParser(description="Run robustness evaluation suite.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--num-worker", type=int, default=0)
    parser.add_argument("--out-file", required=True)
    parser.add_argument(
        "--noise-stds",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 20.0],
        help="Gaussian coordinate noise stds in pixel units.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    base_config = read_yaml(args.config)
    base_name = base_config["Experiment_name"]
    dataset_name = base_config["dataset"]
    label_path = ROOT / "data" / dataset_name / "val_label.pkl"
    data_path = ROOT / "data" / dataset_name / "val_data_joint.npy"
    rows = []

    clean_metrics, score_path = run_eval(
        base_config,
        args.weights,
        f"{base_name}_robust_clean",
        {},
        args.device,
        args.num_worker,
    )
    rows.append({"scenario": "clean", **clean_metrics})

    for missing_prob in [0.1, 0.2, 0.3]:
        metrics, _ = run_eval(
            base_config,
            args.weights,
            f"{base_name}_missing_{str(missing_prob).replace('.', '_')}",
            {"missing_joint_prob": missing_prob},
            args.device,
            args.num_worker,
        )
        rows.append({"scenario": f"missing_{missing_prob:.2f}", **metrics})

    for noise_std in args.noise_stds:
        metrics, _ = run_eval(
            base_config,
            args.weights,
            f"{base_name}_noise_{str(noise_std).replace('.', '_')}",
            {"noise_std": noise_std},
            args.device,
            args.num_worker,
        )
        rows.append({"scenario": f"noise_{noise_std:g}", **metrics})

    if score_path is not None:
        with open(score_path, "rb") as handle:
            score_dict = pickle.load(handle)
        rows.extend(
            bucket_accuracy(
                score_dict=score_dict,
                label_path=label_path,
                data_path=data_path,
                window_size=base_config["test_feeder_args"]["window_size"],
                bone_stream=base_config["test_feeder_args"]["bone_stream"],
                motion_stream=base_config["test_feeder_args"]["motion_stream"],
                confidence_mode=base_config["test_feeder_args"].get(
                    "confidence_mode", "original"
                ),
                confidence_constant_value=base_config["test_feeder_args"].get(
                    "confidence_constant_value", 1.0
                ),
            )
        )

    out_path = Path(args.out_file).resolve()
    write_csv(
        out_path,
        rows,
        build_fieldnames_from_rows(rows, leading_fields=("scenario",)),
    )


if __name__ == "__main__":
    main()
