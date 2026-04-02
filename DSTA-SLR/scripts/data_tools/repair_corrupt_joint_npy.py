import argparse
import pickle
from pathlib import Path

import numpy as np
from script_utils import find_repo_root


ROOT = find_repo_root(__file__)
DATA_ROOT = ROOT / "data"


def load_label_count(label_path: Path) -> int:
    with open(label_path, "rb") as handle:
        labels = pickle.load(handle, encoding="latin1")
    if isinstance(labels, tuple) and len(labels) == 2:
        _, label_list = labels
        return len(label_list)
    return len(labels)


def repair_joint_npy(
    dataset: str,
    split: str,
    data_root: Path,
    header_offset: int,
    coord_min: float,
    coord_max: float,
    conf_min: float,
    conf_max: float,
) -> None:
    dataset_dir = data_root / dataset
    target_path = dataset_dir / f"{split}_data_joint.npy"
    label_path = dataset_dir / f"{split}_label.pkl"
    train_path = dataset_dir / "train_data_joint.npy"

    if not target_path.exists():
        raise FileNotFoundError(target_path)
    if not label_path.exists():
        raise FileNotFoundError(label_path)
    if not train_path.exists():
        raise FileNotFoundError(train_path)

    train_data = np.load(train_path, mmap_mode="r")
    sample_count = load_label_count(label_path)
    sample_shape = tuple(train_data.shape[1:])
    expected_value_count = int(sample_count * np.prod(sample_shape))

    try:
        repaired = np.load(target_path)
    except Exception:
        raw = np.fromfile(target_path, dtype=np.float32, offset=header_offset)
        if raw.size != expected_value_count:
            raise ValueError(
                f"Unexpected raw value count for {target_path}: "
                f"got {raw.size}, expected {expected_value_count}"
            )
        repaired = raw.reshape((sample_count,) + sample_shape)

    if tuple(repaired.shape) != ((sample_count,) + sample_shape):
        raise ValueError(
            f"Unexpected repaired shape for {target_path}: "
            f"got {tuple(repaired.shape)}, expected {((sample_count,) + sample_shape)}"
        )
    invalid_mask = ~np.isfinite(repaired)
    invalid_count = int(invalid_mask.sum())
    if invalid_count:
        repaired = np.nan_to_num(repaired, nan=0.0, posinf=0.0, neginf=0.0)

    coord_low_count = int((repaired[:, :2] < coord_min).sum())
    coord_high_count = int((repaired[:, :2] > coord_max).sum())
    conf_low_count = int((repaired[:, 2] < conf_min).sum())
    conf_high_count = int((repaired[:, 2] > conf_max).sum())

    repaired[:, :2] = np.clip(repaired[:, :2], coord_min, coord_max)
    repaired[:, 2] = np.clip(repaired[:, 2], conf_min, conf_max)

    backup_path = target_path.with_suffix(target_path.suffix + ".pre_repair.bak")
    if not backup_path.exists():
        target_path.replace(backup_path)
    else:
        target_path.unlink()

    np.save(target_path, repaired.astype(np.float32, copy=False))

    verified = np.load(target_path, mmap_mode="r")
    print(
        {
            "dataset": dataset,
            "split": split,
            "shape": tuple(verified.shape),
            "dtype": str(verified.dtype),
            "invalid_values_replaced": invalid_count,
            "coord_low_clipped": coord_low_count,
            "coord_high_clipped": coord_high_count,
            "conf_low_clipped": conf_low_count,
            "conf_high_clipped": conf_high_count,
            "backup": str(backup_path),
            "target": str(target_path),
        }
    )


def main():
    parser = argparse.ArgumentParser(
        description="Repair a corrupt preprocessed joint .npy by rebuilding the header and sanitizing invalid values."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--header-offset", type=int, default=128)
    parser.add_argument("--coord-min", type=float, default=-20000.0)
    parser.add_argument("--coord-max", type=float, default=50000.0)
    parser.add_argument("--conf-min", type=float, default=0.0)
    parser.add_argument("--conf-max", type=float, default=1.25)
    args = parser.parse_args()

    repair_joint_npy(
        dataset=args.dataset,
        split=args.split,
        data_root=Path(args.data_root),
        header_offset=args.header_offset,
        coord_min=args.coord_min,
        coord_max=args.coord_max,
        conf_min=args.conf_min,
        conf_max=args.conf_max,
    )


if __name__ == "__main__":
    main()
