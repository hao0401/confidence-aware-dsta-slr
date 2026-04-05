import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feeders.feeder import compute_sample_quality


STREAM_FLAGS = {
    "joint": {"bone_stream": False, "motion_stream": False},
    "bone": {"bone_stream": True, "motion_stream": False},
    "joint_motion": {"bone_stream": False, "motion_stream": True},
    "bone_motion": {"bone_stream": True, "motion_stream": True},
}


def load_scores(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def per_class_top_k(score_matrix, labels, num_class, k):
    rank = score_matrix.argsort()
    hit_top_k = [label in rank[i, -k:] for i, label in enumerate(labels)]
    acc = [0.0 for _ in range(num_class)]
    for c in range(num_class):
        hit_label = [label == c for label in labels]
        class_count = max(labels.count(c), 1)
        acc[c] = np.sum(
            np.array(hit_top_k, dtype=np.float32)
            * np.array(hit_label, dtype=np.float32)
        ) / class_count
    return float(np.mean(acc))


def compute_stream_weights(
    data,
    window_size,
    confidence_mode="original",
    confidence_constant_value=1.0,
    confidence_transform="identity",
    confidence_transform_power=2.0,
    confidence_transform_threshold=0.5,
):
    weights = {key: [] for key in STREAM_FLAGS}
    for sample_idx, sample in enumerate(data):
        for stream_name, flags in STREAM_FLAGS.items():
            weights[stream_name].append(
                compute_sample_quality(
                    sample,
                    bone_stream=flags["bone_stream"],
                    motion_stream=flags["motion_stream"],
                    window_size=window_size,
                    confidence_mode=confidence_mode,
                    confidence_constant_value=confidence_constant_value,
                    confidence_transform=confidence_transform,
                    confidence_transform_power=confidence_transform_power,
                    confidence_transform_threshold=confidence_transform_threshold,
                    sample_index=sample_idx,
                )
            )
    return {key: np.array(value, dtype=np.float32) for key, value in weights.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Fuse four stream score files with confidence-aware weights."
    )
    parser.add_argument("--label-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--joint", required=True)
    parser.add_argument("--bone", required=True)
    parser.add_argument("--joint-motion", required=True)
    parser.add_argument("--bone-motion", required=True)
    parser.add_argument("--window-size", type=int, default=120)
    parser.add_argument(
        "--confidence-mode",
        choices=["original", "constant", "shuffle"],
        default="original",
    )
    parser.add_argument(
        "--confidence-constant-value", type=float, default=1.0
    )
    parser.add_argument(
        "--confidence-transform",
        choices=["identity", "square", "sqrt", "power", "rank", "binary"],
        default="identity",
    )
    parser.add_argument(
        "--confidence-transform-power",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--confidence-transform-threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.label_path, "rb") as handle:
        sample_names, label_list = pickle.load(handle, encoding="latin1")
    val_data = np.load(args.data_path, mmap_mode="r")

    label_count = len(label_list)
    data_count = int(val_data.shape[0])
    if label_count != data_count:
        aligned_count = min(label_count, data_count)
        print(
            f"[fuse_streams] Mismatched sample count for {args.data_path} / {args.label_path}: "
            f"data={data_count}, labels={label_count}. Trimming to {aligned_count}."
        )
        sample_names = sample_names[:aligned_count]
        label_list = label_list[:aligned_count]
        val_data = val_data[:aligned_count]

    labels = [int(label) for label in label_list]
    num_class = int(max(labels)) + 1

    stream_scores = {
        "joint": load_scores(args.joint),
        "bone": load_scores(args.bone),
        "joint_motion": load_scores(args.joint_motion),
        "bone_motion": load_scores(args.bone_motion),
    }
    stream_weights = compute_stream_weights(
        val_data,
        args.window_size,
        confidence_mode=args.confidence_mode,
        confidence_constant_value=args.confidence_constant_value,
        confidence_transform=args.confidence_transform,
        confidence_transform_power=args.confidence_transform_power,
        confidence_transform_threshold=args.confidence_transform_threshold,
    )

    fused_scores = []
    prediction_rows = []
    weight_rows = []
    right_num = 0
    right_num_5 = 0

    for idx, sample_name in enumerate(sample_names):
        per_stream_scores = []
        per_stream_weights = []
        for stream_name in STREAM_FLAGS:
            stream_score = np.asarray(stream_scores[stream_name][sample_name])
            per_stream_scores.append(stream_score)
            per_stream_weights.append(float(stream_weights[stream_name][idx]))
        weight_vector = np.array(per_stream_weights, dtype=np.float32)
        weight_vector = weight_vector / max(weight_vector.sum(), 1e-6)
        fused_score = sum(
            score * weight for score, weight in zip(per_stream_scores, weight_vector)
        )
        rank_5 = fused_score.argsort()[-5:]
        pred = int(np.argmax(fused_score))
        label = labels[idx]
        right_num += int(pred == label)
        right_num_5 += int(label in rank_5)
        fused_scores.append(fused_score)
        prediction_rows.append((sample_name, pred))
        weight_rows.append((sample_name, *weight_vector.tolist()))

    score_matrix = np.stack(fused_scores)
    top1 = right_num / len(sample_names)
    top5 = right_num_5 / len(sample_names)
    top1_per_class = per_class_top_k(score_matrix, labels, num_class, 1)
    top5_per_class = per_class_top_k(score_matrix, labels, num_class, 5)

    with open(out_dir / "predictions.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_name", "prediction"])
        writer.writerows(prediction_rows)
    with open(out_dir / "quality_weights.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["sample_name", "joint_weight", "bone_weight", "joint_motion_weight", "bone_motion_weight"]
        )
        writer.writerows(weight_rows)
    with open(out_dir / "fused_scores.pkl", "wb") as handle:
        pickle.dump(dict(zip(sample_names, score_matrix)), handle)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "top1": top1,
                "top5": top5,
                "top1_per_class": top1_per_class,
                "top5_per_class": top5_per_class,
            },
            handle,
            indent=2,
        )

    print(f"top1: {top1:.6f}")
    print(f"top1_per_class: {top1_per_class:.6f}")
    print(f"top5: {top5:.6f}")
    print(f"top5_per_class: {top5_per_class:.6f}")


if __name__ == "__main__":
    main()
