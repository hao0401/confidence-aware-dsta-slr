import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np


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


def main():
    parser = argparse.ArgumentParser(
        description="Fuse four stream score files with uniform averaging."
    )
    parser.add_argument("--label-path", required=True)
    parser.add_argument("--joint", required=True)
    parser.add_argument("--bone", required=True)
    parser.add_argument("--joint-motion", required=True)
    parser.add_argument("--bone-motion", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.label_path, "rb") as handle:
        sample_names, label_list = pickle.load(handle, encoding="latin1")
    labels = [int(label) for label in label_list]
    num_class = int(max(labels)) + 1

    stream_scores = {
        "joint": load_scores(args.joint),
        "bone": load_scores(args.bone),
        "joint_motion": load_scores(args.joint_motion),
        "bone_motion": load_scores(args.bone_motion),
    }

    fused_scores = []
    prediction_rows = []
    right_num = 0
    right_num_5 = 0

    for sample_name, label in zip(sample_names, labels):
        per_stream_scores = [
            np.asarray(stream_scores["joint"][sample_name]),
            np.asarray(stream_scores["bone"][sample_name]),
            np.asarray(stream_scores["joint_motion"][sample_name]),
            np.asarray(stream_scores["bone_motion"][sample_name]),
        ]
        fused_score = sum(per_stream_scores) / float(len(per_stream_scores))
        rank_5 = fused_score.argsort()[-5:]
        pred = int(np.argmax(fused_score))
        right_num += int(pred == label)
        right_num_5 += int(label in rank_5)
        fused_scores.append(fused_score)
        prediction_rows.append((sample_name, pred))

    score_matrix = np.stack(fused_scores)
    top1 = right_num / len(sample_names)
    top5 = right_num_5 / len(sample_names)
    top1_per_class = per_class_top_k(score_matrix, labels, num_class, 1)
    top5_per_class = per_class_top_k(score_matrix, labels, num_class, 5)

    with open(out_dir / "predictions.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_name", "prediction"])
        writer.writerows(prediction_rows)
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
