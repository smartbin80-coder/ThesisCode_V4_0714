"""Create train/validation/test splits by trajectory id."""

import argparse
import csv
from pathlib import Path

import numpy as np


def read_trajectory_ids(index_path):
    """Read trajectory ids from dataset_index.csv."""
    with open(index_path, "r", encoding="utf-8") as f:
        return [row["trajectory_id"] for row in csv.DictReader(f)]


def write_ids(path, ids):
    """Write one trajectory id per line."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in ids:
            f.write(f"{item}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Split trajectories without temporal leakage.")
    parser.add_argument("--index", type=str, default="dataset/dataset_index.csv")
    parser.add_argument("--output-dir", type=str, default="dataset/splits")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    ids = sorted(set(read_trajectory_ids(args.index)))
    rng = np.random.default_rng(args.seed)
    ids = list(rng.permutation(ids))
    n_train = int(round(len(ids) * args.train_ratio))
    n_val = int(round(len(ids) * args.val_ratio))
    train = ids[:n_train]
    val = ids[n_train:n_train + n_val]
    test = ids[n_train + n_val:]
    out = Path(args.output_dir)
    write_ids(out / "train_trajectories.txt", train)
    write_ids(out / "val_trajectories.txt", val)
    write_ids(out / "test_trajectories.txt", test)
    print(f"Split written to {out}: train={len(train)}, val={len(val)}, test={len(test)}")


if __name__ == "__main__":
    main()
