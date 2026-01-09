from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from problem_generation import generate_dataset_instances, write_dataset

NUM_INSTANCES = 100


def parse_args():
    parser = argparse.ArgumentParser(description="Run scheduler.")
    parser.add_argument("--seed", type=int, help="Seed for random instance generation.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of algorithm iterations.",
    )
    parser.add_argument(
        "--K-min",
        dest="K_min",
        type=int,
        default=3,
        help="Minimum value for K range.",
    )
    parser.add_argument(
        "--K-max",
        dest="K_max",
        type=int,
        default=5,
        help="Maximum value for K range.",
    )
    parser.add_argument(
        "--J-min",
        dest="J_min",
        type=int,
        default=10,
        help="Minimum value for J range.",
    )
    parser.add_argument(
        "--J-max",
        dest="J_max",
        type=int,
        default=15,
        help="Maximum value for J range.",
    )
    parser.add_argument(
        "--M-min",
        dest="M_min",
        type=int,
        default=5,
        help="Minimum value for M range.",
    )
    parser.add_argument(
        "--M-max",
        dest="M_max",
        type=int,
        default=10,
        help="Maximum value for M range.",
    )
    parser.add_argument(
        "--T-min",
        dest="T_min",
        type=int,
        default=100,
        help="Minimum value for T range.",
    )
    parser.add_argument(
        "--T-max",
        dest="T_max",
        type=int,
        default=200,
        help="Maximum value for T range.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default="dataset",
        help="Output directory for the generated dataset.",
    )
    args = parser.parse_args()
    return args


def generate_instances(args, rng: np.random.Generator):
    return generate_dataset_instances(
        num_instances=NUM_INSTANCES,
        K_range=(args.K_min, args.K_max),
        J_range=(args.J_min, args.J_max),
        M_range=(args.M_min, args.M_max),
        T_range=(args.T_min, args.T_max),
        rng=rng,
    )


def write_instances(instances, output_dir: str):
    return write_dataset(instances, dataset_dir=output_dir)


def write_dataset_parameters_csv(args, *, output_dir: str) -> None:
    dataset_path = Path(output_dir)
    dataset_path.mkdir(parents=True, exist_ok=True)

    csv_path = dataset_path / "dataset_parameters.csv"
    parameters = {"num_instances": NUM_INSTANCES, **vars(args)}
    preferred_order = [
        "seed",
        "num_instances",
        "iterations",
        "K_min",
        "K_max",
        "J_min",
        "J_max",
        "M_min",
        "M_max",
        "T_min",
        "T_max",
        "output_dir",
    ]
    rows: list[tuple[str, str]] = []
    used: set[str] = set()

    for key in preferred_order:
        if key not in parameters:
            continue
        used.add(key)
        value = parameters[key]
        rows.append((key, "" if value is None else str(value)))

    for key in sorted(parameters.keys() - used):
        value = parameters[key]
        rows.append((key, "" if value is None else str(value)))

    with csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["parameter", "value"])
        writer.writerows(rows)


def main():
    args = parse_args()

    seed = args.seed if args.seed is not None else np.random.randint(1_000_000)
    args.seed = seed
    print(f"SEED: {seed}")
    rng = np.random.default_rng(seed)

    instances = generate_instances(args, rng)
    _ = write_instances(instances, args.output_dir)
    write_dataset_parameters_csv(args, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
