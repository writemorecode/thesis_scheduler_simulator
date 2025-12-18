from __future__ import annotations

import argparse

import numpy as np

from problem_generation import generate_dataset


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
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    seed = args.seed if args.seed is not None else np.random.randint(1_000_000)
    print(f"SEED: {seed}")
    rng = np.random.default_rng(seed)

    _ = generate_dataset(
        K_range=(args.K_min, args.K_max),
        J_range=(args.J_min, args.J_max),
        M_range=(args.M_min, args.M_max),
        T_range=(args.T_min, args.T_max),
        num_instances=100,
        rng=rng,
    )


if __name__ == "__main__":
    main()
