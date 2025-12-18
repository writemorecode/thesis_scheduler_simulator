from __future__ import annotations

import argparse
import time
from collections.abc import Callable

from pathlib import Path

import numpy as np

from algorithms import ScheduleResult
from ruin_recreate import ruin_recreate_schedule
from problem_generation import (
    ProblemInstance,
    generate_dataset,
    generate_random_instance,
)


def run_scheduler(
    problem: ProblemInstance,
    scheduler=Callable[[ProblemInstance], ScheduleResult],
) -> None:
    start = time.time()
    try:
        schedule: ScheduleResult = scheduler(problem)
    except KeyboardInterrupt:
        print("Stopping...")
        return None
    end = time.time()
    delta = end - start

    schedule.validate(problem)

    print(f"Total cost:\t{schedule.total_cost}")
    print(f"Machines:\t{schedule.machine_vector}")
    total_machine_count = schedule.machine_vector.sum()
    print(f"Total machine count:\t{total_machine_count}")
    print(f"Execution time:\t{delta:.4f} sec.")


def make_dataset(rng: np.random.Generator):
    K_range = (3, 5)
    M_range = (4, 6)
    J_range = (10, 40)
    T_range = (50, 200)
    dataset_size = 10

    dataset_path = Path("dataset")
    if dataset_path.exists():
        return

    dataset_metadata = generate_dataset(
        num_instances=dataset_size,
        K_range=K_range,
        M_range=M_range,
        J_range=J_range,
        T_range=T_range,
        rng=rng,
    )

    for md in dataset_metadata:
        print(md)


def main():
    parser = argparse.ArgumentParser(description="Run scheduler.")
    parser.add_argument("--seed", type=int, help="Seed for random instance generation.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of algorithm iterations.",
    )
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else np.random.randint(1_000_000)
    print(f"SEED: {seed}")
    rng = np.random.default_rng(seed)

    iterations = args.iterations

    K, J, M, T = 5, 20, 5, 100
    problem = generate_random_instance(K=K, J=J, M=M, T=T, rng=rng)

    print(f"Buy costs:\t{problem.purchase_costs}\nOpen costs:\t{problem.running_costs}")
    print()
    print("Bin capacities:")
    print(problem.capacities)

    run_scheduler(
        problem, lambda prob: ruin_recreate_schedule(prob, iterations, rng=rng)
    )


if __name__ == "__main__":
    main()
