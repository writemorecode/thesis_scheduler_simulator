from __future__ import annotations

import argparse
import time
from typing import Callable

import numpy as np

from algorithms import ScheduleResult
from ruin_recreate import ruin_recreate_schedule
from problem_generation import ProblemInstance, generate_random_instance


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
    iterations = args.iterations
    print(f"SEED: {seed}")

    K, J, M, T = 5, 20, 5, 100
    problem = generate_random_instance(K=K, J=J, M=M, T=T, seed=seed)

    print(f"Buy costs:\t{problem.purchase_costs}\nOpen costs:\t{problem.running_costs}")
    print()

    run_scheduler(problem, lambda prob: ruin_recreate_schedule(prob, iterations))


if __name__ == "__main__":
    main()
