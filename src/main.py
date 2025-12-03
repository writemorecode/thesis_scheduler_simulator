from __future__ import annotations

import sys
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
    schedule: ScheduleResult = scheduler(problem)
    end = time.time()
    delta = end - start

    schedule.validate(problem)

    print(f"Total cost:\t{schedule.total_cost}")
    print(f"Machines:\t{schedule.machine_vector}")
    total_machine_count = schedule.machine_vector.sum()
    print(f"Total machine count:\t{total_machine_count}")
    print(f"Execution time:\t{delta:.4f} sec.")
    # avg_remaining_capacity = schedule.average_remaining_capacity().mean()
    # print(f"Avg. remaining bin capacity:\t{avg_remaining_capacity:.2f}")


def main():
    try:
        seed = int(sys.argv[1])
    except IndexError:
        seed = np.random.randint(1_000_000)
        print(f"SEED: {seed}")

    K, J, M, T = 5, 15, 5, 100
    problem = generate_random_instance(K=K, J=J, M=M, T=T, seed=seed)

    print(f"Buy costs:\t{problem.purchase_costs}\nOpen costs:\t{problem.running_costs}")
    print()

    print("Ruin-recreate scheduler:")
    run_scheduler(problem, lambda prob: ruin_recreate_schedule(prob, num_iterations=20))


if __name__ == "__main__":
    main()
