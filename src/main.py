from __future__ import annotations

import sys
import time

import numpy as np

from algorithms import schedule_jobs
from problem_generation import RandomInstance, generate_random_instance


def solve_with_upper_bound(problem: RandomInstance):
    start_time = time.time()
    schedule = schedule_jobs(
        C=problem.capacities,
        R=problem.requirements,
        L=problem.job_counts,
        purchase_costs=problem.purchase_costs,
        running_costs=problem.running_costs,
    )
    end_time = time.time()
    delta = end_time - start_time
    print(f"Total cost: {schedule.total_cost}")
    print(f"Machines: {schedule.machine_vector}")
    print(f"Execution time: {delta} sec.")


def solve_with_marginal_cost(problem: RandomInstance):
    start_time = time.time()
    schedule = schedule_jobs(
        C=problem.capacities,
        R=problem.requirements,
        L=problem.job_counts,
        purchase_costs=problem.purchase_costs,
        running_costs=problem.running_costs,
        initial_method="marginal_cost",
    )
    end_time = time.time()
    delta = end_time - start_time
    print(f"Total cost: {schedule.total_cost}")
    print(f"Machines: {schedule.machine_vector}")
    print(f"Execution time: {delta} sec.")


def main():
    try:
        seed = int(sys.argv[1])
    except IndexError:
        seed = np.random.randint(1_000_000)

    K, J, M, T = 5, 10, 4, 100
    problem = generate_random_instance(K=K, J=J, M=M, T=T, seed=seed)

    print(f"Buy costs: {problem.purchase_costs}\tOpen costs: {problem.running_costs}")

    print("UPPER-BOUND")
    solve_with_upper_bound(problem)

    print("-" * 50)

    print("MARGINAL COST")
    solve_with_marginal_cost(problem)


if __name__ == "__main__":
    main()
