from __future__ import annotations

import time

from algorithms import schedule_jobs
from problem_generation import generate_random_instance


def main():
    K, J, M, T = 5, 10, 4, 100
    problem = generate_random_instance(K=K, J=J, M=M, T=T)

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


if __name__ == "__main__":
    main()
