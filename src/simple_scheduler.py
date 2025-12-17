from __future__ import annotations


import numpy as np

from algorithms import (
    ScheduleResult,
    ffd_schedule,
    repack_schedule,
)

from problem_generation import ProblemInstance


def simple_scheduler(
    problem: ProblemInstance,
    max_iterations: int = 20,
) -> ScheduleResult:
    C = np.asarray(problem.capacities, dtype=float)
    R = np.asarray(problem.requirements, dtype=float)
    L = np.asarray(problem.job_counts, dtype=int)
    c_p = np.asarray(problem.purchase_costs, dtype=float).reshape(-1)
    c_r = np.asarray(problem.running_costs, dtype=float).reshape(-1)

    if C.ndim != 2 or R.ndim != 2:
        raise ValueError("C and R must be 2D matrices.")
    if C.shape[0] != R.shape[0]:
        raise ValueError("C and R must describe the same resource dimensions.")

    if L.ndim == 1:
        L = L.reshape(1, -1)
    elif L.ndim != 2:
        raise ValueError("L must be a vector or a 2D matrix.")
    if np.any(L < 0):
        raise ValueError("Job counts in L must be non-negative.")

    _, M = C.shape
    _, J = R.shape
    if L.shape[1] != J:
        raise ValueError(f"L must contain {J} job-type columns; got {L.shape[1]}.")

    if c_p.shape[0] != M or c_r.shape[0] != M:
        raise ValueError("Cost vectors must have one entry per machine type.")

    # 1. Compute initial solution
    x_0 = ffd_schedule(
        C=C,
        R=R,
        L=L,
        purchase_costs=c_p,
        running_costs=c_r,
    )
    x = x_0
    x_best = x_0
    iterations_since_improvement = 0

    print(
        f"Iteration {0}:\tCost:\t{x_best.total_cost:.4f}\tMachines:\t{x_best.machine_vector}"
    )
    it = 0

    while iterations_since_improvement < max_iterations:
        it += 1
        iterations_since_improvement += 1

        x_repacked = repack_schedule(x, C, R, c_p, c_r)

        if x_repacked.total_cost < x_best.total_cost:
            x_best = x_repacked
            iterations_since_improvement = 0

        x = x_repacked

        print(
            f"Iteration {it + 1}:\tCost: {x.total_cost:.4f}\tBest cost: {x_best.total_cost:.4f}\tMachines: {x.machine_vector}\tBest machines: {x_best.machine_vector}"
        )

    return x_best
