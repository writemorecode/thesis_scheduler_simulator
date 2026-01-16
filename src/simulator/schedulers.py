from __future__ import annotations

from collections.abc import Callable

import numpy as np

from simulator.algorithms import (
    ScheduleResult,
    ffd_schedule,
    ffd_weighted_sort_schedule,
)
from simulator.best_fit import bfd_schedule
from simulator.packing import BinTypeSelectionMethod, JobTypeOrderingMethod
from simulator.peak_demand_scheduler import peak_demand_scheduler
from simulator.problem import ProblemInstance
from simulator.ruin_recreate import ruin_recreate_schedule
from simulator.simple_scheduler import simple_scheduler

SchedulerFn = Callable[[ProblemInstance], ScheduleResult]

DEFAULT_RUIN_ITERATIONS = 50
DEFAULT_REPACK_ITERATIONS = 20

SCHEDULER_ALIASES: dict[str, str] = {
    "ruin_recreate": "ruin_recreate",
    "rr": "ruin_recreate",
    "ffd": "ffd",
    "first_fit_decreasing": "ffd",
    "ffdws": "ffd_new",
    "ffd_new": "ffd_new",
    "ffd_weighted_sort": "ffd_new",
    "ffd_sum": "ffd_sum",
    "ffd_max": "ffd_max",
    "ffd_prod": "ffd_prod",
    "ffd_l2": "ffd_l2",
    "ffd_with_repack": "ffd_with_repack",
    "simple_scheduler": "ffd_with_repack",
    "bfd": "bfd",
    "peak_demand": "peak_demand",
    "peak": "peak_demand",
    "peak_demand_scheduler": "peak_demand",
    "pds": "peak_demand",
}


def normalize_scheduler_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    if normalized in SCHEDULER_ALIASES:
        return SCHEDULER_ALIASES[normalized]
    known = ", ".join(sorted(SCHEDULER_ALIASES.keys()))
    raise ValueError(f"Unknown scheduler '{name}'. Known names: {known}.")


def ffd_default_schedule(problem: ProblemInstance) -> ScheduleResult:
    return ffd_schedule(problem, BinTypeSelectionMethod.CHEAPEST)


def ffd_new_schedule(problem: ProblemInstance) -> ScheduleResult:
    return ffd_weighted_sort_schedule(problem, BinTypeSelectionMethod.SLACK)


def ffd_sum_schedule(problem: ProblemInstance) -> ScheduleResult:
    return ffd_schedule(
        problem,
        BinTypeSelectionMethod.CHEAPEST,
        job_ordering_method=JobTypeOrderingMethod.SORT_SUM,
    )


def ffd_max_schedule(problem: ProblemInstance) -> ScheduleResult:
    return ffd_schedule(
        problem,
        BinTypeSelectionMethod.CHEAPEST,
        job_ordering_method=JobTypeOrderingMethod.SORT_MAX,
    )


def ffd_prod_schedule(problem: ProblemInstance) -> ScheduleResult:
    return ffd_schedule(
        problem,
        BinTypeSelectionMethod.CHEAPEST,
        job_ordering_method=JobTypeOrderingMethod.SORT_PROD,
    )


def ffd_l2_schedule(problem: ProblemInstance) -> ScheduleResult:
    return ffd_schedule(
        problem,
        BinTypeSelectionMethod.CHEAPEST,
        job_ordering_method=JobTypeOrderingMethod.SORT_L2,
    )


def ffd_with_repack_schedule(
    problem: ProblemInstance, *, iterations: int = DEFAULT_REPACK_ITERATIONS
) -> ScheduleResult:
    return simple_scheduler(problem, max_iterations=iterations)


def ruin_recreate_default_schedule(problem: ProblemInstance) -> ScheduleResult:
    rng = np.random.default_rng()
    return ruin_recreate_schedule(
        problem, max_iterations=DEFAULT_RUIN_ITERATIONS, rng=rng
    )


SCHEDULER_REGISTRY: dict[str, SchedulerFn] = {
    "ruin_recreate": ruin_recreate_default_schedule,
    "ffd": ffd_default_schedule,
    "ffd_new": ffd_new_schedule,
    "ffd_sum": ffd_sum_schedule,
    "ffd_max": ffd_max_schedule,
    "ffd_prod": ffd_prod_schedule,
    "ffd_l2": ffd_l2_schedule,
    "ffd_with_repack": lambda problem: ffd_with_repack_schedule(problem),
    "bfd": bfd_schedule,
    "peak_demand": peak_demand_scheduler,
}


def get_scheduler(
    name: str,
    *,
    iterations: int | None = None,
    rng: np.random.Generator | None = None,
) -> SchedulerFn:
    canonical = normalize_scheduler_name(name)

    if canonical == "ruin_recreate":
        max_iterations = DEFAULT_RUIN_ITERATIONS if iterations is None else iterations
        rng = rng or np.random.default_rng()
        return lambda problem: ruin_recreate_schedule(
            problem, max_iterations=max_iterations, rng=rng
        )

    if canonical == "ffd_with_repack":
        max_iterations = DEFAULT_REPACK_ITERATIONS if iterations is None else iterations
        return lambda problem: ffd_with_repack_schedule(
            problem, iterations=max_iterations
        )

    if canonical in SCHEDULER_REGISTRY:
        return SCHEDULER_REGISTRY[canonical]

    raise ValueError(f"Unsupported scheduler '{name}'.")


def run_instance(
    problem: ProblemInstance,
    scheduler: str = "ffd",
    *,
    iterations: int | None = None,
    rng: np.random.Generator | None = None,
) -> ScheduleResult:
    scheduler_fn = get_scheduler(scheduler, iterations=iterations, rng=rng)
    return scheduler_fn(problem)
