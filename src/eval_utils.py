from __future__ import annotations

from collections.abc import Callable

import numpy as np

from algorithms import ScheduleResult, ffd_schedule, ffd_weighted_sort_schedule
from best_fit import bfd_schedule
from packing import BinTypeSelectionMethod, JobTypeOrderingMethod
from problem_generation import ProblemInstance
from ruin_recreate import ruin_recreate_schedule
from simple_scheduler import simple_scheduler

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
}


def normalize_scheduler_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    if normalized in SCHEDULER_ALIASES:
        return SCHEDULER_ALIASES[normalized]
    known = ", ".join(sorted(SCHEDULER_ALIASES.keys()))
    raise ValueError(f"Unknown scheduler '{name}'. Known names: {known}.")


def parse_scheduler_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def scheduler_output_filename(name: str) -> str:
    canonical = normalize_scheduler_name(name)
    return f"eval_{canonical}.csv"


def build_scheduler(
    name: str, *, iterations: int, rng: np.random.Generator
) -> Callable[[ProblemInstance], ScheduleResult]:
    canonical = normalize_scheduler_name(name)

    if canonical == "ruin_recreate":
        return lambda problem: ruin_recreate_schedule(
            problem, max_iterations=iterations, rng=rng
        )

    if canonical == "ffd":

        def _ffd(problem: ProblemInstance) -> ScheduleResult:
            return ffd_schedule(problem, BinTypeSelectionMethod.CHEAPEST)

        return _ffd

    if canonical == "ffd_new":

        def _ffd(problem: ProblemInstance) -> ScheduleResult:
            return ffd_weighted_sort_schedule(problem, BinTypeSelectionMethod.SLACK)

        return _ffd

    if canonical == "ffd_sum":

        def _ffd(problem: ProblemInstance) -> ScheduleResult:
            return ffd_schedule(
                problem,
                BinTypeSelectionMethod.CHEAPEST,
                job_ordering_method=JobTypeOrderingMethod.SORT_SUM,
            )

        return _ffd

    if canonical == "ffd_max":

        def _ffd(problem: ProblemInstance) -> ScheduleResult:
            return ffd_schedule(
                problem,
                BinTypeSelectionMethod.CHEAPEST,
                job_ordering_method=JobTypeOrderingMethod.SORT_MAX,
            )

        return _ffd

    if canonical == "ffd_prod":

        def _ffd(problem: ProblemInstance) -> ScheduleResult:
            return ffd_schedule(
                problem,
                BinTypeSelectionMethod.CHEAPEST,
                job_ordering_method=JobTypeOrderingMethod.SORT_PROD,
            )

        return _ffd

    if canonical == "ffd_l2":

        def _ffd(problem: ProblemInstance) -> ScheduleResult:
            return ffd_schedule(
                problem,
                BinTypeSelectionMethod.CHEAPEST,
                job_ordering_method=JobTypeOrderingMethod.SORT_L2,
            )

        return _ffd

    if canonical == "ffd_with_repack":
        return lambda problem: simple_scheduler(problem, max_iterations=iterations)

    if canonical == "bfd":
        return bfd_schedule

    raise ValueError(f"Unsupported scheduler '{name}'.")
