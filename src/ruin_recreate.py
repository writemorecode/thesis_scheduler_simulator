from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np

from algorithms import (
    BinInfo,
    ScheduleResult,
    TimeSlotSolution,
    _build_time_slot_solution,
    _machine_counts_from_bins,
    _pack_all_time_slots,
    _solution_cost,
    _sort_bins_by_utilization,
    marginal_cost_initial_solution,
    repack_jobs,
)
from packing import best_fit_decreasing
from problem_generation import ProblemInstance


def _fraction(iteration: int, total_iterations: int) -> float:
    """Cosine-decay schedule that starts at 1 and ends at 0."""

    if total_iterations <= 1:
        return 1.0
    ratio = iteration / float(total_iterations - 1)
    # Cosine decay: smooth drop from 1 to 0 as iterations progress.
    return float(max(0.0, min(1.0, 0.5 * (1.0 + math.cos(math.pi * ratio)))))


def _copy_bins(bins: Sequence[BinInfo]) -> List[BinInfo]:
    """Deep copy bins so we can mutate during ruin/recreate."""

    return [
        BinInfo(
            bin_type=b.bin_type,
            capacity=b.capacity.copy(),
            remaining_capacity=b.remaining_capacity.copy(),
            item_counts=b.item_counts.copy(),
        )
        for b in bins
    ]


def _ruin_slot_bins(
    slot_solution: TimeSlotSolution,
    requirements: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> List[BinInfo]:
    """
    Remove the lowest-utilization bins according to ``fraction``.

    Returns the list of bins to keep (post-ruin); removed bins are discarded.
    """

    if fraction <= 0.0 or not slot_solution.bins:
        return _copy_bins(slot_solution.bins)

    bins = _copy_bins(slot_solution.bins)

    # Shuffle first to randomize tie ordering among equally utilized bins.
    rng.shuffle(np.array(bins))
    _sort_bins_by_utilization(bins, requirements)

    ruin_count = int(math.ceil(fraction * len(bins)))
    ruin_count = min(ruin_count, len(bins))

    return bins[ruin_count:]


def _recreate_slot(
    slot_index: int,
    kept_bins: Sequence[BinInfo],
    job_matrix: np.ndarray,
    capacities: np.ndarray,
    requirements: np.ndarray,
    purchase_costs: np.ndarray,
    running_costs: np.ndarray,
) -> TimeSlotSolution:
    """
    Re-pack a single time slot using BFD, seeding with kept bin counts.
    """

    job_counts = job_matrix[slot_index].reshape(-1)
    opened_bins = _machine_counts_from_bins(kept_bins, capacities.shape[1])

    bfd_result = best_fit_decreasing(
        capacities,
        requirements,
        purchase_costs,
        running_costs,
        L=job_counts,
        opened_bins=opened_bins,
    )

    return _build_time_slot_solution(
        bfd_result.bins, capacities.shape[1], requirements, running_costs
    )


def ruin_recreate_schedule(
    problem: ProblemInstance,
    num_iterations: int = 50,
    *,
    rng: np.random.Generator | None = None,
) -> ScheduleResult:
    """
    Schedule jobs using a ruin-and-recreate metaheuristic.

    The procedure starts from the marginal-cost machine vector, then iteratively
    improves it by (a) re-packing each slot, (b) ruining a fraction of the worst
    bins, and (c) recreating slot packings with best-fit decreasing. Inferior
    candidates are accepted with probability following a cosine decay from 1 to
    0, allowing exploration early and converging later.
    """

    rng = rng or np.random.default_rng()

    capacities = np.asarray(problem.capacities, dtype=float)
    requirements = np.asarray(problem.requirements, dtype=float)
    job_matrix = np.asarray(problem.job_counts, dtype=int)
    purchase_costs = np.asarray(problem.purchase_costs, dtype=float).reshape(-1)
    running_costs = np.asarray(problem.running_costs, dtype=float).reshape(-1)

    if capacities.ndim != 2 or requirements.ndim != 2:
        raise ValueError("C and R must be 2D matrices.")
    if capacities.shape[0] != requirements.shape[0]:
        raise ValueError("C and R must describe the same resource dimensions.")

    if job_matrix.ndim == 1:
        job_matrix = job_matrix.reshape(1, -1)
    elif job_matrix.ndim != 2:
        raise ValueError("L must be a vector or a 2D matrix.")
    if np.any(job_matrix < 0):
        raise ValueError("Job counts in L must be non-negative.")

    _, M = capacities.shape
    _, J = requirements.shape
    if job_matrix.shape[1] != J:
        raise ValueError(
            f"L must contain {J} job-type columns; got {job_matrix.shape[1]}."
        )

    if purchase_costs.shape[0] != M or running_costs.shape[0] != M:
        raise ValueError("Cost vectors must have one entry per machine type.")

    machine_vector = marginal_cost_initial_solution(
        capacities, requirements, job_matrix, purchase_costs, running_costs
    )

    time_slot_solutions = _pack_all_time_slots(
        machine_vector, capacities, requirements, job_matrix, running_costs
    )
    best_cost, best_machine_vector = _solution_cost(
        time_slot_solutions, purchase_costs, running_costs
    )
    print(f"x:\t{machine_vector}\tcost: {best_cost}")
    best_slots = [slot.copy() for slot in time_slot_solutions]
    current_slots = [slot.copy() for slot in time_slot_solutions]

    for iteration in range(num_iterations):
        p = _fraction(iteration, num_iterations)

        repacked_slots = [
            repack_jobs(slot, capacities, requirements, running_costs)
            for slot in current_slots
        ]
        repacked_cost, repacked_machine_vector = _solution_cost(
            repacked_slots, purchase_costs, running_costs
        )

        ruined_slots: List[TimeSlotSolution] = []
        for slot_idx, slot in enumerate(repacked_slots):
            kept_bins = _ruin_slot_bins(slot, requirements, p, rng)
            recreated_slot = _recreate_slot(
                slot_idx,
                kept_bins,
                job_matrix,
                capacities,
                requirements,
                purchase_costs,
                running_costs,
            )
            ruined_slots.append(recreated_slot)

        candidate_cost, candidate_machine_vector = _solution_cost(
            ruined_slots, purchase_costs, running_costs
        )

        accept_prob = p
        if candidate_cost < best_cost or rng.random() <= accept_prob:
            current_slots = ruined_slots
            current_cost = candidate_cost
            current_machine_vector = candidate_machine_vector
        else:
            current_slots = repacked_slots
            current_cost = repacked_cost
            current_machine_vector = repacked_machine_vector

        if current_cost < best_cost:
            best_cost = current_cost
            best_machine_vector = current_machine_vector
            best_slots = [slot.copy() for slot in current_slots]

        print(
            f"i:{iteration}\tx:\t{best_machine_vector}\tcost: {best_cost}\tp: {p:.4f}"
        )

    return ScheduleResult(
        total_cost=best_cost,
        machine_vector=best_machine_vector,
        upper_bound=None,
        time_slot_solutions=best_slots,
    )
