from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from algorithms import (
    BinInfo,
    ScheduleResult,
    TimeSlotSolution,
    build_time_slot_solution,
    _machine_counts_from_bins,
    _solution_cost,
    _sort_bins_by_utilization,
    pack_time_slot,
    repack_jobs,
)
from packing import best_fit_decreasing
from problem_generation import ProblemInstance


def _copy_bins(bins: Sequence[BinInfo]) -> list[BinInfo]:
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


def _ruin_lowest_utilization_bins(
    slot_solution: TimeSlotSolution,
    requirements: np.ndarray,
    max_fraction: float,
    rng: np.random.Generator,
) -> list[BinInfo]:
    """
    Remove the lowest-utilization bins up to ``max_fraction`` of available bins.

    Returns the list of bins to keep (post-ruin); removed bins are discarded.
    """

    bins = _copy_bins(slot_solution.bins)

    # Shuffle first to randomize tie ordering among equally utilized bins.
    rng.shuffle(bins)
    _sort_bins_by_utilization(bins, requirements)

    num_bins = len(bins)
    if max_fraction <= 0.0 or num_bins == 0:
        ruin_count = 0
    else:
        max_removal = min(num_bins, int(math.ceil(max_fraction * num_bins)))
        ruin_count = int(rng.integers(0, max_removal + 1))

    if ruin_count == 0:
        return bins

    return bins[ruin_count:]


def _ruin_random_bins(
    slot_solution: TimeSlotSolution,
    max_fraction: float,
    rng: np.random.Generator,
) -> list[BinInfo]:
    """
    Remove a random subset of open bins, limited by ``max_fraction``.

    Returns the list of bins to keep (post-ruin); removed bins are discarded.
    """

    bins = _copy_bins(slot_solution.bins)

    num_bins = len(bins)
    if max_fraction <= 0.0 or num_bins == 0:
        ruin_count = 0
    else:
        max_removal = min(num_bins, int(math.ceil(max_fraction * num_bins)))
        ruin_count = int(rng.integers(0, max_removal + 1))

    if ruin_count == 0:
        return bins

    removed_indices = set(rng.choice(len(bins), size=ruin_count, replace=False))
    return [bin_info for idx, bin_info in enumerate(bins) if idx not in removed_indices]


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

    return build_time_slot_solution(
        bfd_result.bins, capacities.shape[1], requirements, running_costs
    )


def ruin_recreate_schedule(
    problem: ProblemInstance,
    num_iterations: int = 50,
    alpha: float = 0.50,
    *,
    rng: np.random.Generator | None = None,
) -> ScheduleResult:
    """
    Schedule jobs using a ruin-and-recreate metaheuristic.

    The procedure starts from the marginal-cost machine vector, then iteratively
    improves it by (a) re-packing each slot, (b) ruining bins using one of two
    random operators (lowest utilization or random removal), and (c) recreating
    slot packings with best-fit decreasing. Inferior
    candidates are accepted with probability following a cosine decay from 1 to
    0, allowing exploration early and converging later. ``alpha`` limits the
    maximum fraction of bins the ruin step may remove.
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

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")

    purchased_counts = np.zeros(M, dtype=int)
    initial_solution = [
        pack_time_slot(
            capacities=capacities,
            requirements=requirements,
            job_counts=time_slot,
            purchase_costs=purchase_costs,
            running_costs=running_costs,
            purchased_counts=purchased_counts,
        )
        for time_slot in job_matrix
    ]

    print(f"Purchased machines: {purchased_counts}")

    best_cost, best_machine_vector = _solution_cost(
        initial_solution, purchase_costs, running_costs
    )
    print(f"x:\t{best_machine_vector}\tcost: {best_cost}")
    best_slots = [slot.copy() for slot in initial_solution]
    current_slots = [slot.copy() for slot in initial_solution]

    for iteration in range(num_iterations):
        repacked_slots = [
            repack_jobs(slot, capacities, requirements, running_costs)
            for slot in current_slots
        ]
        repacked_cost, repacked_machine_vector = _solution_cost(
            repacked_slots, purchase_costs, running_costs
        )

        ruined_slots: list[TimeSlotSolution] = []
        ruin_ops = (
            lambda s: _ruin_lowest_utilization_bins(s, requirements, alpha, rng),
            lambda s: _ruin_random_bins(s, alpha, rng),
        )
        for slot_idx, slot in enumerate(repacked_slots):
            ruin_choice = rng.integers(0, len(ruin_ops))
            kept_bins = ruin_ops[ruin_choice](slot)
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

        if candidate_cost < best_cost or rng.random() <= 0.50:
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
            f"i:{iteration}\tx:\t{best_machine_vector}\tcost: {best_cost}\tcandidate cost: {candidate_cost}\tcandidate machine vector: {candidate_machine_vector}"
        )

    return ScheduleResult(
        total_cost=best_cost,
        machine_vector=best_machine_vector,
        upper_bound=None,
        time_slot_solutions=best_slots,
    )
