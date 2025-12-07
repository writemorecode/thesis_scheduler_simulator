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
    _sort_bins_by_utilization,
    ffd_schedule,
    repack_schedule,
)
from packing import best_fit_decreasing, first_fit_decreasing
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


def _shake_lowest_utilization_bins(
    schedule: ScheduleResult,
    job_matrix: np.ndarray,
    capacities: np.ndarray,
    requirements: np.ndarray,
    purchase_costs: np.ndarray,
    running_costs: np.ndarray,
    *,
    max_fraction: float,
    rng: np.random.Generator,
) -> ScheduleResult:
    """
    Remove low-utilization bins and rebuild the schedule with BFD.
    """

    job_matrix = np.asarray(job_matrix, dtype=int)
    if job_matrix.ndim == 1:
        job_matrix = job_matrix.reshape(1, -1)
    num_types = capacities.shape[1]

    shaken_slots: list[TimeSlotSolution] = []
    for slot_idx, slot in enumerate(schedule.time_slot_solutions):
        bins = _copy_bins(slot.bins)
        rng.shuffle(bins)
        _sort_bins_by_utilization(bins, requirements, running_costs)

        ruin_count = 0
        if max_fraction > 0.0 and bins:
            max_removal = min(len(bins), int(math.ceil(max_fraction * len(bins))))
            ruin_count = int(rng.integers(0, max_removal + 1))

        kept_bins = bins[ruin_count:]
        opened_bins = _machine_counts_from_bins(kept_bins, num_types)
        job_counts = job_matrix[slot_idx].reshape(-1)

        packing_result = best_fit_decreasing(
            capacities,
            requirements,
            purchase_costs,
            running_costs,
            L=job_counts,
            opened_bins=opened_bins,
        )

        recreated_slot = build_time_slot_solution(
            packing_result.bins,
            num_types,
            requirements,
            running_costs,
        )
        shaken_slots.append(recreated_slot)

    purchase_vec = np.asarray(purchase_costs, dtype=float).reshape(-1)
    running_vec = np.asarray(running_costs, dtype=float).reshape(-1)
    machine_vector = np.zeros_like(purchase_vec, dtype=int)
    total_cost = 0.0

    for slot in shaken_slots:
        counts = np.asarray(slot.machine_counts, dtype=int).reshape(-1)
        machine_vector = np.maximum(machine_vector, counts)
        total_cost += float(np.dot(running_vec, counts))

    total_cost += float(np.dot(purchase_vec, machine_vector))

    return ScheduleResult(
        total_cost=total_cost,
        machine_vector=machine_vector,
        upper_bound=schedule.upper_bound,
        time_slot_solutions=shaken_slots,
    )


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

    max_fraction = 0.50
    ruin_count = min(num_bins, int(math.ceil(max_fraction * len(bins))))

    removed_indices = set(rng.choice(len(bins), size=ruin_count, replace=False))
    print(f"Removed bin indices: {removed_indices}")
    print()
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
    _, M = capacities.shape

    job_counts = job_matrix[slot_index].reshape(-1)
    opened_bins = np.zeros(M, dtype=int)
    for bin_info in kept_bins:
        opened_bins[bin_info.bin_type] += 1

    bins_before = _machine_counts_from_bins(kept_bins, M)
    print(f"Bins before recreate: {bins_before}")

    packing_result = first_fit_decreasing(
        capacities,
        requirements,
        purchase_costs,
        running_costs,
        L=job_counts,
        opened_bins=opened_bins,
    )

    time_slot_solution = build_time_slot_solution(
        packing_result.bins, capacities.shape[1], requirements, running_costs
    )
    bins_after_recreate = _machine_counts_from_bins(time_slot_solution.bins, M)
    print(f"Bins after recreate: {bins_after_recreate}")
    return time_slot_solution


def _shake_schedule(
    schedule: ScheduleResult,
    job_matrix: np.ndarray,
    capacities: np.ndarray,
    requirements: np.ndarray,
    purchase_costs: np.ndarray,
    running_costs: np.ndarray,
    *,
    max_fraction: float,
    rng: np.random.Generator,
) -> ScheduleResult:
    """
    Remove a used bin type, penalize it, and rebuild the schedule.
    """

    purchase_vec = np.asarray(purchase_costs, dtype=float).reshape(-1)
    running_vec = np.asarray(running_costs, dtype=float).reshape(-1)
    machine_vec = np.asarray(schedule.machine_vector, dtype=int).reshape(-1)
    used_types = np.nonzero(machine_vec > 0)[0]
    if used_types.size == 0:
        return schedule

    shaken_type = int(rng.choice(used_types))
    # print(f"Shaking bin type: {shaken_type}")

    penalty_cost = float(purchase_vec.sum() + running_vec.sum())
    updated_purchase = purchase_vec.copy()
    updated_running = running_vec.copy()
    updated_purchase[shaken_type] = penalty_cost
    updated_running[shaken_type] = penalty_cost

    shaken_slots: list[TimeSlotSolution] = []
    num_types = capacities.shape[1]
    for slot_idx, slot in enumerate(schedule.time_slot_solutions):
        kept_bins = _copy_bins(
            [bin_info for bin_info in slot.bins if bin_info.bin_type != shaken_type]
        )
        opened_bins = _machine_counts_from_bins(kept_bins, num_types)
        job_counts = job_matrix[slot_idx].reshape(-1)

        packing_result = first_fit_decreasing(
            capacities,
            requirements,
            updated_purchase,
            updated_running,
            L=job_counts,
            opened_bins=opened_bins,
        )
        recreated_slot = build_time_slot_solution(
            packing_result.bins,
            num_types,
            requirements,
            running_costs=updated_running,
        )
        shaken_slots.append(recreated_slot)

    machine_vector = np.zeros_like(updated_purchase, dtype=int)
    total_cost = 0.0

    for slot in shaken_slots:
        counts = np.asarray(slot.machine_counts, dtype=int).reshape(-1)
        machine_vector = np.maximum(machine_vector, counts)
        total_cost += float(np.dot(updated_running, counts))

    total_cost += float(np.dot(updated_purchase, machine_vector))

    return ScheduleResult(
        total_cost=total_cost,
        machine_vector=machine_vector,
        upper_bound=schedule.upper_bound,
        time_slot_solutions=shaken_slots,
    )


def ruin_recreate_schedule(
    problem: ProblemInstance,
    max_iterations: int = 20,
    rng: np.random.Generator | None = None,
) -> ScheduleResult:
    rng = rng or np.random.default_rng()

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
        purchased_bins=None,
    )
    x = x_0
    x_best = x

    print(
        f"Iteration {0}:\tbest cost {x_best.total_cost}\tmachines {x_best.machine_vector}"
    )
    for it in range(max_iterations):
        # 2. Global search phase
        x_shaken = _shake_schedule(
            # x_shaken = _shake_lowest_utilization_bins(
            schedule=x,
            job_matrix=L,
            capacities=C,
            requirements=R,
            purchase_costs=c_p,
            running_costs=c_r,
            max_fraction=0.6,
            rng=rng,
        )
        # 3. Local improvement phase
        x_repacked = repack_schedule(x_shaken, C, R, c_p, c_r)

        print(
            f"Iteration {it + 1}:\tcost\t{x_repacked.total_cost:.4f}\t(best {x_best.total_cost:.4f})\tmachines {x_repacked.machine_vector}"
        )

        if x_repacked.total_cost < x_best.total_cost:
            print("Accepted inferior solution")
            x_best = x_repacked
        x = x_repacked

    return x_best
