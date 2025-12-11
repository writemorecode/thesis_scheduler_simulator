from __future__ import annotations

import math
from collections.abc import Sequence

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
from packing import first_fit_decreasing_largest
from problem_generation import ProblemInstance

MAX_FRACTION = 0.95


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


def _shake_remove_lowest_utilization_bins(
    schedule: ScheduleResult,
    job_matrix: np.ndarray,
    capacities: np.ndarray,
    requirements: np.ndarray,
    purchase_costs: np.ndarray,
    running_costs: np.ndarray,
    *,
    rng: np.random.Generator,
) -> ScheduleResult:
    """
    Remove low-utilization bins and rebuild the schedule with FFDL.
    """

    job_matrix = np.asarray(job_matrix, dtype=int)
    if job_matrix.ndim == 1:
        job_matrix = job_matrix.reshape(1, -1)
    num_types = capacities.shape[1]

    shaken_slots: list[TimeSlotSolution] = []
    for slot_idx, slot in enumerate(schedule.time_slot_solutions):
        bins = _copy_bins(slot.bins)
        rng.shuffle(np.array(bins))
        _sort_bins_by_utilization(bins, requirements, running_costs)

        ruin_count = 0
        if bins:
            max_removal = min(len(bins), int(math.ceil(MAX_FRACTION * len(bins))))
            ruin_count = int(rng.integers(0, max_removal + 1))

        kept_bins = bins[ruin_count:]
        opened_bins = _machine_counts_from_bins(kept_bins, num_types)
        job_counts = job_matrix[slot_idx].reshape(-1)

        packing_result = first_fit_decreasing_largest(
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


def _shake_uniform_bin_counts(
    schedule: ScheduleResult,
    job_matrix: np.ndarray,
    capacities: np.ndarray,
    requirements: np.ndarray,
    purchase_costs: np.ndarray,
    running_costs: np.ndarray,
    rng: np.random.Generator | None = None,
) -> ScheduleResult:
    """
    Flatten bin counts to a uniform level before recreating with FFD.
    """

    job_matrix = np.asarray(job_matrix, dtype=int)
    if job_matrix.ndim == 1:
        job_matrix = job_matrix.reshape(1, -1)
    num_types = capacities.shape[1]
    if num_types == 0:
        return schedule

    machine_vec = np.asarray(schedule.machine_vector, dtype=int).reshape(-1)
    total_bins = int(np.sum(machine_vec))
    uniform_count = int(math.ceil(total_bins / num_types))
    opened_bins = np.full(num_types, uniform_count, dtype=int)

    shaken_slots: list[TimeSlotSolution] = []
    for slot_idx, _slot in enumerate(schedule.time_slot_solutions):
        job_counts = job_matrix[slot_idx].reshape(-1)
        packing_result = first_fit_decreasing_largest(
            capacities,
            requirements,
            purchase_costs,
            running_costs,
            L=job_counts,
            opened_bins=opened_bins.copy(),
        )
        recreated_slot = build_time_slot_solution(
            packing_result.bins, num_types, requirements, running_costs
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

    sampled_fraction = float(rng.uniform(0.0, max_fraction))
    ruin_count = min(num_bins, int(math.ceil(sampled_fraction * len(bins))))

    removed_indices = set(rng.choice(len(bins), size=ruin_count, replace=False))
    return [bin_info for idx, bin_info in enumerate(bins) if idx not in removed_indices]


def _shake_remove_random_bins(
    schedule: ScheduleResult,
    job_matrix: np.ndarray,
    capacities: np.ndarray,
    requirements: np.ndarray,
    purchase_costs: np.ndarray,
    running_costs: np.ndarray,
    *,
    rng: np.random.Generator,
    # config: ShakeConfig | None = None,
) -> ScheduleResult:
    """
    Perturb the schedule by dropping random bins, penalizing types, and rebuilding.
    """

    purchase_vec = np.asarray(purchase_costs, dtype=float).reshape(-1)
    running_vec = np.asarray(running_costs, dtype=float).reshape(-1)
    machine_vec = np.asarray(schedule.machine_vector, dtype=int).reshape(-1)
    used_types = np.nonzero(machine_vec > 0)[0]
    if used_types.size == 0:
        return schedule

    shaken_slots: list[TimeSlotSolution] = []
    num_types = capacities.shape[1]
    for slot_idx, slot in enumerate(schedule.time_slot_solutions):
        kept_bins = _ruin_random_bins(
            slot_solution=slot,
            max_fraction=MAX_FRACTION,
            rng=rng,
        )
        opened_bins = _machine_counts_from_bins(kept_bins, num_types)
        job_counts = job_matrix[slot_idx].reshape(-1)

        packing_result = first_fit_decreasing_largest(
            capacities,
            requirements,
            purchase_vec,
            running_vec,
            L=job_counts,
            opened_bins=opened_bins,
        )
        recreated_slot = build_time_slot_solution(
            packing_result.bins,
            num_types,
            requirements,
            running_costs=running_costs,
        )
        shaken_slots.append(recreated_slot)

    machine_vector = np.zeros_like(purchase_vec, dtype=int)
    total_cost = 0.0

    for slot in shaken_slots:
        counts = np.asarray(slot.machine_counts, dtype=int).reshape(-1)
        machine_vector = np.maximum(machine_vector, counts)
        total_cost += float(np.dot(running_costs, counts))

    total_cost += float(np.dot(purchase_costs, machine_vector))

    return ScheduleResult(
        total_cost=total_cost,
        machine_vector=machine_vector,
        upper_bound=schedule.upper_bound,
        time_slot_solutions=shaken_slots,
    )


def _dominant_bin_type(schedule: ScheduleResult) -> int | None:
    """Identify the most common bin type in the current schedule."""
    dominant_bin_type = np.argmax(schedule.machine_vector)
    if schedule.machine_vector[dominant_bin_type] == 0:
        return None
    return int(dominant_bin_type)


def _shake_penalize_dominant_type(
    schedule: ScheduleResult,
    job_matrix: np.ndarray,
    capacities: np.ndarray,
    requirements: np.ndarray,
    purchase_costs: np.ndarray,
    running_costs: np.ndarray,
    *,
    rng: np.random.Generator,
) -> ScheduleResult:
    """
    Remove the dominant bin type, penalize it, and rebuild with FFD Largest.
    """
    job_matrix = np.asarray(job_matrix, dtype=int)
    if job_matrix.ndim == 1:
        job_matrix = job_matrix.reshape(1, -1)
    num_types = capacities.shape[1]

    dominant_type = _dominant_bin_type(schedule)
    if dominant_type is None:
        return schedule

    purchase_vec = np.asarray(purchase_costs, dtype=float).reshape(-1)
    running_vec = np.asarray(running_costs, dtype=float).reshape(-1)
    updated_purchase = purchase_vec.copy()
    updated_running = running_vec.copy()
    updated_purchase[dominant_type] = np.sum(updated_purchase)
    updated_running[dominant_type] = np.sum(updated_running)

    shaken_slots: list[TimeSlotSolution] = []
    for slot_idx, slot in enumerate(schedule.time_slot_solutions):
        kept_bins = [
            bin_info
            for bin_info in _copy_bins(slot.bins)
            if bin_info.bin_type != dominant_type
        ]
        opened_bins = _machine_counts_from_bins(kept_bins, num_types)
        job_counts = job_matrix[slot_idx].reshape(-1)

        packing_result = first_fit_decreasing_largest(
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
    x_best = x_0
    iterations_since_improvement = 0

    print(
        f"Iteration {0}:\tCost:\t{x_best.total_cost:.4f}\tMachines:\t{x_best.machine_vector}"
    )
    it = 0

    shake_operators = [
        _shake_remove_lowest_utilization_bins,
        _shake_remove_random_bins,
        _shake_uniform_bin_counts,
        _shake_penalize_dominant_type,
    ]
    p = np.ones(len(shake_operators)) / len(shake_operators)
    shake_success_counts = np.zeros(len(shake_operators))

    while iterations_since_improvement < max_iterations:
        it += 1
        iterations_since_improvement += 1

        # 2. Global search phase
        operator_index = rng.choice(len(shake_operators), p=p)
        shake_operator = shake_operators[operator_index]
        x_shaken = shake_operator(
            schedule=x,
            job_matrix=L,
            capacities=C,
            requirements=R,
            purchase_costs=c_p,
            running_costs=c_r,
            rng=rng,
        )

        if x_shaken.total_cost < x_best.total_cost:
            iterations_since_improvement = 0

        # 3. Local improvement phase
        x_repacked = repack_schedule(x_shaken, C, R, c_p, c_r)

        if x_repacked.total_cost < x_best.total_cost:
            x_best = x_repacked
            iterations_since_improvement = 0
            shake_success_counts[operator_index] += 1

        x = x_repacked

        print(
            f"Iteration {it + 1}:\tCost: {x.total_cost:.4f}\tBest cost: {x_best.total_cost:.4f}\tMachines: {x.machine_vector}\tBest machines: {x_best.machine_vector}"
        )

    print(f"shake success counts: {shake_success_counts}")
    return x_best
