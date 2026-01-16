from __future__ import annotations

import numpy as np

from simulator.algorithms import (
    ScheduleResult,
    TimeSlotSolution,
    build_time_slot_solution,
)
from simulator.packing import (
    BinTypeSelectionMethod,
    JobTypeOrderingMethod,
    first_fit_decreasing,
)
from simulator.problem import ProblemInstance


def peak_demand_scheduler(problem: ProblemInstance) -> ScheduleResult:
    """
    Schedule jobs by packing the most expensive slot first, then carrying machines forward.

    Slot costs are computed as ``resource_weights.T @ requirements @ job_counts.T``
    to identify the heaviest time slot. That slot is packed first with FFD. Subsequent
    slots are packed in chronological order while reusing the machine vector from prior
    slots as already-open bins. Machines are never closed; new machines may be opened
    as needed.
    """

    capacities = np.asarray(problem.capacities, dtype=float)
    requirements = np.asarray(problem.requirements, dtype=float)
    job_counts = np.asarray(problem.job_counts, dtype=int)
    purchase_costs = np.asarray(problem.purchase_costs, dtype=float).reshape(-1)
    running_costs = np.asarray(problem.running_costs, dtype=float).reshape(-1)
    resource_weights = np.asarray(problem.resource_weights, dtype=float).reshape(-1)

    if capacities.ndim != 2 or requirements.ndim != 2:
        raise ValueError("capacities and requirements must be 2D matrices.")
    if capacities.shape[0] != requirements.shape[0]:
        raise ValueError(
            "capacities and requirements must describe the same resources."
        )
    if resource_weights.shape[0] != capacities.shape[0]:
        raise ValueError(
            f"resource_weights must have length {capacities.shape[0]}, got {resource_weights.shape[0]}."
        )

    if job_counts.ndim == 1:
        job_counts = job_counts.reshape(1, -1)
    elif job_counts.ndim != 2:
        raise ValueError("job_counts must be a vector or a 2D matrix.")
    if np.any(job_counts < 0):
        raise ValueError("job_counts entries must be non-negative.")

    T, J = job_counts.shape
    K, M = capacities.shape

    if requirements.shape[1] != J:
        raise ValueError(
            f"requirements must have {J} job-type columns; got {requirements.shape[1]}."
        )
    if purchase_costs.shape[0] != M or running_costs.shape[0] != M:
        raise ValueError("Cost vectors must have one entry per machine type.")

    if T == 0:
        empty_counts = np.zeros(M, dtype=int)
        return ScheduleResult(
            total_cost=0.0,
            machine_vector=empty_counts,
            time_slot_solutions=[],
            purchased_baseline=empty_counts.copy(),
        )

    slot_costs = (resource_weights.reshape(1, K) @ requirements @ job_counts.T).reshape(
        -1
    )
    t_max = int(np.argmax(slot_costs))

    purchased_bins = np.zeros(M, dtype=int)
    opened_bins = np.zeros(M, dtype=int)
    machine_vector = np.zeros(M, dtype=int)
    time_slot_solutions: list[TimeSlotSolution] = [None] * T  # type: ignore
    running_total = 0.0

    def _pack_slot(slot_jobs: np.ndarray) -> TimeSlotSolution:
        """Pack a single slot using FFD with carried machines."""

        if np.all(slot_jobs == 0):
            return TimeSlotSolution(machine_counts=np.zeros(M, dtype=int), bins=[])

        result = first_fit_decreasing(
            C=capacities,
            R=requirements,
            purchase_costs=purchase_costs,
            opening_costs=running_costs,
            L=slot_jobs,
            opened_bins=opened_bins,
            purchased_bins=purchased_bins,
            job_ordering_method=JobTypeOrderingMethod.SORT_BY_WEIGHT,
            selection_method=BinTypeSelectionMethod.SLACK,
        )
        return build_time_slot_solution(
            result.bins,
            M,
            requirements,
            running_costs,
            resource_weights=resource_weights,
        )

    # Pack the heaviest slot first.
    heavy_solution = _pack_slot(job_counts[t_max])
    time_slot_solutions[t_max] = heavy_solution
    purchased_bins = np.maximum(purchased_bins, heavy_solution.machine_counts)
    opened_bins = np.maximum(opened_bins, heavy_solution.machine_counts)
    machine_vector = np.maximum(machine_vector, heavy_solution.machine_counts)
    running_total += float(np.dot(running_costs, heavy_solution.machine_counts))

    # Pack remaining slots in chronological order.
    for t in range(T):
        if t == t_max:
            continue

        slot_solution = _pack_slot(job_counts[t])
        time_slot_solutions[t] = slot_solution
        purchased_bins = np.maximum(purchased_bins, slot_solution.machine_counts)
        opened_bins = np.maximum(opened_bins, slot_solution.machine_counts)
        machine_vector = np.maximum(machine_vector, slot_solution.machine_counts)
        running_total += float(np.dot(running_costs, slot_solution.machine_counts))

    purchase_total = float(np.dot(purchase_costs, purchased_bins))
    total_cost = purchase_total + running_total

    solution = ScheduleResult(
        total_cost=total_cost,
        machine_vector=machine_vector,
        time_slot_solutions=time_slot_solutions,
        purchased_baseline=np.zeros(M, dtype=int),
    )

    return solution
