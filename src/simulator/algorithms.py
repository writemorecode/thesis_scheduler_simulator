from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from simulator.packing import BinInfo as _BaseBinInfo
from simulator.packing import (
    BinSelectionFn,
    BinTypeSelectionMethod,
    JobTypeOrderingMethod,
    first_fit_sorted,
)
from simulator.problem import ProblemInstance


class BinInfo(_BaseBinInfo):
    """Extended bin info with convenience metrics."""

    def utilization(self, resource_weights: np.ndarray | None = None) -> float:
        """
        Weighted remaining capacity based on resource_weights.

        Parameters
        ----------
        resource_weights : np.ndarray, optional
            Resource weight vector with shape ``(K,)``.
        """

        if resource_weights is not None:
            self.update_utilization_cache(resource_weights)

        return float(self._cached_utilization)


@dataclass
class TimeSlotSolution:
    """Packing state for a single time slot."""

    machine_counts: np.ndarray  # vector z_t
    bins: list[BinInfo]

    def copy(self) -> TimeSlotSolution:
        """Deep copy helper used before mutating a slot solution."""

        return TimeSlotSolution(
            machine_counts=self.machine_counts.copy(),
            bins=[
                BinInfo(
                    bin_type=b.bin_type,
                    capacity=b.capacity.copy(),
                    remaining_capacity=b.remaining_capacity.copy(),
                    item_counts=b.item_counts.copy(),
                    resource_weights=(
                        None
                        if b.resource_weights is None
                        else b.resource_weights.copy()
                    ),
                )
                for b in self.bins
            ],
        )


@dataclass
class ScheduleResult:
    """Result container for the scheduler."""

    total_cost: float
    machine_vector: np.ndarray
    time_slot_solutions: list[TimeSlotSolution]
    purchased_baseline: np.ndarray | None = None

    def validate(
        self,
        problem: ProblemInstance,
        *,
        atol: float = 1e-9,
    ) -> None:
        """
        Verify that this schedule respects capacity and job-count constraints.

        Parameters mirror the fields consumed by :func:`schedule_jobs`. A
        ``ValueError`` is raised as soon as an inconsistency is detected.
        """

        capacities = problem.capacities
        requirements = problem.requirements
        job_matrix = problem.job_counts
        purchase_costs = problem.purchase_costs
        running_costs = problem.running_costs

        capacities = np.asarray(capacities, dtype=float)
        requirements = np.asarray(requirements, dtype=float)
        job_matrix = np.asarray(job_matrix, dtype=int)

        if capacities.ndim != 2 or requirements.ndim != 2:
            raise ValueError("capacities and requirements must be 2D matrices.")
        if capacities.shape[0] != requirements.shape[0]:
            raise ValueError(
                "capacities and requirements must describe the same resource dimensions."
            )
        K, M = capacities.shape
        _, J = requirements.shape

        if job_matrix.ndim == 1:
            job_matrix = job_matrix.reshape(1, -1)
        elif job_matrix.ndim != 2:
            raise ValueError("job_matrix must be a vector or a 2D matrix.")
        if job_matrix.shape[1] != J:
            raise ValueError(
                f"job_matrix must have {J} job-type columns; got {job_matrix.shape[1]}."
            )
        if len(self.time_slot_solutions) != job_matrix.shape[0]:
            raise ValueError(
                "Number of time slots in the schedule does not match job_matrix rows."
            )

        purchase_vec = None
        running_vec = None
        if purchase_costs is not None:
            purchase_vec = np.asarray(purchase_costs, dtype=float).reshape(-1)
            if purchase_vec.shape[0] != M:
                raise ValueError(
                    f"purchase_costs must have one entry per machine type ({M})."
                )
        if running_costs is not None:
            running_vec = np.asarray(running_costs, dtype=float).reshape(-1)
            if running_vec.shape[0] != M:
                raise ValueError(
                    f"running_costs must have one entry per machine type ({M})."
                )
        baseline_vec = None
        if self.purchased_baseline is not None:
            baseline_vec = np.asarray(self.purchased_baseline, dtype=int).reshape(-1)
            if baseline_vec.shape[0] != M:
                raise ValueError(
                    f"purchased_baseline must have one entry per machine type ({M})."
                )
            if np.any(baseline_vec < 0):
                raise ValueError("purchased_baseline must contain non-negative counts.")

        expected_machine_vector = np.zeros(M, dtype=int)

        for slot_idx, (slot, required_jobs) in enumerate(
            zip(self.time_slot_solutions, job_matrix, strict=False)
        ):
            machine_counts = np.asarray(slot.machine_counts, dtype=int).reshape(-1)
            if machine_counts.shape[0] != M:
                raise ValueError(
                    f"Slot {slot_idx} machine_counts length mismatch; expected {M}."
                )
            if np.any(machine_counts < 0):
                raise ValueError(f"Slot {slot_idx} has negative machine counts.")

            bin_counts = np.zeros(M, dtype=int)
            assigned_jobs = np.zeros(J, dtype=int)

            for bin_idx, bin_info in enumerate(slot.bins):
                bin_type = int(bin_info.bin_type)
                if not (0 <= bin_type < M):
                    raise ValueError(
                        f"Bin {bin_idx} in slot {slot_idx} has invalid type {bin_type}."
                    )
                bin_counts[bin_type] += 1

                capacity = np.asarray(bin_info.capacity, dtype=float).reshape(-1)
                remaining = np.asarray(
                    bin_info.remaining_capacity, dtype=float
                ).reshape(-1)
                counts_vec = np.asarray(bin_info.item_counts, dtype=int).reshape(-1)

                if capacity.shape[0] != K or remaining.shape[0] != K:
                    raise ValueError(
                        f"Bin {bin_idx} in slot {slot_idx} has incorrect capacity shape."
                    )
                if counts_vec.shape[0] != J:
                    raise ValueError(
                        f"Bin {bin_idx} in slot {slot_idx} has incorrect item_counts length."
                    )
                if np.any(counts_vec < 0):
                    raise ValueError(
                        f"Bin {bin_idx} in slot {slot_idx} contains negative item counts."
                    )

                expected_capacity = capacities[:, bin_type].reshape(-1)
                if not np.allclose(capacity, expected_capacity, atol=atol):
                    raise ValueError(
                        f"Bin {bin_idx} in slot {slot_idx} capacity does not match type {bin_type}."
                    )

                if np.any(remaining < -atol):
                    raise ValueError(
                        f"Bin {bin_idx} in slot {slot_idx} is overpacked (negative remaining capacity)."
                    )

                load = requirements @ counts_vec
                if load.shape[0] != K:
                    raise ValueError(
                        f"Bin {bin_idx} in slot {slot_idx} produced invalid load shape."
                    )
                if np.any(load > capacity + atol):
                    raise ValueError(
                        f"Bin {bin_idx} in slot {slot_idx} exceeds capacity constraints."
                    )
                if not np.allclose(capacity - load, remaining, atol=atol):
                    raise ValueError(
                        f"Bin {bin_idx} in slot {slot_idx} has inconsistent remaining capacity."
                    )

                assigned_jobs += counts_vec

            if not np.array_equal(bin_counts, machine_counts):
                raise ValueError(
                    f"Slot {slot_idx} machine counts do not match bin list."
                )

            if not np.array_equal(assigned_jobs, required_jobs.reshape(-1)):
                raise ValueError(
                    f"Slot {slot_idx} does not cover the required job counts."
                )

            expected_machine_vector = np.maximum(
                expected_machine_vector, machine_counts
            )

        machine_vector = np.asarray(self.machine_vector, dtype=int).reshape(-1)
        if machine_vector.shape[0] != M:
            raise ValueError(
                f"machine_vector must have one entry per machine type ({M})."
            )
        if not np.array_equal(machine_vector, expected_machine_vector):
            raise ValueError("machine_vector does not match the per-slot usage.")

        if purchase_vec is not None and running_vec is not None:
            if baseline_vec is None:
                baseline_vec = np.zeros(M, dtype=int)

            incremental_purchases = np.maximum(machine_vector - baseline_vec, 0)
            recomputed = float(np.dot(purchase_vec, incremental_purchases))
            running_total = float(
                sum(
                    float(
                        np.dot(running_vec, np.asarray(slot.machine_counts, dtype=int))
                    )
                    for slot in self.time_slot_solutions
                )
            )
            recomputed += running_total
            if not np.isclose(recomputed, float(self.total_cost), atol=atol):
                raise ValueError("total_cost does not match purchase+running costs.")

    def average_remaining_capacity(self) -> np.ndarray:
        """
        Average the remaining-capacity vectors across all used bins.

        Returns
        -------
        np.ndarray
            Length-K vector with the mean remaining capacity per resource.
            Returns an empty vector when no bins are active.
        """

        total_remaining: np.ndarray | None = None
        bin_count = 0
        for slot in self.time_slot_solutions:
            for bin_info in slot.bins:
                remaining = np.asarray(bin_info.remaining_capacity, dtype=float)
                if total_remaining is None:
                    total_remaining = np.zeros_like(remaining, dtype=float)
                total_remaining += remaining
                bin_count += 1

        if total_remaining is None or bin_count == 0:
            return np.zeros(0, dtype=float)

        return (total_remaining / bin_count).reshape(-1)

    def __str__(self) -> str:
        def _format_vector(vec: np.ndarray) -> str:
            flat = np.asarray(vec).reshape(-1)
            if flat.size == 0:
                return "[]"
            entries = []
            for value in flat:
                num = float(value)
                if np.isfinite(num) and num.is_integer():
                    entries.append(str(int(num)))
                else:
                    entries.append(f"{num:g}")
            return f"[{', '.join(entries)}]"

        lines = [f"Total cost: {self.total_cost:.2f}"]
        lines.append("Machine selection (count):")

        machine_counts = np.asarray(self.machine_vector).reshape(-1)
        max_len = machine_counts.size
        if max_len == 0:
            lines.append("  (no machines)")
        else:
            for idx in range(max_len):
                count = int(machine_counts[idx]) if idx < machine_counts.size else 0
                lines.append(f"  Type {idx}: {count} selected")

        lines.append("Time slots:")
        if not self.time_slot_solutions:
            lines.append("  (no time slots)")
        else:
            for slot_idx, slot in enumerate(self.time_slot_solutions):
                lines.append(f"  Slot {slot_idx}:")
                lines.append(
                    f"    Machine counts: {_format_vector(slot.machine_counts)}"
                )
                if not slot.bins:
                    lines.append("    No machines scheduled.")
                    continue

                for bin_idx, bin_info in enumerate(slot.bins):
                    lines.append(f"    Machine {bin_idx} (type {bin_info.bin_type}):")
                    lines.append(
                        f"      Total capacity: {_format_vector(bin_info.capacity)}"
                    )
                    lines.append(
                        f"      Remaining capacity: {_format_vector(bin_info.remaining_capacity)}"
                    )

                    items = [
                        f"{int(count)}x job {job_type}"
                        for job_type, count in enumerate(bin_info.item_counts)
                        if int(count) > 0
                    ]
                    items_str = ", ".join(items) if items else "(empty)"
                    lines.append(f"      Items: {items_str}")

        return "\n".join(lines)


def marginal_cost_bin_selection(
    requirements: np.ndarray,
    purchase_costs: np.ndarray,
    running_costs: np.ndarray,
    purchased_counts: np.ndarray,
    open_counts: np.ndarray,
) -> BinSelectionFn:
    """
    Build a bin selection function that prioritizes lowest marginal cost.

    The returned callable is useful for custom heuristics; the core packing
    routines already apply this marginal-cost rule internally. ``requirements``
    should be ordered the same way that the packing routine iterates over job
    types.
    """

    def _select(job_type: int, capacities: np.ndarray) -> int:
        demand = requirements[:, [job_type]]
        fits_mask = np.all(capacities >= demand, axis=0)
        if not np.any(fits_mask):
            raise ValueError(
                f"Job type {job_type} does not fit in any available machine type."
            )

        best_type = None
        best_key = None
        for bin_type, fits in enumerate(fits_mask):
            if not fits:
                continue
            already_owned = open_counts[bin_type] < purchased_counts[bin_type]
            marginal = running_costs[bin_type]
            if not already_owned:
                marginal += purchase_costs[bin_type]
            key = (marginal, running_costs[bin_type], purchase_costs[bin_type])
            if best_key is None or key < best_key:
                best_key = key
                best_type = bin_type

        if best_type is None:
            raise ValueError(
                f"Failed to choose a machine type for job type {job_type}."
            )

        open_counts[best_type] += 1
        if open_counts[best_type] > purchased_counts[best_type]:
            purchased_counts[best_type] = open_counts[best_type]

        return best_type

    return _select


def _machine_counts_from_bins(
    bins: Sequence[_BaseBinInfo], num_types: int
) -> np.ndarray:
    """Count how many machines of each type are active in a set of bins."""

    counts = np.zeros(num_types, dtype=int)
    for bin_info in bins:
        counts[bin_info.bin_type] += 1
    return counts


def build_time_slot_solution(
    bins: Sequence[_BaseBinInfo],
    num_types: int,
    requirements: np.ndarray,
    running_costs: np.ndarray | None = None,
    resource_weights: np.ndarray | None = None,
) -> TimeSlotSolution:
    """Construct a time-slot solution from a raw bin list."""

    weights = (
        None if resource_weights is None else np.asarray(resource_weights, dtype=float)
    )
    if weights is not None:
        weights = weights.reshape(-1)

    active_bins: list[BinInfo] = []
    for bin_info in bins:
        if int(np.sum(bin_info.item_counts)) == 0:
            continue
        active_bins.append(
            BinInfo(
                bin_type=bin_info.bin_type,
                capacity=bin_info.capacity.copy(),
                remaining_capacity=bin_info.remaining_capacity.copy(),
                item_counts=bin_info.item_counts.copy(),
                resource_weights=None if weights is None else weights.copy(),
            )
        )

    _sort_bins_by_utilization(active_bins, resource_weights, running_costs)
    machine_counts = _machine_counts_from_bins(active_bins, num_types)
    return TimeSlotSolution(machine_counts=machine_counts, bins=active_bins)


def ffd_schedule(
    problem: ProblemInstance,
    bin_selection_method: BinTypeSelectionMethod,
    job_ordering_method: JobTypeOrderingMethod = JobTypeOrderingMethod.SORT_LEX,
) -> ScheduleResult:
    """
    Build a multi-slot schedule by running FFD independently per slot.

    Uses the packing notation fields from ``ProblemInstance``:
    ``capacities`` is the ``(K, M)`` capacity matrix, ``requirements`` is the
    ``(K, J)`` requirement matrix, and ``job_counts`` is either a ``(T, J)``
    matrix of per-slot job counts or a length-``J`` vector. Costs are
    length-``M`` vectors for purchasing and running machine types.

    ``job_ordering_method`` controls how job types are sorted before packing.
    ``purchased_bins`` is an optional length-``M`` vector describing how many
    machines of each type are already purchased. A copy is tracked internally
    and updated as additional purchases are needed while scheduling.
    """

    C = np.asarray(problem.capacities, dtype=float)
    R = np.asarray(problem.requirements, dtype=float)
    L = np.asarray(problem.job_counts, dtype=int)
    purchase_vec = np.asarray(problem.purchase_costs, dtype=float).reshape(-1)
    running_vec = np.asarray(problem.running_costs, dtype=float).reshape(-1)
    resource_weights = np.asarray(problem.resource_weights, dtype=float).reshape(-1)

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

    M = C.shape[1]
    J = R.shape[1]
    if L.shape[1] != J:
        raise ValueError(f"L must contain {J} job-type columns; got {L.shape[1]}.")
    if purchase_vec.shape[0] != M or running_vec.shape[0] != M:
        raise ValueError("Cost vectors must have one entry per machine type.")

    initial_purchased = np.zeros(M, dtype=int)

    time_slot_solutions: list[TimeSlotSolution] = []
    machine_vector = np.zeros(M, dtype=int)
    total_cost = 0.0

    for slot_jobs in L:
        if np.all(slot_jobs == 0):
            slot_solution = TimeSlotSolution(
                machine_counts=np.zeros(M, dtype=int), bins=[]
            )
        else:
            ffd_result = first_fit_sorted(
                C=C,
                R=R,
                purchase_costs=purchase_vec,
                opening_costs=running_vec,
                L=slot_jobs,
                purchased_bins=initial_purchased,
                selection_method=bin_selection_method,
                job_ordering_method=job_ordering_method,
                weights=resource_weights,
            )

            slot_solution = build_time_slot_solution(
                ffd_result.bins,
                M,
                R,
                running_vec,
                resource_weights=resource_weights,
            )

        time_slot_solutions.append(slot_solution)
        machine_vector = np.maximum(machine_vector, slot_solution.machine_counts)
        total_cost += float(np.dot(running_vec, slot_solution.machine_counts))

    return ScheduleResult(
        total_cost=total_cost,
        machine_vector=machine_vector,
        time_slot_solutions=time_slot_solutions,
        purchased_baseline=initial_purchased,
    )


def ffd_weighted_sort_schedule(
    problem: ProblemInstance, bin_selection_method: BinTypeSelectionMethod
) -> ScheduleResult:
    """
    Build a multi-slot schedule by running weighted-sort FFD per slot.
    """
    return ffd_schedule(
        problem,
        bin_selection_method,
        job_ordering_method=JobTypeOrderingMethod.SORT_BY_WEIGHT,
    )


def _sort_bins_by_utilization(
    bins: list[BinInfo],
    resource_weights: np.ndarray | None,
    running_costs: np.ndarray | None = None,
) -> None:
    """Sort bins by weighted remaining capacity (highest first) and running cost."""

    if not bins:
        return

    cost_vector = None
    if running_costs is not None:
        cost_vector = np.asarray(running_costs, dtype=float).reshape(-1)

    def _sort_key(bin_info: BinInfo) -> tuple[float, float]:
        cost_component = (
            -float(cost_vector[bin_info.bin_type]) if cost_vector is not None else 0.0
        )
        utilization = bin_info.utilization(resource_weights)
        return (-utilization, cost_component)

    bins.sort(key=_sort_key)


def _refresh_bin_utilization(bins: Sequence[BinInfo]) -> None:
    """Refresh utilization caches for a collection of bins."""

    for bin_info in bins:
        bin_info.update_utilization_cache()


def _sorted_jobs_for_bin(bin_info: BinInfo, requirements: np.ndarray) -> list[int]:
    """Return a non-increasing ordering of job indices contained in a bin."""

    jobs: list[int] = []
    for job_type, count in enumerate(bin_info.item_counts):
        jobs.extend([job_type] * int(count))

    jobs.sort(
        key=lambda idx: tuple(requirements[:, idx].tolist()),  # lexicographic size
        reverse=True,
    )
    return jobs


def _maybe_downsize_bin(
    bin_info: BinInfo,
    capacities: np.ndarray,
    requirements: np.ndarray,
    running_costs: np.ndarray,
) -> bool:
    """
    Try to swap a bin to a smaller/cheaper type that still fits its contents.

    Returns True if the bin type was changed.
    """

    if int(np.sum(bin_info.item_counts)) == 0:
        return False

    load = requirements @ bin_info.item_counts.reshape(-1, 1)
    current_type = int(bin_info.bin_type)
    current_capacity = capacities[:, [current_type]]
    current_cost = float(running_costs[current_type])
    current_size = float(np.sum(current_capacity))

    best_type = None
    best_key = None
    for candidate_type in range(capacities.shape[1]):
        if candidate_type == current_type:
            continue

        candidate_capacity = capacities[:, [candidate_type]]
        if np.any(load > candidate_capacity):
            continue

        candidate_cost = float(running_costs[candidate_type])
        candidate_size = float(np.sum(candidate_capacity))

        # Require some improvement: either cheaper to run or strictly smaller.
        if candidate_cost >= current_cost and candidate_size >= current_size:
            continue

        key = (candidate_cost, candidate_size)
        if best_key is None or key < best_key:
            best_key = key
            best_type = candidate_type

    if best_type is None:
        return False

    new_capacity = capacities[:, [best_type]].copy()
    bin_info.bin_type = best_type
    bin_info.capacity = new_capacity
    bin_info.remaining_capacity = new_capacity - load
    bin_info.update_utilization_cache()
    return True


def repack_jobs(
    slot_solution: TimeSlotSolution,
    capacities: np.ndarray,
    requirements: np.ndarray,
    running_costs: np.ndarray,
    resource_weights: np.ndarray | None = None,
) -> TimeSlotSolution:
    """
    Heuristic job re-packing as described in ``method.typ``.

    Jobs are iteratively moved from bins with the most weighted remaining capacity
    to bins with less remaining capacity (breaking ties by per-time-slot running
    cost). When a bin becomes empty it is removed, which reduces the powered-on
    machine count for the slot. After moving items out of a source bin, the
    remaining contents are also checked to see if they fit in a cheaper/smaller
    bin type; if so, the bin is downsized and its capacity/remaining state is
    updated.
    """

    if not slot_solution.bins:
        return slot_solution.copy()

    capacities = np.asarray(capacities, dtype=float)
    requirements = np.asarray(requirements, dtype=float)
    running_costs = np.asarray(running_costs, dtype=float).reshape(-1)
    if resource_weights is not None:
        resource_weights = np.asarray(resource_weights, dtype=float).reshape(-1)

    if capacities.ndim != 2:
        raise ValueError("capacities must be a 2D matrix.")
    if capacities.shape[0] != requirements.shape[0]:
        raise ValueError(
            "capacities and requirements must have the same number of resource dimensions."
        )
    if capacities.shape[1] != running_costs.shape[0]:
        raise ValueError(
            "running_costs length must match the number of machine types in capacities."
        )

    bins: list[BinInfo] = [
        BinInfo(
            bin_type=b.bin_type,
            capacity=b.capacity.copy(),
            remaining_capacity=b.remaining_capacity.copy(),
            item_counts=b.item_counts.copy(),
            resource_weights=(
                None if b.resource_weights is None else b.resource_weights.copy()
            ),
        )
        for b in slot_solution.bins
    ]

    _refresh_bin_utilization(bins)
    # _sort_bins_by_utilization(bins, resource_weights, running_costs)

    while True:
        moved = False
        _sort_bins_by_utilization(bins, resource_weights, running_costs)
        for source_idx, source in enumerate(bins):
            if int(np.sum(source.item_counts)) == 0:
                continue

            source_util = source.utilization(resource_weights)
            job_sequence = _sorted_jobs_for_bin(source, requirements)

            for dest_idx in range(len(bins) - 1, source_idx, -1):
                dest = bins[dest_idx]
                if dest is source:
                    continue
                dest_util = dest.utilization(resource_weights)
                if dest_util >= source_util:
                    continue

                job_moved = False
                for job_type in job_sequence:
                    if source.item_counts[job_type] <= 0:
                        continue
                    demand = requirements[:, [job_type]]
                    if np.all(dest.remaining_capacity >= demand):
                        source.remaining_capacity += demand
                        source.item_counts[job_type] -= 1
                        dest.remaining_capacity -= demand
                        dest.item_counts[job_type] += 1
                        source.update_utilization_cache()
                        dest.update_utilization_cache()
                        job_moved = True
                        moved = True

                        if int(np.sum(source.item_counts)) > 0:
                            _maybe_downsize_bin(
                                source, capacities, requirements, running_costs
                            )
                        break

                if int(np.sum(source.item_counts)) == 0:
                    break

                if job_moved:
                    break

            if moved:
                break

        bins = [b for b in bins if int(np.sum(b.item_counts)) > 0]
        if not moved or not bins:
            break

    new_counts = _machine_counts_from_bins(bins, running_costs.shape[0])
    return TimeSlotSolution(machine_counts=new_counts, bins=bins)


def repack_schedule(
    schedule: ScheduleResult,
    capacities: np.ndarray,
    requirements: np.ndarray,
    purchase_costs: np.ndarray,
    running_costs: np.ndarray,
    resource_weights: np.ndarray | None = None,
) -> ScheduleResult:
    """Repack all time slots with ``repack_jobs`` and recompute costs."""

    repacked_slots: list[TimeSlotSolution] = []
    for slot in schedule.time_slot_solutions:
        repacked_slot = repack_jobs(
            slot,
            capacities=capacities,
            requirements=requirements,
            running_costs=running_costs,
            resource_weights=resource_weights,
        )
        _refresh_bin_utilization(repacked_slot.bins)
        repacked_slots.append(repacked_slot)

    purchase_vec = np.asarray(purchase_costs, dtype=float).reshape(-1)
    running_vec = np.asarray(running_costs, dtype=float).reshape(-1)
    machine_vector = np.zeros_like(purchase_vec, dtype=int)
    total_cost = 0.0

    for slot in repacked_slots:
        counts = np.asarray(slot.machine_counts, dtype=int).reshape(-1)
        machine_vector = np.maximum(machine_vector, counts)
        total_cost += float(np.dot(running_vec, counts))

    total_cost += float(np.dot(purchase_vec, machine_vector))

    return ScheduleResult(
        total_cost=total_cost,
        machine_vector=machine_vector,
        time_slot_solutions=repacked_slots,
    )
