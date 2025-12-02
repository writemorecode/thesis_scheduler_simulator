from __future__ import annotations

import csv
from dataclasses import dataclass
from os import PathLike
from typing import List, Sequence, Tuple

import numpy as np

from problem_generation import ProblemInstance
from packing import BinInfo as _BaseBinInfo, BinSelectionFn, first_fit_decreasing


class BinInfo(_BaseBinInfo):
    """Extended bin info with convenience metrics."""

    def utilization(self, requirements: np.ndarray) -> float:
        """
        Maximum utilization ratio across all resource dimensions.

        Parameters
        ----------
        requirements : np.ndarray
            ``(K, J)`` matrix describing resource demand per job type.
        """

        req = np.asarray(requirements, dtype=float)
        if req.ndim != 2:
            raise ValueError("requirements must be a 2D matrix.")

        if req.shape[1] != np.asarray(self.item_counts).reshape(-1).shape[0]:
            raise ValueError(
                "requirements column count must match the size of item_counts."
            )

        return float(self._cached_utilization)


@dataclass
class TimeSlotSolution:
    """Packing state for a single time slot."""

    machine_counts: np.ndarray  # vector z_t
    bins: List[BinInfo]

    def copy(self) -> "TimeSlotSolution":
        """Deep copy helper used before mutating a slot solution."""

        return TimeSlotSolution(
            machine_counts=self.machine_counts.copy(),
            bins=[
                BinInfo(
                    bin_type=b.bin_type,
                    capacity=b.capacity.copy(),
                    remaining_capacity=b.remaining_capacity.copy(),
                    item_counts=b.item_counts.copy(),
                )
                for b in self.bins
            ],
        )


@dataclass
class ScheduleResult:
    """Result container for the scheduler."""

    total_cost: float
    machine_vector: np.ndarray
    upper_bound: np.ndarray | None
    time_slot_solutions: List[TimeSlotSolution]

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

        expected_machine_vector = np.zeros(M, dtype=int)

        for slot_idx, (slot, required_jobs) in enumerate(
            zip(self.time_slot_solutions, job_matrix)
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

        # upper_bound_vec = np.asarray(self.upper_bound, dtype=int).reshape(-1)
        # if upper_bound_vec.shape[0] != M:
        #     raise ValueError("upper_bound must have one entry per machine type.")
        # if np.any(machine_vector > upper_bound_vec):
        #     raise ValueError("machine_vector exceeds the provided upper bound.")

        if purchase_vec is not None and running_vec is not None:
            recomputed = float(np.dot(purchase_vec, machine_vector))
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
        lines.append("Machine selection (count / upper bound):")

        machine_counts = np.asarray(self.machine_vector).reshape(-1)
        bounds = np.asarray(self.upper_bound).reshape(-1)
        max_len = max(machine_counts.size, bounds.size)
        if max_len == 0:
            lines.append("  (no machines)")
        else:
            for idx in range(max_len):
                count = int(machine_counts[idx]) if idx < machine_counts.size else 0
                bound = int(bounds[idx]) if idx < bounds.size else 0
                lines.append(f"  Type {idx}: {count} selected (upper bound {bound})")

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


def dump_schedule_to_csv(
    schedule: ScheduleResult, output_path: str | PathLike[str]
) -> None:
    """Write per-machine assignment information for ``schedule`` to ``output_path``."""

    job_type_count = 0
    for slot in schedule.time_slot_solutions:
        for bin_info in slot.bins:
            if bin_info.item_counts.size > job_type_count:
                job_type_count = int(bin_info.item_counts.size)

    headers = ["time_slot", "machine_type"] + [
        f"job_{idx}" for idx in range(job_type_count)
    ]

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)

        for slot_idx, slot in enumerate(schedule.time_slot_solutions):
            # Rows are emitted in slot order to keep rows grouped by time slot.
            for bin_info in slot.bins:
                raw_counts = bin_info.item_counts.reshape(-1)
                row = [slot_idx, int(bin_info.bin_type)]
                row.extend(
                    int(raw_counts[idx]) if idx < raw_counts.size else 0
                    for idx in range(job_type_count)
                )
                writer.writerow(row)


def _pareto_non_dominated(time_slots: np.ndarray) -> np.ndarray:
    """Return only the time slots that are not strictly Pareto dominated by another slot."""

    if time_slots.size == 0:
        return time_slots

    # Remove duplicate rows first; duplicate slots convey no new information.
    unique_slots, unique_indices = np.unique(time_slots, axis=0, return_index=True)
    unique_slots = unique_slots[np.argsort(unique_indices)]

    keep_mask = np.ones(len(unique_slots), dtype=bool)
    for i in range(len(unique_slots)):
        if not keep_mask[i]:
            continue
        for j in range(len(unique_slots)):
            if i == j:
                continue
            dominates = np.all(unique_slots[j] >= unique_slots[i]) and np.any(
                unique_slots[j] > unique_slots[i]
            )
            if dominates:
                keep_mask[i] = False
                break

    return unique_slots[keep_mask]


def _ffd_single_machine_type(
    capacity: np.ndarray, requirements: np.ndarray, job_counts: np.ndarray
) -> int:
    """
    Wrapper that runs ``first_fit_decreasing`` for a single bin type.

    Parameters
    ----------
    capacity : np.ndarray
        One-dimensional array with the remaining capacity per resource for the bin type.
    requirements : np.ndarray
        Matrix of shape (K, J) containing per-resource demand for each job type.
    job_counts : np.ndarray
        Length-J vector with the number of jobs of each type to place.

    Returns
    -------
    int
        Number of bins opened by the heuristic.
    """

    cap_vec = np.asarray(capacity, dtype=float).reshape(-1, 1)
    req = np.asarray(requirements, dtype=float)
    counts = np.asarray(job_counts, dtype=int).reshape(-1)

    if req.ndim != 2:
        raise ValueError("requirements must be a 2D matrix.")
    if counts.shape[0] != req.shape[1]:
        raise ValueError(
            f"job_counts length must match the number of job types. Expected {req.shape[1]}, got {counts.shape[0]}."
        )

    # Purchase and opening costs do not matter for the upper-bound calculation,
    # so we provide zeros and only look at the number of bins returned.
    purchase_costs = np.zeros(1)
    opening_costs = np.zeros(1)

    result = first_fit_decreasing(
        C=cap_vec,
        R=req,
        purchase_costs=purchase_costs,
        opening_costs=opening_costs,
        L=counts,
    )

    return len(result.bins)


def machines_upper_bound(
    C: np.ndarray, R: np.ndarray, time_slots: np.ndarray
) -> np.ndarray:
    """
    Compute the component-wise upper bound on the required number of machines of each type.

    The procedure follows the method described in ``method.typ``:
    1. Filter time slots down to the Pareto frontier so only the most demanding slots remain.
    2. For every remaining slot and machine type, drop job types that physically cannot
       run on that machine (any requirement exceeds the machine capacity in some dimension).
    3. Run FFD using only that machine type to get an upper bound for the slot.
    4. Take the maximum bound across all surviving time slots.

    Parameters
    ----------
    C : np.ndarray
        Matrix of shape (K, M) with the per-resource capacity of each machine type.
    R : np.ndarray
        Matrix of shape (K, J) with the per-resource demand of each job type.
    time_slots : np.ndarray
        Either a length-J vector or a (T, J) matrix with counts of jobs scheduled in
        each time slot.

    Returns
    -------
    np.ndarray
        Length-M vector where entry ``i`` is the upper bound on machines of type ``i``.
    """

    capacities = np.asarray(C, dtype=float)
    requirements = np.asarray(R, dtype=float)
    if capacities.ndim != 2 or requirements.ndim != 2:
        raise ValueError("C and R must be 2D matrices.")

    K, M = capacities.shape
    K_req, J = requirements.shape
    if K != K_req:
        raise ValueError(
            f"C and R must describe the same number of resources. Got {K} and {K_req}."
        )

    slot_array = np.asarray(time_slots, dtype=int)
    if slot_array.ndim == 1:
        if slot_array.shape[0] != J:
            raise ValueError(
                f"time_slots vector length must match the number of job types ({J})."
            )
        slot_array = slot_array.reshape(1, -1)
    elif slot_array.ndim == 2:
        if slot_array.shape[1] != J:
            raise ValueError(
                f"time_slots matrix must have {J} columns (one per job type), got {slot_array.shape[1]}."
            )
    else:
        raise ValueError("time_slots must be a vector or a 2D matrix.")

    # Remove empty slots early to avoid unnecessary FFD runs.
    nonzero_mask = np.any(slot_array > 0, axis=1)
    slot_array = slot_array[nonzero_mask]
    if slot_array.size == 0:
        return np.zeros(M, dtype=int)

    candidate_slots = _pareto_non_dominated(slot_array)
    upper_bound = np.zeros(M, dtype=int)

    for m in range(M):
        capacity = capacities[:, m]
        # Identify job types that cannot run on machine type m.
        unsupported_jobs = np.any(requirements > capacity.reshape(-1, 1), axis=0)
        max_bins = 0
        for slot in candidate_slots:
            lambda_vec = slot.copy()
            lambda_vec[unsupported_jobs] = 0
            bins_needed = _ffd_single_machine_type(capacity, requirements, lambda_vec)
            if bins_needed > max_bins:
                max_bins = bins_needed
        upper_bound[m] = max_bins

    return upper_bound


def _ffd_sorted_requirements(requirements: np.ndarray) -> np.ndarray:
    """Mirror the job-type ordering used by ``first_fit_decreasing``."""

    req = np.asarray(requirements, dtype=float)
    if req.ndim != 2:
        raise ValueError("requirements must be a 2D matrix.")

    _, J = req.shape
    if J == 0 or req.shape[0] == 0:
        return req.copy()

    sort_keys = -req[::-1, :]
    sorted_indices = np.lexsort(sort_keys)
    return req[:, sorted_indices]


def marginal_cost_bin_selection(
    requirements: np.ndarray,
    purchase_costs: np.ndarray,
    running_costs: np.ndarray,
    purchased_counts: np.ndarray,
    open_counts: np.ndarray,
) -> BinSelectionFn:
    """
    Build a bin selection function that prioritizes lowest marginal cost.

    The returned callable matches the ``bin_selection_fn`` expected by
    ``first_fit_decreasing`` and assumes ``requirements`` is ordered the same way
    that function iterates over job types.
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


def marginal_cost_initial_solution(
    C: np.ndarray,
    R: np.ndarray,
    L: np.ndarray,
    purchase_costs: np.ndarray,
    running_costs: np.ndarray,
) -> np.ndarray:
    """
    Greedy initial machine vector based on marginal costs.

    Each time slot is packed with ``first_fit_decreasing`` using a marginal-cost
    bin selection rule that prefers reusing already-purchased machines before
    buying new ones. Machines purchased in earlier time slots only incur the
    running cost when reused. The resulting vector is the maximum number of
    simultaneously open machines of each type across all slots.
    """

    capacities = np.asarray(C, dtype=float)
    requirements = np.asarray(R, dtype=float)
    job_matrix = np.asarray(L, dtype=int)

    if capacities.ndim != 2 or requirements.ndim != 2:
        raise ValueError("C and R must be 2D matrices.")

    K, M = capacities.shape
    K_req, J = requirements.shape
    if K != K_req:
        raise ValueError(
            f"C and R must describe the same number of resources. Got {K} and {K_req}."
        )

    if job_matrix.ndim == 1:
        job_matrix = job_matrix.reshape(1, -1)
    elif job_matrix.ndim != 2:
        raise ValueError("L must be a vector or a 2D matrix.")
    if job_matrix.shape[1] != J:
        raise ValueError(
            f"L must contain {J} job-type columns; got {job_matrix.shape[1]}."
        )

    purchase_vec = np.asarray(purchase_costs, dtype=float).reshape(-1)
    running_vec = np.asarray(running_costs, dtype=float).reshape(-1)
    if purchase_vec.shape[0] != M or running_vec.shape[0] != M:
        raise ValueError("Cost vectors must have one entry per machine type.")

    sorted_requirements = _ffd_sorted_requirements(requirements)
    purchased_counts = np.zeros(M, dtype=int)

    for slot_jobs in job_matrix:
        open_counts = np.zeros(M, dtype=int)
        selection_fn = marginal_cost_bin_selection(
            sorted_requirements,
            purchase_vec,
            running_vec,
            purchased_counts,
            open_counts,
        )

        first_fit_decreasing(
            capacities,
            requirements,
            purchase_vec,
            running_vec,
            L=slot_jobs,
            bin_selection_fn=selection_fn,
        )

    return purchased_counts


def _machine_counts_from_bins(bins: Sequence[BinInfo], num_types: int) -> np.ndarray:
    """Count how many machines of each type are active in a set of bins."""

    counts = np.zeros(num_types, dtype=int)
    for bin_info in bins:
        counts[bin_info.bin_type] += 1
    return counts


def _build_time_slot_solution(
    bins: Sequence[BinInfo],
    num_types: int,
    requirements: np.ndarray,
    running_costs: np.ndarray | None = None,
) -> TimeSlotSolution:
    """Construct a time-slot solution from a raw bin list."""

    active_bins: List[BinInfo] = []
    for bin_info in bins:
        if int(np.sum(bin_info.item_counts)) == 0:
            continue
        active_bins.append(
            BinInfo(
                bin_type=bin_info.bin_type,
                capacity=bin_info.capacity.copy(),
                remaining_capacity=bin_info.remaining_capacity.copy(),
                item_counts=bin_info.item_counts.copy(),
            )
        )

    _sort_bins_by_utilization(active_bins, requirements, running_costs)
    machine_counts = _machine_counts_from_bins(active_bins, num_types)
    return TimeSlotSolution(machine_counts=machine_counts, bins=active_bins)


def _pack_time_slot_jobs(
    machine_vector: np.ndarray,
    capacities: np.ndarray,
    requirements: np.ndarray,
    job_counts: np.ndarray,
    running_costs: np.ndarray | None = None,
) -> TimeSlotSolution:
    """Pack a single time slot using FFD with the supplied machine vector."""

    machine_vector = np.asarray(machine_vector, dtype=int).reshape(-1)
    job_counts = np.asarray(job_counts, dtype=int).reshape(-1)

    M = capacities.shape[1]
    J = requirements.shape[1]
    if machine_vector.shape[0] != M:
        raise ValueError(
            f"machine_vector must have one entry per machine type ({M}); got {machine_vector.shape[0]}."
        )
    if job_counts.shape[0] != J:
        raise ValueError(
            f"job_counts must have one entry per job type ({J}); got {job_counts.shape[0]}."
        )

    running_cost_vec = None
    if running_costs is not None:
        running_cost_vec = np.asarray(running_costs, dtype=float).reshape(-1)
        if running_cost_vec.shape[0] != M:
            raise ValueError(
                f"running_costs must have one entry per machine type ({M}); got {running_cost_vec.shape[0]}."
            )

    zeros = np.zeros(M, dtype=float)
    result = first_fit_decreasing(
        capacities,
        requirements,
        purchase_costs=zeros,
        opening_costs=zeros,
        L=job_counts,
        opened_bins=machine_vector,
    )

    slot_solution = _build_time_slot_solution(
        result.bins, M, requirements, running_cost_vec
    )
    if np.any(slot_solution.machine_counts > machine_vector):
        # print(
        #    "WARNING: Packing exceeded provided machine vector; the upper bound is insufficient."
        # )
        # raise ValueError(
        #     "Packing exceeded provided machine vector; the upper bound is insufficient."
        # )
        pass

    return slot_solution


def _pack_time_slot_marginal_cost(
    capacities: np.ndarray,
    requirements: np.ndarray,
    job_counts: np.ndarray,
    purchase_costs: np.ndarray,
    running_costs: np.ndarray,
    purchased_counts: np.ndarray,
    sorted_requirements: np.ndarray | None = None,
) -> TimeSlotSolution:
    """Pack a time slot using marginal-cost bin selection.

    Machines already listed in ``purchased_counts`` are pre-opened for the
    slot. Additional machines are chosen using
    :func:`marginal_cost_bin_selection`, which updates ``purchased_counts``
    in-place when new machine types are acquired.
    """

    if sorted_requirements is None:
        sorted_requirements = _ffd_sorted_requirements(requirements)

    purchase_vec = np.asarray(purchase_costs, dtype=float).reshape(-1)
    running_vec = np.asarray(running_costs, dtype=float).reshape(-1)

    open_counts = np.asarray(purchased_counts, dtype=int).copy()
    selection_fn = marginal_cost_bin_selection(
        sorted_requirements,
        purchase_vec,
        running_vec,
        purchased_counts,
        open_counts,
    )

    result = first_fit_decreasing(
        capacities,
        requirements,
        purchase_vec,
        running_vec,
        L=job_counts,
        opened_bins=open_counts,
        bin_selection_fn=selection_fn,
    )

    purchased_counts[:] = np.maximum(purchased_counts, open_counts)

    return _build_time_slot_solution(
        result.bins, capacities.shape[1], requirements, running_vec
    )


def _pack_all_time_slots(
    machine_vector: np.ndarray,
    capacities: np.ndarray,
    requirements: np.ndarray,
    job_matrix: np.ndarray,
    running_costs: np.ndarray | None = None,
) -> List[TimeSlotSolution]:
    """Pack every time slot independently using the provided machine vector."""

    solutions = []
    for job_counts in job_matrix:
        solutions.append(
            _pack_time_slot_jobs(
                machine_vector,
                capacities,
                requirements,
                job_counts,
                running_costs=running_costs,
            )
        )
    return solutions


def _sort_bins_by_utilization(
    bins: List[BinInfo],
    requirements: np.ndarray,
    running_costs: np.ndarray | None = None,
) -> None:
    """Sort bins in-place by utilization and running cost tie breaker."""

    if not bins:
        return

    cost_vector = None
    if running_costs is not None:
        cost_vector = np.asarray(running_costs, dtype=float).reshape(-1)

    def _sort_key(bin_info: BinInfo) -> Tuple[float, float]:
        cost_component = (
            -float(cost_vector[bin_info.bin_type]) if cost_vector is not None else 0.0
        )
        return (bin_info.utilization(requirements), cost_component)

    bins.sort(key=_sort_key)


def _sorted_jobs_for_bin(bin_info: BinInfo, requirements: np.ndarray) -> List[int]:
    """Return a non-increasing ordering of job indices contained in a bin."""

    jobs: List[int] = []
    for job_type, count in enumerate(bin_info.item_counts):
        jobs.extend([job_type] * int(count))

    jobs.sort(
        key=lambda idx: tuple(requirements[:, idx].tolist()),  # lexicographic size
        reverse=True,
    )
    return jobs


def _solution_cost(
    time_slot_solutions: Sequence[TimeSlotSolution],
    purchase_costs: np.ndarray,
    running_costs: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Compute total cost and implied machine vector from a schedule."""

    if not time_slot_solutions:
        machine_vec = np.zeros_like(purchase_costs, dtype=int)
        return 0.0, machine_vec

    z_matrix = np.vstack([slot.machine_counts for slot in time_slot_solutions])
    machine_vec = np.max(z_matrix, axis=0)
    purchase_total = float(np.dot(purchase_costs, machine_vec))
    running_total = float(
        sum(
            float(np.dot(running_costs, slot.machine_counts))
            for slot in time_slot_solutions
        )
    )
    return purchase_total + running_total, machine_vec


def _solution_signature(
    time_slot_solutions: Sequence[TimeSlotSolution],
) -> Tuple[Tuple[Tuple[int, ...], Tuple[Tuple[int, Tuple[int, ...]], ...]], ...]:
    """Hashable signature representing an entire schedule."""

    slot_entries = []
    for slot in time_slot_solutions:
        machine_counts = tuple(int(v) for v in slot.machine_counts.tolist())
        bin_entries = tuple(
            (bin_info.bin_type, tuple(int(v) for v in bin_info.item_counts.tolist()))
            for bin_info in slot.bins
        )
        slot_entries.append((machine_counts, bin_entries))
    return tuple(slot_entries)


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
) -> TimeSlotSolution:
    """
    Heuristic job re-packing as described in ``method.typ``.

    Jobs are iteratively moved from the least utilized bin to bins with higher
    utilization (breaking ties by per-time-slot running cost). When a bin becomes
    empty it is removed, which reduces the powered-on machine count for the slot.
    After moving items out of a source bin, the remaining contents are also checked
    to see if they fit in a cheaper/smaller bin type; if so, the bin is downsized
    and its capacity/remaining state is updated.
    """

    if not slot_solution.bins:
        return slot_solution.copy()

    capacities = np.asarray(capacities, dtype=float)
    requirements = np.asarray(requirements, dtype=float)
    running_costs = np.asarray(running_costs, dtype=float).reshape(-1)

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

    bins: List[BinInfo] = [
        BinInfo(
            bin_type=b.bin_type,
            capacity=b.capacity.copy(),
            remaining_capacity=b.remaining_capacity.copy(),
            item_counts=b.item_counts.copy(),
        )
        for b in slot_solution.bins
    ]

    _sort_bins_by_utilization(bins, requirements, running_costs)

    while True:
        moved = False
        _sort_bins_by_utilization(bins, requirements, running_costs)
        for source_idx, source in enumerate(bins):
            if int(np.sum(source.item_counts)) == 0:
                continue

            source_util = source.utilization(requirements)
            job_sequence = _sorted_jobs_for_bin(source, requirements)

            for dest_idx in range(len(bins) - 1, source_idx, -1):
                dest = bins[dest_idx]
                if dest is source:
                    continue
                dest_util = dest.utilization(requirements)
                if dest_util <= source_util:
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


def schedule_jobs_peak_demand(problem: ProblemInstance) -> ScheduleResult:
    """
    Schedule jobs by seeding with the most resource-intensive time slot.

    The algorithm computes the per-slot resource totals ``A = R @ L.T`` and picks
    the column with the largest total demand. That slot is packed first with no
    pre-opened machines; the resulting machine vector is then used as the set of
    open bins for packing every other time slot (which may open additional
    machines if needed). The final machine vector is the maximum count across
    all packed slots.
    """

    capacities = np.asarray(problem.capacities, dtype=float)
    requirements = np.asarray(problem.requirements, dtype=float)
    job_matrix = np.asarray(problem.job_counts, dtype=int)

    purchase_costs = np.asarray(problem.purchase_costs, dtype=float).reshape(-1)
    running_costs = np.asarray(problem.running_costs, dtype=float).reshape(-1)

    if capacities.ndim != 2 or requirements.ndim != 2:
        raise ValueError("C and R must be 2D matrices.")
    if capacities.shape[0] != requirements.shape[0]:
        raise ValueError("C and R must have the same number of resource dimensions.")

    if job_matrix.ndim == 1:
        job_matrix = job_matrix.reshape(1, -1)
    elif job_matrix.ndim != 2:
        raise ValueError("L must be a vector or a 2D matrix.")
    if np.any(job_matrix < 0):
        raise ValueError("Job counts in L must be non-negative.")

    K, M = capacities.shape
    K_req, J = requirements.shape
    if K != K_req:
        raise ValueError(
            f"C and R must describe the same number of resources. Got {K} and {K_req}."
        )
    if job_matrix.shape[1] != J:
        raise ValueError(
            f"L must contain {J} job-type columns; got {job_matrix.shape[1]}."
        )
    if purchase_costs.shape[0] != M or running_costs.shape[0] != M:
        raise ValueError("Cost vectors must have one entry per machine type.")

    num_slots = job_matrix.shape[0]
    if num_slots == 0:
        machine_vector = np.zeros(M, dtype=int)
        return ScheduleResult(
            total_cost=0.0,
            machine_vector=machine_vector,
            upper_bound=None,
            time_slot_solutions=[],
        )

    demand_matrix = requirements @ job_matrix.T  # Shape (K, T)
    per_slot_totals = np.sum(demand_matrix, axis=0)
    peak_slot_index = int(np.argsort(-per_slot_totals)[0])

    initial_solution = _pack_time_slot_jobs(
        machine_vector=np.zeros(M, dtype=int),
        capacities=capacities,
        requirements=requirements,
        job_counts=job_matrix,
        running_costs=running_costs,
    )
    base_machine_vector = initial_solution.machine_counts

    time_slot_solutions: List[TimeSlotSolution] = []
    for slot_idx in range(num_slots):
        if slot_idx == peak_slot_index:
            slot_solution = initial_solution
            time_slot_solutions.append(slot_solution)
        else:
            slot_solution = _pack_time_slot_jobs(
                base_machine_vector,
                capacities,
                requirements,
                job_matrix[slot_idx],
                running_costs=running_costs,
                purchase_costs=purchase_costs,
            )
            repacked_slot_solution = repack_jobs(
                slot_solution, capacities, requirements, running_costs
            )
            time_slot_solutions.append(repacked_slot_solution)

    total_cost, machine_vector = _solution_cost(
        time_slot_solutions, purchase_costs, running_costs
    )

    return ScheduleResult(
        total_cost=total_cost,
        machine_vector=machine_vector,
        upper_bound=None,
        time_slot_solutions=time_slot_solutions,
    )


def schedule_jobs(
    problem: ProblemInstance,
    max_iterations: int = 25,
    initial_method: str = "upper_bound",
) -> ScheduleResult:
    """
    Scheduler loop implementing the method outlined in ``method.typ``.

    Parameters follow the fields in :class:`ProblemInstance`: ``capacities`` and
    ``requirements`` describe machine capacities and job requirements, ``job_counts``
    lists scheduled jobs per time slot, ``purchase_costs`` corresponds to
    :math:`c^p`, and ``running_costs`` to the per time slot operational cost
    :math:`c^r`. The ``initial_method`` parameter selects how to construct the first
    machine vector; supported values are ``"upper_bound"`` and ``"marginal_cost"``.
    """

    capacities = np.asarray(problem.capacities, dtype=float)
    requirements = np.asarray(problem.requirements, dtype=float)
    job_matrix = np.asarray(problem.job_counts, dtype=int)

    purchase_costs = problem.purchase_costs
    running_costs = problem.running_costs

    if capacities.ndim != 2 or requirements.ndim != 2:
        raise ValueError("C and R must be 2D matrices.")
    if capacities.shape[0] != requirements.shape[0]:
        raise ValueError("C and R must have the same number of resource dimensions.")

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

    purchase_costs = np.asarray(purchase_costs, dtype=float).reshape(-1)
    running_costs = np.asarray(running_costs, dtype=float).reshape(-1)
    if purchase_costs.shape[0] != M or running_costs.shape[0] != M:
        raise ValueError("Cost vectors must have one entry per machine type.")

    upper_bound = machines_upper_bound(capacities, requirements, job_matrix)

    if initial_method == "upper_bound":
        machine_vector = upper_bound.copy()
    elif initial_method == "marginal_cost":
        machine_vector = marginal_cost_initial_solution(
            capacities, requirements, job_matrix, purchase_costs, running_costs
        )
    else:
        raise ValueError(
            f'Unknown initial_method "{initial_method}". '
            'Supported options are "upper_bound" and "marginal_cost".'
        )

    if np.any(machine_vector > upper_bound):
        raise ValueError(
            "Initial machine vector exceeds the computed upper bound; "
            "the marginal cost heuristic produced an inconsistent result."
        )

    time_slot_solutions = _pack_all_time_slots(
        machine_vector, capacities, requirements, job_matrix, running_costs
    )
    total_cost, machine_vector = _solution_cost(
        time_slot_solutions, purchase_costs, running_costs
    )

    current_signature = _solution_signature(time_slot_solutions)
    seen_solutions = {current_signature}

    for _ in range(max_iterations):
        neighbor_solutions = [
            repack_jobs(slot, capacities, requirements, running_costs)
            for slot in time_slot_solutions
        ]
        neighbor_signature = _solution_signature(neighbor_solutions)
        if neighbor_signature == current_signature:
            break

        neighbor_cost, neighbor_machine_vector = _solution_cost(
            neighbor_solutions, purchase_costs, running_costs
        )

        if neighbor_signature in seen_solutions:
            break

        if np.any(neighbor_machine_vector > upper_bound):
            break

        if not np.array_equal(neighbor_machine_vector, machine_vector):
            neighbor_solutions = _pack_all_time_slots(
                neighbor_machine_vector,
                capacities,
                requirements,
                job_matrix,
                running_costs,
            )
            neighbor_cost, neighbor_machine_vector = _solution_cost(
                neighbor_solutions, purchase_costs, running_costs
            )
            neighbor_signature = _solution_signature(neighbor_solutions)

        if neighbor_signature in seen_solutions:
            break

        if neighbor_cost < total_cost:
            time_slot_solutions = neighbor_solutions
            total_cost = neighbor_cost
            machine_vector = neighbor_machine_vector
            seen_solutions.add(neighbor_signature)
            current_signature = neighbor_signature
        else:
            break

    return ScheduleResult(
        total_cost=total_cost,
        machine_vector=machine_vector,
        upper_bound=upper_bound,
        time_slot_solutions=time_slot_solutions,
    )


def schedule_jobs_marginal_cost(
    problem: ProblemInstance,
    max_iterations: int = 25,
) -> ScheduleResult:
    """
    Version of 'schedule_jobs' which uses marginal-cost based method for
    computing initial machine vector.
    """

    capacities = np.asarray(problem.capacities, dtype=float)
    requirements = np.asarray(problem.requirements, dtype=float)
    job_matrix = np.asarray(problem.job_counts, dtype=int)

    purchase_costs = problem.purchase_costs
    running_costs = problem.running_costs

    if capacities.ndim != 2 or requirements.ndim != 2:
        raise ValueError("C and R must be 2D matrices.")
    if capacities.shape[0] != requirements.shape[0]:
        raise ValueError("C and R must have the same number of resource dimensions.")

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

    purchase_costs = np.asarray(purchase_costs, dtype=float).reshape(-1)
    running_costs = np.asarray(running_costs, dtype=float).reshape(-1)
    if purchase_costs.shape[0] != M or running_costs.shape[0] != M:
        raise ValueError("Cost vectors must have one entry per machine type.")

    machine_vector = marginal_cost_initial_solution(
        capacities, requirements, job_matrix, purchase_costs, running_costs
    )
    time_slot_solutions = _pack_all_time_slots(
        machine_vector, capacities, requirements, job_matrix, running_costs
    )
    total_cost, machine_vector = _solution_cost(
        time_slot_solutions, purchase_costs, running_costs
    )

    current_signature = _solution_signature(time_slot_solutions)
    seen_solutions = {current_signature}

    for _ in range(max_iterations):
        neighbor_solutions = [
            repack_jobs(slot, capacities, requirements, running_costs)
            for slot in time_slot_solutions
        ]
        neighbor_signature = _solution_signature(neighbor_solutions)
        if neighbor_signature == current_signature:
            break

        neighbor_cost, neighbor_machine_vector = _solution_cost(
            neighbor_solutions, purchase_costs, running_costs
        )

        if neighbor_signature in seen_solutions:
            break

        if not np.array_equal(neighbor_machine_vector, machine_vector):
            neighbor_solutions = _pack_all_time_slots(
                neighbor_machine_vector,
                capacities,
                requirements,
                job_matrix,
                running_costs,
            )
            neighbor_cost, neighbor_machine_vector = _solution_cost(
                neighbor_solutions, purchase_costs, running_costs
            )
            neighbor_signature = _solution_signature(neighbor_solutions)

        if neighbor_signature in seen_solutions:
            break

        if neighbor_cost < total_cost:
            time_slot_solutions = neighbor_solutions
            total_cost = neighbor_cost
            machine_vector = neighbor_machine_vector
            seen_solutions.add(neighbor_signature)
            current_signature = neighbor_signature
        else:
            break

    return ScheduleResult(
        total_cost=total_cost,
        machine_vector=machine_vector,
        upper_bound=None,
        time_slot_solutions=time_slot_solutions,
    )
