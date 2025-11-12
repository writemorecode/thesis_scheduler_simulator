from __future__ import annotations
from collections import Counter

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class BinInfo:
    """State for a single bin during packing."""

    bin_type: int
    remaining_capacity: np.ndarray
    item_counts: np.ndarray

    def __str__(self) -> str:
        parts = [
            f"Bin type: {self.bin_type}",
            f"Remaining capacity:\n{self.remaining_capacity}",
            f"Item counts:\n{self.item_counts}",
        ]
        return "\n".join(parts)


@dataclass
class BinPackingResult:
    """Container for the packing outcome."""

    total_cost: float
    bins: List[BinInfo]

    def __str__(self):
        cost_text = f"Total cost: {self.total_cost}"
        bins_text = ""
        bin_counts = Counter(bin_info.bin_type for bin_info in self.bins).most_common()
        for bin_type, count in bin_counts:
            bins_text += f"Bin type {bin_type}: {count}\n"
        return f"{cost_text}\n{bins_text}"


def _prepare_vector(vec: np.ndarray, length: int, name: str) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(-1)
    if arr.shape[0] != length:
        raise ValueError(f"{name} must have length {length}, got {arr.shape[0]}")
    return arr


def first_fit(
    C: np.ndarray,
    R: np.ndarray,
    purchase_costs: np.ndarray,
    opening_costs: np.ndarray,
    L: np.ndarray,
) -> BinPackingResult:
    """
    Run the first-fit heterogeneous multidimensional bin packing heuristic.

    Parameters
    ----------
    C : np.ndarray
        Bin capacity matrix with shape (K, M). Each column is a bin type.
    R : np.ndarray
        Item requirement matrix with shape (K, J). Each column is an item type.
    purchase_costs : np.ndarray
        Length M column vector with the cost of purchasing each bin type.
    opening_costs : np.ndarray
        Length M column vector with the opening cost for each bin type.
    L : np.ndarray
        Length J column vector with the number of items (per type) to pack.

    Returns
    -------
    BinPackingResult
        total_cost : float
            Sum of purchase and opening costs for all bins used.
        bins : list of dict
            Each entry contains the bin type, remaining capacity, and per-type item counts.
    """

    C = np.asarray(C, dtype=float)
    R = np.asarray(R, dtype=float)

    if C.ndim != 2 or R.ndim != 2:
        raise ValueError("C and R must be 2D matrices.")

    K, M = C.shape
    K_items, J = R.shape

    if K != K_items:
        raise ValueError(
            f"Bin and item matrices must have the same number of rows (dimensions). Got {K} and {K_items}."
        )

    purchase_costs = _prepare_vector(purchase_costs, M, "purchase_costs")
    opening_costs = _prepare_vector(opening_costs, M, "opening_costs")
    L = np.asarray(L, dtype=int).reshape(-1)
    if L.shape[0] != J:
        raise ValueError(f"L must have length {J}, got {L.shape[0]}.")

    bins: List[BinInfo] = []
    total_cost = 0.0

    def _create_bin(bin_type: int) -> BinInfo:
        capacity = C[:, [bin_type]].copy()
        bin_info = BinInfo(
            bin_type=bin_type,
            remaining_capacity=capacity,
            item_counts=np.zeros(J, dtype=int),
        )
        bins.append(bin_info)
        return bin_info

    for j in range(J):
        demand = R[:, [j]]
        if np.any(demand < 0):
            raise ValueError(
                f"Item requirements cannot be negative. Found negative entries in item type {j}."
            )

        for _ in range(int(L[j])):
            placed = False
            for bin_info in bins:
                if np.all(bin_info.remaining_capacity >= demand):
                    bin_info.remaining_capacity -= demand
                    bin_info.item_counts[j] += 1
                    placed = True
                    break

            if placed:
                continue

            for bin_type in range(M):
                if np.all(C[:, [bin_type]] >= demand):
                    bin_info = _create_bin(bin_type)
                    total_cost += float(
                        purchase_costs[bin_type] + opening_costs[bin_type]
                    )
                    bin_info.remaining_capacity -= demand
                    bin_info.item_counts[j] += 1
                    placed = True
                    break

            if not placed:
                raise ValueError(
                    f"Item type {j} does not fit in any available bin type."
                )

    return BinPackingResult(total_cost=total_cost, bins=bins)


def first_fit_decreasing(
    C: np.ndarray,
    R: np.ndarray,
    purchase_costs: np.ndarray,
    opening_costs: np.ndarray,
    L: np.ndarray,
) -> BinPackingResult:
    """
    Run first-fit after sorting item requirements per dimension in non-increasing order.

    The sorting uses ``np.sort`` (ascending) followed by ``np.fliplr`` to flip the
    columns, yielding a decreasing order for each row.
    """

    R_array = np.asarray(R, dtype=float)
    if R_array.ndim != 2:
        raise ValueError("R must be a 2D matrix.")

    L_array = np.asarray(L, dtype=int).reshape(-1)
    if L_array.shape[0] != R_array.shape[1]:
        raise ValueError(
            f"L must have one entry per item type. Expected {R_array.shape[1]}, got {L_array.shape[0]}."
        )

    # Sort ascending along each row (dimension) then flip columns for decreasing order.
    R_sorted = np.fliplr(np.sort(R_array.copy(), axis=1))

    return first_fit(C, R_sorted, purchase_costs, opening_costs, L_array)


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
    Wrapper that runs ``first_fit`` for a single bin type.

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

    result = first_fit(
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
