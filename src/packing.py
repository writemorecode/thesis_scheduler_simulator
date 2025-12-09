from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, List, Sequence

import numpy as np


@dataclass
class BinInfo:
    """State for a single bin during packing."""

    bin_type: int
    capacity: np.ndarray
    remaining_capacity: np.ndarray
    item_counts: np.ndarray
    _cached_utilization: float = field(init=False, repr=False, default=0.0)

    def __post_init__(self) -> None:
        self.update_utilization_cache()

    def __str__(self) -> str:
        parts = [
            f"Bin type: {self.bin_type}",
            f"Total capacity:\n{self.capacity}",
            f"Remaining capacity:\n{self.remaining_capacity}",
            f"Item counts:\n{self.item_counts}",
        ]
        return "\n".join(parts)

    def _compute_utilization(self) -> float:
        """Maximum utilization across resources based on capacity usage."""

        capacity = np.asarray(self.capacity, dtype=float).reshape(-1, 1)
        remaining = np.asarray(self.remaining_capacity, dtype=float).reshape(-1, 1)
        load = capacity - remaining

        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.divide(
                load, capacity, out=np.zeros_like(load), where=capacity > 0
            )

        return float(ratios.max()) if ratios.size else 0.0

    def update_utilization_cache(self) -> None:
        """Refresh cached utilization after bin contents change."""

        self._cached_utilization = self._compute_utilization()


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


BinSelectionFn = Callable[[int, np.ndarray], int]


def _prepare_vector(vec: np.ndarray, length: int, name: str) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(-1)
    if arr.shape[0] != length:
        raise ValueError(f"{name} must have length {length}, got {arr.shape[0]}")
    return arr


def _prepare_count_vector(
    vec: np.ndarray | Sequence[int], length: int, name: str
) -> np.ndarray:
    arr = np.asarray(vec, dtype=int).reshape(-1)
    if arr.shape[0] != length:
        raise ValueError(f"{name} must have length {length}, got {arr.shape[0]}")
    if np.any(arr < 0):
        raise ValueError(f"{name} entries must be non-negative.")
    return arr


def _sort_items_decreasing(
    R: np.ndarray, L: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort items in non-increasing lexicographic order of demand."""

    R_array = np.asarray(R, dtype=float)
    if R_array.ndim != 2:
        raise ValueError("R must be a 2D matrix.")

    L_array = np.asarray(L, dtype=int).reshape(-1)
    if L_array.shape[0] != R_array.shape[1]:
        raise ValueError(
            f"L must have one entry per item type. Expected {R_array.shape[1]}, got {L_array.shape[0]}."
        )
    if np.any(L_array < 0):
        raise ValueError("L entries must be non-negative.")

    _, J = R_array.shape
    if J == 0:
        sorted_indices = np.arange(J, dtype=int)
        return R_array.copy(), L_array.copy(), sorted_indices

    if R_array.shape[0] == 0:
        sorted_indices = np.arange(J, dtype=int)
    else:
        sort_keys = -R_array[::-1, :]
        sorted_indices = np.lexsort(sort_keys)

    R_sorted = R_array[:, sorted_indices]
    L_sorted = L_array[sorted_indices]
    return R_sorted, L_sorted, sorted_indices


def _select_bin_type_marginal_cost(
    item_type: int,
    demand: np.ndarray,
    capacities: np.ndarray,
    purchase_costs: np.ndarray,
    opening_costs: np.ndarray,
    purchased_counts: np.ndarray,
    open_counts: np.ndarray,
) -> tuple[int, bool]:
    """
    Choose the bin type with the lowest marginal cost for ``demand``.

    Returns the selected bin type and whether a new purchase is required.
    """

    fits_mask = np.all(capacities >= demand, axis=0)
    if not np.any(fits_mask):
        raise ValueError(
            f"Item type {item_type} does not fit in any available bin type."
        )

    best_type = None
    best_key = None
    best_requires_purchase = False

    for bin_type, fits in enumerate(fits_mask):
        if not fits:
            continue

        requires_purchase = open_counts[bin_type] >= purchased_counts[bin_type]
        marginal_cost = (
            opening_costs[bin_type]
            if not requires_purchase
            else purchase_costs[bin_type] + opening_costs[bin_type]
        )
        key = (marginal_cost, opening_costs[bin_type], purchase_costs[bin_type])
        if best_key is None or key < best_key:
            best_key = key
            best_type = bin_type
            best_requires_purchase = requires_purchase

    if best_type is None:
        raise ValueError(
            f"Failed to choose a bin type for item type {item_type} using marginal costs."
        )

    return best_type, best_requires_purchase


def _select_bin_type_random(
    item_type: int,
    demand: np.ndarray,
    capacities: np.ndarray,
    purchased_counts: np.ndarray,
    open_counts: np.ndarray,
) -> tuple[int, bool]:
    """Pick a random feasible bin type for ``demand``."""

    fits_mask = np.all(capacities >= demand, axis=0)
    if not np.any(fits_mask):
        raise ValueError(
            f"Item type {item_type} does not fit in any available bin type."
        )

    feasible_types = np.flatnonzero(fits_mask)
    bin_type = int(np.random.default_rng().choice(feasible_types))
    requires_purchase = open_counts[bin_type] >= purchased_counts[bin_type]
    return bin_type, requires_purchase


def first_fit(
    C: np.ndarray,
    R: np.ndarray,
    purchase_costs: np.ndarray,
    opening_costs: np.ndarray,
    L: np.ndarray,
    opened_bins: np.ndarray | Sequence[int] | None = None,
    purchased_bins: np.ndarray | Sequence[int] | None = None,
    initial: bool = False,
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
    opened_bins : array-like of shape (M,), optional
        Pre-opened bin counts per type. These bins are available before packing starts
        and incur only the opening cost if they were already purchased.
    purchased_bins : array-like of shape (M,), optional
        Bins already purchased but not necessarily opened. When provided, the marginal
        cost rule will prefer reusing these machines (opening cost only) before buying
        additional ones. This vector is updated in place when a new purchase occurs.

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

    if not np.all(R >= 0):
        raise ValueError("All job demand values must be non-negative")

    purchase_costs = _prepare_vector(purchase_costs, M, "purchase_costs")
    opening_costs = _prepare_vector(opening_costs, M, "opening_costs")
    L = np.asarray(L, dtype=int).reshape(-1)
    if L.shape[0] != J:
        raise ValueError(f"L must have length {J}, got {L.shape[0]}.")
    if np.any(L < 0):
        raise ValueError("Job counts in L must be non-negative.")
    if np.any(L < 0):
        raise ValueError("Job counts in L must be non-negative.")

    opened_bins_vec = (
        np.zeros(M, dtype=int)
        if opened_bins is None
        else _prepare_count_vector(opened_bins, M, "opened_bins")
    )

    purchased_counts = (
        np.zeros(M, dtype=int)
        if purchased_bins is None
        else _prepare_count_vector(purchased_bins, M, "purchased_bins")
    )
    initial_purchased = purchased_counts.copy()
    purchased_counts[:] = np.maximum(purchased_counts, opened_bins_vec)

    open_counts = opened_bins_vec.copy()
    preopened_extra = np.maximum(open_counts - initial_purchased, 0)

    bins: List[BinInfo] = []
    total_cost = float(np.dot(open_counts, opening_costs))
    if np.any(preopened_extra):
        total_cost += float(np.dot(preopened_extra, purchase_costs))

    def _create_bin(bin_type: int) -> BinInfo:
        capacity = C[:, [bin_type]].copy()
        bin_info = BinInfo(
            bin_type=bin_type,
            capacity=capacity.copy(),
            remaining_capacity=capacity.copy(),
            item_counts=np.zeros(J, dtype=int),
        )
        bins.append(bin_info)
        return bin_info

    for bin_type, count in enumerate(open_counts):
        for _ in range(int(count)):
            _create_bin(bin_type)

    for j in range(J):
        demand = R[:, [j]]

        for _ in range(int(L[j])):
            placed = False
            for bin_info in bins:
                if np.all(bin_info.remaining_capacity >= demand):
                    bin_info.remaining_capacity -= demand
                    bin_info.item_counts[j] += 1
                    bin_info.update_utilization_cache()
                    placed = True
                    break

            if placed:
                continue

            if not initial:
                bin_type, requires_purchase = _select_bin_type_marginal_cost(
                    j,
                    demand,
                    C,
                    purchase_costs,
                    opening_costs,
                    purchased_counts,
                    open_counts,
                )
            else:
                bin_type, requires_purchase = _select_bin_type_random(
                    j,
                    demand,
                    C,
                    purchased_counts,
                    open_counts,
                )

            incremental_cost = (
                float(opening_costs[bin_type])
                if not requires_purchase
                else float(purchase_costs[bin_type] + opening_costs[bin_type])
            )
            total_cost += incremental_cost
            open_counts[bin_type] += 1
            if open_counts[bin_type] > purchased_counts[bin_type]:
                purchased_counts[bin_type] = open_counts[bin_type]

            bin_info = _create_bin(bin_type)
            bin_info.remaining_capacity -= demand
            bin_info.item_counts[j] += 1
            bin_info.update_utilization_cache()

    return BinPackingResult(total_cost=total_cost, bins=bins)


def first_fit_decreasing(
    C: np.ndarray,
    R: np.ndarray,
    purchase_costs: np.ndarray,
    opening_costs: np.ndarray,
    L: np.ndarray,
    opened_bins: np.ndarray | Sequence[int] | None = None,
    purchased_bins: np.ndarray | Sequence[int] | None = None,
    initial: bool = False,
) -> BinPackingResult:
    """
    Run first-fit after sorting job types in non-increasing order of resource demand.

    Job requirement columns and their corresponding counts are permuted in tandem using
    a lexicographic ordering on the resource dimensions so that larger jobs are packed
    before smaller ones. The returned bin contents are mapped back to the original job
    ordering before being returned.

    Parameters mirror :func:`first_fit`; ``opened_bins`` and ``purchased_bins`` are
    forwarded directly without modification.
    """

    R_sorted, L_sorted, sorted_indices = _sort_items_decreasing(R, L)
    J = R_sorted.shape[1]

    result = first_fit(
        C,
        R_sorted,
        purchase_costs,
        opening_costs,
        L_sorted,
        opened_bins=opened_bins,
        purchased_bins=purchased_bins,
        initial=initial,
    )

    if J > 0:
        inverse_perm = np.empty_like(sorted_indices)
        inverse_perm[sorted_indices] = np.arange(J)
        for bin_info in result.bins:
            bin_info.item_counts = bin_info.item_counts[inverse_perm]

    return result


def best_fit(
    C: np.ndarray,
    R: np.ndarray,
    purchase_costs: np.ndarray,
    opening_costs: np.ndarray,
    L: np.ndarray,
    opened_bins: np.ndarray | Sequence[int] | None = None,
    purchased_bins: np.ndarray | Sequence[int] | None = None,
) -> BinPackingResult:
    """
    Run the best-fit heterogeneous multidimensional bin packing heuristic.

    Items are placed into the existing bin that yields the highest utilization
    (i.e., the tightest fit) while respecting all capacity dimensions. When no
    current bin fits, a new bin type is chosen using a marginal-cost rule that
    prefers already-purchased bin types before buying new ones.
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

    if not np.all(R >= 0):
        raise ValueError("All job demand values must be non-negative")

    purchase_costs = _prepare_vector(purchase_costs, M, "purchase_costs")
    opening_costs = _prepare_vector(opening_costs, M, "opening_costs")
    L = np.asarray(L, dtype=int).reshape(-1)
    if L.shape[0] != J:
        raise ValueError(f"L must have length {J}, got {L.shape[0]}.")

    opened_bins_vec = (
        np.zeros(M, dtype=int)
        if opened_bins is None
        else _prepare_count_vector(opened_bins, M, "opened_bins")
    )

    purchased_counts = (
        np.zeros(M, dtype=int)
        if purchased_bins is None
        else _prepare_count_vector(purchased_bins, M, "purchased_bins")
    )
    initial_purchased = purchased_counts.copy()
    purchased_counts[:] = np.maximum(purchased_counts, opened_bins_vec)

    open_counts = opened_bins_vec.copy()
    preopened_extra = np.maximum(open_counts - initial_purchased, 0)

    bins: List[BinInfo] = []
    total_cost = float(np.dot(open_counts, opening_costs))
    if np.any(preopened_extra):
        total_cost += float(np.dot(preopened_extra, purchase_costs))

    def _create_bin(bin_type: int) -> BinInfo:
        capacity = C[:, [bin_type]].copy()
        bin_info = BinInfo(
            bin_type=bin_type,
            capacity=capacity.copy(),
            remaining_capacity=capacity.copy(),
            item_counts=np.zeros(J, dtype=int),
        )
        bins.append(bin_info)
        return bin_info

    for bin_type, count in enumerate(open_counts):
        for _ in range(int(count)):
            _create_bin(bin_type)

    for j in range(J):
        demand = R[:, [j]]

        for _ in range(int(L[j])):
            best_idx = -1
            best_utilization = -np.inf
            best_remaining_sum = np.inf

            for idx, bin_info in enumerate(bins):
                if not np.all(bin_info.remaining_capacity >= demand):
                    continue

                candidate_remaining = bin_info.remaining_capacity - demand
                load = bin_info.capacity - candidate_remaining
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratios = np.divide(
                        load,
                        bin_info.capacity,
                        out=np.zeros_like(load),
                        where=bin_info.capacity > 0,
                    )
                utilization = float(ratios.max()) if ratios.size else 0.0
                remaining_sum = float(candidate_remaining.sum())

                if utilization > best_utilization or (
                    np.isclose(utilization, best_utilization)
                    and remaining_sum < best_remaining_sum
                ):
                    best_idx = idx
                    best_utilization = utilization
                    best_remaining_sum = remaining_sum

            if best_idx != -1:
                bin_info = bins[best_idx]
                bin_info.remaining_capacity -= demand
                bin_info.item_counts[j] += 1
                bin_info.update_utilization_cache()
                continue

            bin_type, requires_purchase = _select_bin_type_marginal_cost(
                j,
                demand,
                C,
                purchase_costs,
                opening_costs,
                purchased_counts,
                open_counts,
            )

            incremental_cost = (
                float(opening_costs[bin_type])
                if not requires_purchase
                else float(purchase_costs[bin_type] + opening_costs[bin_type])
            )
            total_cost += incremental_cost
            open_counts[bin_type] += 1
            if open_counts[bin_type] > purchased_counts[bin_type]:
                purchased_counts[bin_type] = open_counts[bin_type]

            bin_info = _create_bin(bin_type)
            bin_info.remaining_capacity -= demand
            bin_info.item_counts[j] += 1
            bin_info.update_utilization_cache()

    return BinPackingResult(total_cost=total_cost, bins=bins)


def best_fit_decreasing(
    C: np.ndarray,
    R: np.ndarray,
    purchase_costs: np.ndarray,
    opening_costs: np.ndarray,
    L: np.ndarray,
    opened_bins: np.ndarray | Sequence[int] | None = None,
    purchased_bins: np.ndarray | Sequence[int] | None = None,
) -> BinPackingResult:
    """
    Run best-fit after sorting job types in non-increasing order of resource demand.

    Sorting mirrors :func:`first_fit_decreasing`; bin placement uses
    :func:`best_fit`. Parameters mirror :func:`best_fit`; ``opened_bins`` and
    ``purchased_bins`` are forwarded without modification.
    """

    R_sorted, L_sorted, sorted_indices = _sort_items_decreasing(R, L)
    J = R_sorted.shape[1]

    result = best_fit(
        C,
        R_sorted,
        purchase_costs,
        opening_costs,
        L_sorted,
        opened_bins=opened_bins,
        purchased_bins=purchased_bins,
    )

    if J > 0:
        inverse_perm = np.empty_like(sorted_indices)
        inverse_perm[sorted_indices] = np.arange(J)
        for bin_info in result.bins:
            bin_info.item_counts = bin_info.item_counts[inverse_perm]

    return result
