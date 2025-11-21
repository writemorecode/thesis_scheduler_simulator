from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np


@dataclass
class BinInfo:
    """State for a single bin during packing."""

    bin_type: int
    capacity: np.ndarray
    remaining_capacity: np.ndarray
    item_counts: np.ndarray

    def __str__(self) -> str:
        parts = [
            f"Bin type: {self.bin_type}",
            f"Total capacity:\n{self.capacity}",
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


BinSelectionFn = Callable[[int, np.ndarray], int]


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
    opened_bins: np.ndarray | Sequence[int] | None = None,
    bin_selection_fn: BinSelectionFn | None = None,
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
        and their purchase+opening cost is charged up front.
    bin_selection_fn : Callable[[int, np.ndarray], int], optional
        Function that chooses which bin type to open when none of the existing bins fit.
        Receives the item type index and the capacity matrix ``C`` and should return the
        index of the bin type to open. Defaults to selecting the cheapest (by opening
        cost) bin type that can accommodate the item.

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

    per_bin_costs = purchase_costs + opening_costs

    opened_bins_vec = None
    if opened_bins is not None:
        opened_bins_vec = np.asarray(opened_bins, dtype=int).reshape(-1)
        if opened_bins_vec.shape[0] != M:
            raise ValueError(
                f"opened_bins must have one entry per bin type ({M}); got {opened_bins_vec.shape[0]}."
            )
        if np.any(opened_bins_vec < 0):
            raise ValueError("opened_bins entries must be non-negative.")

    bins: List[BinInfo] = []
    total_cost = 0.0

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

    if opened_bins_vec is not None:
        total_cost += float(np.dot(opened_bins_vec, per_bin_costs))
        for bin_type, count in enumerate(opened_bins_vec):
            for _ in range(int(count)):
                _create_bin(bin_type)

    def _default_bin_selection(item_type: int, C: np.ndarray) -> int:
        running_costs_sorted_indicies = np.argsort(opening_costs)
        C_sorted = C[:, running_costs_sorted_indicies]
        demand_vec = R[:, [item_type]].reshape(-1, 1)
        feasible_bin_types = np.all(C_sorted >= demand_vec, axis=0)
        if not np.any(feasible_bin_types):
            raise ValueError(
                f"Item type {item_type} does not fit in any available bin type."
            )
        best_bin_type = np.argmax(feasible_bin_types)
        bin_vector = C_sorted[:, best_bin_type].reshape(-1, 1)
        if not np.all(bin_vector >= demand_vec):
            raise ValueError(
                f"Selected invalid bin type {best_bin_type} for item type {item_type}"
            )
        return running_costs_sorted_indicies[best_bin_type]

    select_bin_type = bin_selection_fn or _default_bin_selection

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

            bin_type = select_bin_type(j, C)
            if not (0 <= bin_type < M):
                raise ValueError(
                    f"Bin selection function returned invalid bin type {bin_type} for item type {j}."
                )
            if not np.all(C[:, [bin_type]] >= demand):
                raise ValueError(
                    f"Bin selection function chose type {bin_type}, but item type {j} does not fit in it."
                )
            bin_info = _create_bin(bin_type)
            total_cost += float(per_bin_costs[bin_type])
            bin_info.remaining_capacity -= demand
            bin_info.item_counts[j] += 1

    return BinPackingResult(total_cost=total_cost, bins=bins)


def first_fit_decreasing(
    C: np.ndarray,
    R: np.ndarray,
    purchase_costs: np.ndarray,
    opening_costs: np.ndarray,
    L: np.ndarray,
    opened_bins: np.ndarray | Sequence[int] | None = None,
    bin_selection_fn: BinSelectionFn | None = None,
) -> BinPackingResult:
    """
    Run first-fit after sorting job types in non-increasing order of resource demand.

    Job requirement columns and their corresponding counts are permuted in tandem using
    a lexicographic ordering on the resource dimensions so that larger jobs are packed
    before smaller ones. The returned bin contents are mapped back to the original job
    ordering before being returned.

    Parameters mirror :func:`first_fit`; ``opened_bins`` and ``bin_selection_fn`` are
    forwarded directly without modification.
    """

    R_array = np.asarray(R, dtype=float)
    if R_array.ndim != 2:
        raise ValueError("R must be a 2D matrix.")

    L_array = np.asarray(L, dtype=int).reshape(-1)
    if L_array.shape[0] != R_array.shape[1]:
        raise ValueError(
            f"L must have one entry per item type. Expected {R_array.shape[1]}, got {L_array.shape[0]}."
        )

    _, J = R_array.shape
    if J == 0:
        sorted_indices = np.arange(J, dtype=int)
        R_sorted = R_array.copy()
        L_sorted = L_array.copy()
    else:
        if R_array.shape[0] == 0:
            sorted_indices = np.arange(J, dtype=int)
        else:
            sort_keys = -R_array[::-1, :]
            sorted_indices = np.lexsort(sort_keys)

        R_sorted = R_array[:, sorted_indices]
        L_sorted = L_array[sorted_indices]

    result = first_fit(
        C,
        R_sorted,
        purchase_costs,
        opening_costs,
        L_sorted,
        opened_bins=opened_bins,
        bin_selection_fn=bin_selection_fn,
    )

    if J > 0:
        inverse_perm = np.empty_like(sorted_indices)
        inverse_perm[sorted_indices] = np.arange(J)
        for bin_info in result.bins:
            bin_info.item_counts = bin_info.item_counts[inverse_perm]

    return result
