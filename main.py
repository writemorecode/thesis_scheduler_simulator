from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class BinPackingResult:
    """Container for the packing outcome."""

    total_cost: float
    bins: List[dict]

    def __str__(self):
        cost_text = f"Total cost: {self.total_cost}"
        bins_text = ""
        for bin in self.bins:
            bin_type = bin['bin_type']
            remaining_capacity = bin['remaining_capacity']
            item_counts = bin['item_counts']
            bins_text += f"Bin type: {bin_type}\n" 
            bins_text += f"Remaining capacity:\n{remaining_capacity}\n"
            bins_text += f"Item counts:\n{item_counts}\n"
            bins_text += "\n"
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

    bins: List[dict] = []
    total_cost = 0.0

    def _create_bin(bin_type: int) -> dict:
        capacity = C[:, [bin_type]].copy()
        bin_info = {
            "bin_type": bin_type,
            "remaining_capacity": capacity,
            "item_counts": np.zeros(J, dtype=int),
        }
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
                if np.all(bin_info["remaining_capacity"] >= demand):
                    bin_info["remaining_capacity"] -= demand
                    bin_info["item_counts"][j] += 1
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
                    bin_info["remaining_capacity"] -= demand
                    bin_info["item_counts"][j] += 1
                    placed = True
                    break

            if not placed:
                raise ValueError(
                    f"Item type {j} does not fit in any available bin type."
                )

    return BinPackingResult(total_cost=total_cost, bins=bins)


def example():
    C = np.array([[10, 15], [8, 12]], dtype=float)  # 2 dimensions, 2 bin types
    R = np.array([[6, 4], [4, 3]], dtype=float)  # 2 item types
    purchase_costs = np.array([5.0, 7.0])
    opening_costs = np.array([1.0, 1.5])
    L = np.array([2, 3])

    print(f"Bins:\n{C}")
    print(f"Items:\n{R}\n")

    result = first_fit(C, R, purchase_costs, opening_costs, L)
    print(result)


if __name__ == "__main__":
    example()
