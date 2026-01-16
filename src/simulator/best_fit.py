from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from simulator.algorithms import ScheduleResult, build_time_slot_solution
from simulator.packing import BinInfo, BinPackingResult, sort_items_by_weight
from simulator.problem import ProblemInstance


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


def _select_open_bin(
    demand_vec: np.ndarray,
    remaining: np.ndarray,
    bin_types: np.ndarray,
    opening_costs: np.ndarray,
    weights: np.ndarray,
    remaining_items: int,
    eps: float = 1e-12,
) -> tuple[int | None, int]:
    if remaining.shape[1] == 0:
        return None, 0

    demand_flat = np.asarray(demand_vec, dtype=float).reshape(-1)
    positive = demand_flat > 0
    if not np.any(positive):
        return 0, remaining_items

    ratios = remaining[positive, :] / demand_flat[positive, None]
    max_add = np.floor(np.min(ratios, axis=0) + eps).astype(int)
    fits_mask = max_add >= 1
    if not np.any(fits_mask):
        return None, 0

    fit_indices = np.nonzero(fits_mask)[0]
    max_add_fit = max_add[fits_mask]
    place_counts = np.minimum(max_add_fit, remaining_items)

    # Score bins by weighted squared slack after placing as many as possible.
    remaining_after = remaining[:, fits_mask] - demand_flat[:, None] * place_counts
    weighted_slack = weights[:, None] * remaining_after**2
    scores = np.sum(weighted_slack, axis=0)

    costs = opening_costs[bin_types[fit_indices]]
    order = np.lexsort((fit_indices, costs, scores))
    chosen_local = int(order[0])
    chosen_idx = int(fit_indices[chosen_local])
    return chosen_idx, int(place_counts[chosen_local])


def _select_new_bin_type(
    demand_vec: np.ndarray,
    capacities: np.ndarray,
    purchase_costs: np.ndarray,
    opening_costs: np.ndarray,
    purchased_counts: np.ndarray,
    open_counts: np.ndarray,
    weights: np.ndarray,
    remaining_items: int,
    eps: float = 1e-12,
) -> tuple[int, bool, int]:
    demand_flat = np.asarray(demand_vec, dtype=float).reshape(-1)
    positive = demand_flat > 0
    if not np.any(positive):
        cheapest = int(np.argmin(opening_costs))
        return (
            cheapest,
            open_counts[cheapest] >= purchased_counts[cheapest],
            remaining_items,
        )

    fits_mask = np.all(capacities >= demand_flat[:, None], axis=0)
    if not np.any(fits_mask):
        raise ValueError("Demand does not fit in any available bin type.")

    best_type = None
    best_key = None
    best_requires_purchase = False
    best_place = 1

    for bin_type, fits in enumerate(fits_mask):
        if not fits:
            continue

        requires_purchase = open_counts[bin_type] >= purchased_counts[bin_type]
        incremental_cost = (
            float(opening_costs[bin_type])
            if not requires_purchase
            else float(purchase_costs[bin_type] + opening_costs[bin_type])
        )

        capacity_vec = capacities[:, bin_type]
        ratios = capacity_vec[positive] / demand_flat[positive]
        max_fit = int(np.floor(np.min(ratios) + eps))
        max_fit = max(1, max_fit)
        place_count = min(remaining_items, max_fit)
        remaining_after = capacity_vec - demand_flat * place_count

        # Normalize both cost and slack by weighted capacity to compare bin types.
        capacity_weight = float(np.dot(weights, capacity_vec))
        capacity_weight = max(capacity_weight, eps)
        slack_score = float(np.dot(weights, remaining_after**2)) / capacity_weight
        key = (slack_score, incremental_cost, bin_type)

        if best_key is None or key < best_key:
            best_key = key
            best_type = bin_type
            best_requires_purchase = requires_purchase
            best_place = place_count

    if best_type is None:
        raise ValueError("Failed to select a bin type for new placement.")

    return int(best_type), best_requires_purchase, int(best_place)


def bfd_weighted_best_fit(
    C: np.ndarray,
    R: np.ndarray,
    purchase_costs: np.ndarray,
    opening_costs: np.ndarray,
    L: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    opened_bins: np.ndarray | Sequence[int] | None = None,
    purchased_bins: np.ndarray | Sequence[int] | None = None,
) -> BinPackingResult:
    """
    Weighted best-fit packing with cost-aware bin-type selection.

    Items are sorted by weighted demand, then packed into open bins by minimizing
    weighted squared slack after placement. New bin types are chosen by a combined
    cost and slack score normalized by weighted capacity.
    """

    C = np.asarray(C, dtype=float)
    R = np.asarray(R, dtype=float)

    if C.ndim != 2 or R.ndim != 2:
        raise ValueError("C and R must be 2D matrices.")

    K, M = C.shape
    K_items, J = R.shape

    if K_items != K:
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

    weight_vec = (
        np.ones(K, dtype=float)
        if weights is None
        else np.asarray(weights, dtype=float).reshape(-1)
    )
    if weight_vec.shape[0] != K:
        raise ValueError(f"weights must have length {K}, got {weight_vec.shape[0]}.")

    R_sorted, L_sorted, sorted_indices = sort_items_by_weight(R, L, weight_vec)
    J_sorted = R_sorted.shape[1]

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

    bins: list[BinInfo] = []
    total_cost = float(np.dot(open_counts, opening_costs))
    if np.any(preopened_extra):
        total_cost += float(np.dot(preopened_extra, purchase_costs))

    bin_types = np.repeat(np.arange(M, dtype=int), open_counts)
    if bin_types.size:
        remaining = C[:, bin_types].copy()
    else:
        remaining = np.zeros((K, 0), dtype=float)
    B = np.zeros((J_sorted, bin_types.size), dtype=int)

    for j in range(J_sorted):
        remaining_items = int(L_sorted[j])
        if remaining_items == 0:
            continue

        demand_vec = R_sorted[:, j].astype(float, copy=False)
        demand = demand_vec.reshape(-1, 1)

        while remaining_items > 0:
            chosen_idx, place_count = _select_open_bin(
                demand_vec,
                remaining,
                bin_types,
                opening_costs,
                weight_vec,
                remaining_items,
            )

            if chosen_idx is not None and place_count > 0:
                remaining[:, chosen_idx] -= demand_vec * place_count
                B[j, chosen_idx] += place_count
                remaining_items -= place_count
                continue

            # No open bin can take the item(s); open a new bin type.
            bin_type, requires_purchase, place_count = _select_new_bin_type(
                demand_vec,
                C,
                purchase_costs,
                opening_costs,
                purchased_counts,
                open_counts,
                weight_vec,
                remaining_items,
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

            new_remaining = (C[:, [bin_type]] - demand * place_count).copy()
            remaining = np.hstack((remaining, new_remaining))
            bin_types = np.concatenate((bin_types, np.array([bin_type], dtype=int)))

            new_B = np.zeros((J_sorted, 1), dtype=int)
            new_B[j, 0] = place_count
            B = np.hstack((B, new_B))
            remaining_items -= place_count

    if J_sorted > 0:
        inverse_perm = np.empty_like(sorted_indices)
        inverse_perm[sorted_indices] = np.arange(J_sorted)
        B = B[inverse_perm, :]

    for i, bin_type in enumerate(bin_types.tolist()):
        capacity = C[:, [bin_type]].copy()
        bin_info = BinInfo(
            bin_type=int(bin_type),
            capacity=capacity.copy(),
            remaining_capacity=remaining[:, [i]].copy(),
            item_counts=B[:, i].copy(),
        )
        bins.append(bin_info)

    return BinPackingResult(total_cost=total_cost, bins=bins)


def bfd_schedule(problem: ProblemInstance) -> ScheduleResult:
    """
    Build a multi-slot schedule using bfd_weighted_best_fit packing.
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

    time_slot_solutions = []
    machine_vector = np.zeros(M, dtype=int)
    total_cost = 0.0

    for slot_jobs in L:
        if np.all(slot_jobs == 0):
            slot_solution = build_time_slot_solution(
                [], M, R, running_vec, resource_weights=resource_weights
            )
        else:
            packing_result = bfd_weighted_best_fit(
                C=C,
                R=R,
                purchase_costs=purchase_vec,
                opening_costs=running_vec,
                L=slot_jobs,
                purchased_bins=initial_purchased,
                weights=resource_weights,
            )
            slot_solution = build_time_slot_solution(
                packing_result.bins,
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
