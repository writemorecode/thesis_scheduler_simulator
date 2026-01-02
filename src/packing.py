from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


@dataclass
class BinInfo:
    """State for a single bin during packing."""

    bin_type: int
    capacity: np.ndarray
    remaining_capacity: np.ndarray
    item_counts: np.ndarray
    resource_weights: np.ndarray | None = field(default=None, repr=False)
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
        """Weighted remaining capacity based on resource_weights."""

        if self.resource_weights is None:
            return 0.0

        weights = np.asarray(self.resource_weights, dtype=float).reshape(-1)
        remaining = np.asarray(self.remaining_capacity, dtype=float).reshape(-1)
        if weights.shape[0] != remaining.shape[0]:
            raise ValueError(
                "resource_weights must match remaining_capacity length "
                f"({remaining.shape[0]}), got {weights.shape[0]}."
            )

        return float(np.dot(weights, remaining))

    def update_utilization_cache(
        self, resource_weights: np.ndarray | None = None
    ) -> None:
        """Refresh cached utilization after bin contents change."""

        if resource_weights is not None:
            weights = np.asarray(resource_weights, dtype=float).reshape(-1)
            capacity_len = np.asarray(self.capacity, dtype=float).reshape(-1).shape[0]
            if weights.shape[0] != capacity_len:
                raise ValueError(
                    f"resource_weights must have length {capacity_len}, got {weights.shape[0]}."
                )
            self.resource_weights = weights
        self._cached_utilization = self._compute_utilization()


@dataclass
class BinPackingResult:
    """Container for the packing outcome."""

    total_cost: float
    bins: list[BinInfo]

    def __str__(self):
        cost_text = f"Total cost: {self.total_cost}"
        bins_text = ""
        bin_counts = Counter(bin_info.bin_type for bin_info in self.bins).most_common()
        for bin_type, count in bin_counts:
            bins_text += f"Bin type {bin_type}: {count}\n"
        return f"{cost_text}\n{bins_text}"


BinSelectionFn = Callable[[int, np.ndarray], int]


class BinTypeSelectionMethod(Enum):
    MARGINAL_COST = "marginal cost"
    LARGEST = "largest"
    SMALLEST = "smallest"
    SLACK = "weighted slack"


class JobTypeOrderingMethod(Enum):
    SORT_LEX = "sort lex"
    SORT_BY_WEIGHT = "sort by weight"


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


def sort_items_by_weight(
    R: np.ndarray, L: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort item types by weighted demand (largest first)."""

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

    weight_vec = np.asarray(weights, dtype=float).reshape(-1)
    if weight_vec.shape[0] != R_array.shape[0]:
        raise ValueError(
            f"weights must have length {R_array.shape[0]}, got {weight_vec.shape[0]}."
        )

    weighted_sizes = weight_vec @ R_array
    sorted_indices = np.argsort(-weighted_sizes, kind="mergesort")

    R_sorted = R_array[:, sorted_indices]
    L_sorted = L_array[sorted_indices]
    return R_sorted, L_sorted, sorted_indices


def _sort_job_types(
    R: np.ndarray,
    L: np.ndarray,
    ordering_method: JobTypeOrderingMethod,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    if ordering_method == JobTypeOrderingMethod.SORT_LEX:
        R_sorted, L_sorted, sorted_indices = _sort_items_decreasing(R, L)
        return R_sorted, L_sorted, sorted_indices, weights

    if ordering_method == JobTypeOrderingMethod.SORT_BY_WEIGHT:
        R_array = np.asarray(R, dtype=float)
        if R_array.ndim != 2:
            raise ValueError("R must be a 2D matrix.")

        weight_vec = (
            np.ones(R_array.shape[0], dtype=float)
            if weights is None
            else np.asarray(weights, dtype=float).reshape(-1)
        )
        if weight_vec.shape[0] != R_array.shape[0]:
            raise ValueError(
                f"weights must have length {R_array.shape[0]}, got {weight_vec.shape[0]}."
            )

        R_sorted, L_sorted, sorted_indices = sort_items_by_weight(
            R_array, L, weight_vec
        )
        return R_sorted, L_sorted, sorted_indices, weight_vec

    raise ValueError(f"Unknown job ordering method: {ordering_method!r}")


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


def _select_bin_type_slack(
    item_type: int,
    demand: np.ndarray,
    capacities: np.ndarray,
    purchase_costs: np.ndarray,
    opening_costs: np.ndarray,
    purchased_counts: np.ndarray,
    open_counts: np.ndarray,
    weights: np.ndarray | None,
    remaining_items: int,
    eps: float = 1e-12,
) -> tuple[int, bool]:
    """
    Choose the bin type with the lowest weighted slack score after placement.
    """

    demand_flat = np.asarray(demand, dtype=float).reshape(-1)
    positive = demand_flat > 0
    if not np.any(positive):
        cheapest = int(np.argmin(opening_costs))
        requires_purchase = open_counts[cheapest] >= purchased_counts[cheapest]
        return cheapest, requires_purchase

    fits_mask = np.all(capacities >= demand_flat[:, None], axis=0)
    if not np.any(fits_mask):
        raise ValueError(
            f"Item type {item_type} does not fit in any available bin type."
        )

    if weights is None:
        weight_vec = np.ones(capacities.shape[0], dtype=float)
    else:
        weight_vec = np.asarray(weights, dtype=float).reshape(-1)
        if weight_vec.shape[0] != capacities.shape[0]:
            raise ValueError(
                "weights must have length "
                f"{capacities.shape[0]}, got {weight_vec.shape[0]}."
            )

    best_type = None
    best_key = None
    best_requires_purchase = False

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

        capacity_weight = float(np.dot(weight_vec, capacity_vec))
        capacity_weight = max(capacity_weight, eps)
        slack_score = float(np.dot(weight_vec, remaining_after**2)) / capacity_weight
        key = (slack_score, incremental_cost, bin_type)

        if best_key is None or key < best_key:
            best_key = key
            best_type = bin_type
            best_requires_purchase = requires_purchase

    if best_type is None:
        raise ValueError(
            f"Failed to choose a bin type for item type {item_type} using slack rule."
        )

    return int(best_type), best_requires_purchase


def _select_bin_type(
    selection_method: BinTypeSelectionMethod,
    item_type: int,
    demand: np.ndarray,
    capacities: np.ndarray,
    purchase_costs: np.ndarray,
    opening_costs: np.ndarray,
    purchased_counts: np.ndarray,
    open_counts: np.ndarray,
    weights: np.ndarray | None = None,
    remaining_items: int = 1,
) -> tuple[int, bool]:
    if selection_method == BinTypeSelectionMethod.MARGINAL_COST:
        return _select_bin_type_marginal_cost(
            item_type,
            demand,
            capacities,
            purchase_costs,
            opening_costs,
            purchased_counts,
            open_counts,
        )

    if selection_method == BinTypeSelectionMethod.LARGEST:
        bin_type = _select_bin_type_largest(
            item_type, demand, capacities, purchase_costs, opening_costs
        )
        requires_purchase = open_counts[bin_type] >= purchased_counts[bin_type]
        return bin_type, requires_purchase

    if selection_method == BinTypeSelectionMethod.SMALLEST:
        bin_type = _select_bin_type_smallest(
            item_type, demand, capacities, purchase_costs, opening_costs
        )
        requires_purchase = open_counts[bin_type] >= purchased_counts[bin_type]
        return bin_type, requires_purchase

    if selection_method == BinTypeSelectionMethod.SLACK:
        return _select_bin_type_slack(
            item_type,
            demand,
            capacities,
            purchase_costs,
            opening_costs,
            purchased_counts,
            open_counts,
            weights,
            remaining_items,
        )

    raise ValueError(f"Unknown bin selection method: {selection_method!r}")


def _normalize_columns(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=0)
    denom = np.maximum(norms, eps)
    return mat / denom


def _select_open_bin_min_dot(
    demand_vec: np.ndarray,
    remaining: np.ndarray,
    bin_types: np.ndarray,
    opening_costs: np.ndarray,
    eps: float = 1e-12,
) -> int | None:
    if remaining.shape[1] == 0:
        return None

    demand_flat = np.asarray(demand_vec, dtype=float).reshape(-1)
    fits_mask = np.all(remaining >= demand_flat[:, None], axis=0)
    if not np.any(fits_mask):
        return None

    fit_indices = np.nonzero(fits_mask)[0]
    remaining_fit = remaining[:, fits_mask]
    demand_norm = demand_flat / max(float(np.linalg.norm(demand_flat)), eps)
    remaining_norm = _normalize_columns(remaining_fit, eps=eps)
    scores = remaining_norm.T @ demand_norm

    costs = opening_costs[bin_types[fit_indices]]
    order = np.lexsort((fit_indices, costs, scores))
    return int(fit_indices[int(order[0])])


def first_fit(
    C: np.ndarray,
    R: np.ndarray,
    purchase_costs: np.ndarray,
    opening_costs: np.ndarray,
    L: np.ndarray,
    selection_method: BinTypeSelectionMethod,
    weights: np.ndarray | None = None,
    opened_bins: np.ndarray | Sequence[int] | None = None,
    purchased_bins: np.ndarray | Sequence[int] | None = None,
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
    selection_method : BinTypeSelectionMethod, optional
        Determines how new bin types are selected when a new bin is needed.
        Defaults to marginal cost.
    weights : np.ndarray, optional
        Resource weight vector used by the SLACK selection method. Defaults to
        equal weights when omitted.
    opened_bins : array-like of shape (M,), optional
        Pre-opened bin counts per type. These bins are available before packing starts
        and incur only the opening cost if they were already purchased.
    purchased_bins : array-like of shape (M,), optional
        Bins already purchased but not necessarily opened. When provided and using the
        marginal cost selection rule, the algorithm prefers reusing these machines
        (opening cost only) before buying additional ones. This vector is updated in
        place when a new purchase occurs.

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
    if np.any(L < 0):
        raise ValueError("Job counts in L must be non-negative.")

    weight_vec = None
    if selection_method == BinTypeSelectionMethod.SLACK:
        weight_vec = (
            np.ones(K, dtype=float)
            if weights is None
            else np.asarray(weights, dtype=float).reshape(-1)
        )
        if weight_vec.shape[0] != K:
            raise ValueError(
                f"weights must have length {K}, got {weight_vec.shape[0]}."
            )

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
    B = np.zeros((J, bin_types.size), dtype=int)

    for j in range(J):
        count = int(L[j])
        if count == 0:
            continue

        demand_vec = R[:, j].astype(float, copy=False)
        demand = demand_vec.reshape(-1, 1)

        if remaining.shape[1]:
            positive = demand_vec > 0
            ratios = remaining[positive, :] / demand_vec[positive, None]
            max_add = np.floor(np.min(ratios, axis=0) + 1e-12).astype(int)
            max_add = np.maximum(max_add, 0)

            if max_add.size:
                prefix_before = np.cumsum(max_add) - max_add
                placed = np.clip(count - prefix_before, 0, max_add)
                placed = placed.astype(int)
                if np.any(placed):
                    remaining -= demand * placed[None, :]
                    B[j, :] += placed
                placed_total = int(np.sum(placed))
            else:
                placed_total = 0
        else:
            placed_total = 0

        remaining_items = count - placed_total
        if remaining_items <= 0:
            continue

        new_bin_types: list[int] = []
        new_bin_counts: list[int] = []

        while remaining_items > 0:
            bin_type, requires_purchase = _select_bin_type(
                selection_method,
                j,
                demand,
                C,
                purchase_costs,
                opening_costs,
                purchased_counts,
                open_counts,
                weights=weight_vec,
                remaining_items=remaining_items,
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

            capacity_vec = C[:, bin_type]
            positive = demand_vec > 0
            if np.any(positive):
                ratios = capacity_vec[positive] / demand_vec[positive]
                max_fit = int(np.floor(np.min(ratios) + 1e-12))
            else:
                max_fit = remaining_items
            if max_fit <= 0:
                max_fit = 1

            placed_here = min(remaining_items, max_fit)
            remaining_items -= placed_here
            new_bin_types.append(bin_type)
            new_bin_counts.append(placed_here)

        new_bin_types_arr = np.asarray(new_bin_types, dtype=int)
        new_counts_arr = np.asarray(new_bin_counts, dtype=int)
        new_remaining = C[:, new_bin_types_arr].copy()
        new_remaining -= demand * new_counts_arr[None, :]

        remaining = np.hstack((remaining, new_remaining))
        bin_types = np.concatenate((bin_types, new_bin_types_arr))

        new_B = np.zeros((J, new_counts_arr.size), dtype=int)
        new_B[j, :] = new_counts_arr
        B = np.hstack((B, new_B))

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


def first_fit_sorted(
    C: np.ndarray,
    R: np.ndarray,
    purchase_costs: np.ndarray,
    opening_costs: np.ndarray,
    L: np.ndarray,
    selection_method: BinTypeSelectionMethod,
    job_ordering_method: JobTypeOrderingMethod,
    weights: np.ndarray | None = None,
    opened_bins: np.ndarray | Sequence[int] | None = None,
    purchased_bins: np.ndarray | Sequence[int] | None = None,
) -> BinPackingResult:
    """
    Run first-fit after ordering job types according to ``job_ordering_method``.

    When using the weighted ordering rule, ``weights`` controls the ordering score and
    is also forwarded to the SLACK bin selection method.
    """

    R_sorted, L_sorted, sorted_indices, weight_vec = _sort_job_types(
        R, L, ordering_method=job_ordering_method, weights=weights
    )
    J = R_sorted.shape[1]

    result = first_fit(
        C,
        R_sorted,
        purchase_costs,
        opening_costs,
        L_sorted,
        opened_bins=opened_bins,
        purchased_bins=purchased_bins,
        selection_method=selection_method,
        weights=weight_vec,
    )

    if J > 0:
        inverse_perm = np.empty_like(sorted_indices)
        inverse_perm[sorted_indices] = np.arange(J)
        for bin_info in result.bins:
            bin_info.item_counts = bin_info.item_counts[inverse_perm]

    return result


def first_fit_decreasing(
    C: np.ndarray,
    R: np.ndarray,
    purchase_costs: np.ndarray,
    opening_costs: np.ndarray,
    L: np.ndarray,
    selection_method: BinTypeSelectionMethod,
    weights: np.ndarray | None = None,
    opened_bins: np.ndarray | Sequence[int] | None = None,
    purchased_bins: np.ndarray | Sequence[int] | None = None,
) -> BinPackingResult:
    """
    Run first-fit after sorting job types in non-increasing order of resource demand.

    Job requirement columns and their corresponding counts are permuted in tandem using
    a lexicographic ordering on the resource dimensions so that larger jobs are packed
    before smaller ones. The returned bin contents are mapped back to the original job
    ordering before being returned.

    Parameters mirror :func:`first_fit`; ``opened_bins``, ``purchased_bins``, and
    ``selection_method`` are forwarded directly without modification.
    """
    return first_fit_sorted(
        C,
        R,
        purchase_costs,
        opening_costs,
        L,
        selection_method=selection_method,
        job_ordering_method=JobTypeOrderingMethod.SORT_LEX,
        weights=weights,
        opened_bins=opened_bins,
        purchased_bins=purchased_bins,
    )


def ffd_weighted_sort(
    C: np.ndarray,
    R: np.ndarray,
    purchase_costs: np.ndarray,
    opening_costs: np.ndarray,
    L: np.ndarray,
    selection_method: BinTypeSelectionMethod,
    *,
    weights: np.ndarray | None = None,
    opened_bins: np.ndarray | Sequence[int] | None = None,
    purchased_bins: np.ndarray | Sequence[int] | None = None,
) -> BinPackingResult:
    """
    Run FFD after sorting job types by weighted demand.

    The packing logic mirrors :func:`first_fit_decreasing`, but the job types are
    ordered using the weighted-demand rule from the best-fit heuristic. The same
    weights are reused for the SLACK bin selection method.
    """
    return first_fit_sorted(
        C,
        R,
        purchase_costs,
        opening_costs,
        L,
        selection_method=selection_method,
        job_ordering_method=JobTypeOrderingMethod.SORT_BY_WEIGHT,
        weights=weights,
        opened_bins=opened_bins,
        purchased_bins=purchased_bins,
    )


def first_fit_largest(
    C: np.ndarray,
    R: np.ndarray,
    purchase_costs: np.ndarray,
    opening_costs: np.ndarray,
    L: np.ndarray,
    opened_bins: np.ndarray | Sequence[int] | None = None,
    purchased_bins: np.ndarray | Sequence[int] | None = None,
) -> BinPackingResult:
    """
    First-fit packing that opens the largest feasible bin type when needed.
    """
    return first_fit(
        C,
        R,
        purchase_costs,
        opening_costs,
        L,
        opened_bins=opened_bins,
        purchased_bins=purchased_bins,
        selection_method=BinTypeSelectionMethod.LARGEST,
    )


def first_fit_smallest(
    C: np.ndarray,
    R: np.ndarray,
    purchase_costs: np.ndarray,
    opening_costs: np.ndarray,
    L: np.ndarray,
    opened_bins: np.ndarray | Sequence[int] | None = None,
    purchased_bins: np.ndarray | Sequence[int] | None = None,
) -> BinPackingResult:
    """
    First-fit packing that opens the smallest feasible bin type when needed.
    """
    return first_fit(
        C,
        R,
        purchase_costs,
        opening_costs,
        L,
        opened_bins=opened_bins,
        purchased_bins=purchased_bins,
        selection_method=BinTypeSelectionMethod.SMALLEST,
    )


def first_fit_decreasing_largest(
    C: np.ndarray,
    R: np.ndarray,
    purchase_costs: np.ndarray,
    opening_costs: np.ndarray,
    L: np.ndarray,
    opened_bins: np.ndarray | Sequence[int] | None = None,
    purchased_bins: np.ndarray | Sequence[int] | None = None,
) -> BinPackingResult:
    """
    FFD variant that opens the largest feasible bin type for new placements.
    """

    R_sorted, L_sorted, sorted_indices = _sort_items_decreasing(R, L)
    J = R_sorted.shape[1]

    result = first_fit_decreasing(
        C,
        R_sorted,
        purchase_costs,
        opening_costs,
        L_sorted,
        opened_bins=opened_bins,
        purchased_bins=purchased_bins,
        selection_method=BinTypeSelectionMethod.LARGEST,
    )

    if J > 0:
        inverse_perm = np.empty_like(sorted_indices)
        inverse_perm[sorted_indices] = np.arange(J)
        for bin_info in result.bins:
            bin_info.item_counts = bin_info.item_counts[inverse_perm]

    return result


def _select_bin_type_largest(
    item_type: int,
    demand: np.ndarray,
    capacities: np.ndarray,
    purchase_costs: np.ndarray,
    opening_costs: np.ndarray,
) -> int:
    """
    Choose the feasible bin type with the largest total capacity.

    Ties are broken by lower operating cost, then lower purchase cost, then index.
    """

    fits_mask = np.all(capacities >= demand, axis=0)
    if not np.any(fits_mask):
        raise ValueError(
            f"Item type {item_type} does not fit in any available bin type."
        )

    sizes = np.asarray(capacities, dtype=float).sum(axis=0)
    purchase_vec = np.asarray(purchase_costs, dtype=float).reshape(-1)
    opening_vec = np.asarray(opening_costs, dtype=float).reshape(-1)

    best_type = None
    best_key = None
    for bin_type, fits in enumerate(fits_mask):
        if not fits:
            continue
        size = float(sizes[bin_type])
        key = (-size, opening_vec[bin_type], purchase_vec[bin_type], bin_type)
        if best_key is None or key < best_key:
            best_key = key
            best_type = bin_type

    if best_type is None:
        raise ValueError(
            f"Failed to choose a bin type for item type {item_type} using largest capacity rule."
        )

    return int(best_type)


def _select_bin_type_smallest(
    item_type: int,
    demand: np.ndarray,
    capacities: np.ndarray,
    purchase_costs: np.ndarray,
    opening_costs: np.ndarray,
) -> int:
    """
    Choose the feasible bin type with the smallest total capacity.

    Ties are broken by lower operating cost, then lower purchase cost, then index.
    """

    fits_mask = np.all(capacities >= demand, axis=0)
    if not np.any(fits_mask):
        raise ValueError(
            f"Item type {item_type} does not fit in any available bin type."
        )

    sizes = np.asarray(capacities, dtype=float).sum(axis=0)
    purchase_vec = np.asarray(purchase_costs, dtype=float).reshape(-1)
    opening_vec = np.asarray(opening_costs, dtype=float).reshape(-1)

    best_type = None
    best_key = None
    for bin_type, fits in enumerate(fits_mask):
        if not fits:
            continue
        size = float(sizes[bin_type])
        key = (size, opening_vec[bin_type], purchase_vec[bin_type], bin_type)
        if best_key is None or key < best_key:
            best_key = key
            best_type = bin_type

    if best_type is None:
        raise ValueError(
            f"Failed to choose a bin type for item type {item_type} using smallest capacity rule."
        )

    return int(best_type)
