from __future__ import annotations
from collections import Counter

from dataclasses import dataclass
from typing import List, Sequence, Tuple

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


@dataclass
class _RepackBinState:
    """Internal mutable view of a bin used by the re-packing heuristic."""

    index: int
    bin_type: int
    capacity: np.ndarray
    load: np.ndarray
    items: List[int]
    item_counts: np.ndarray
    running_cost: float
    utilization: float
    had_items_initially: bool
    emptied: bool = False

    def available_capacity(self) -> np.ndarray:
        """Return the per-dimension free capacity of the bin."""

        return self.capacity - self.load


def _bin_utilization(load: np.ndarray, capacity: np.ndarray) -> float:
    """Compute the max load-to-capacity ratio for a bin (see method.typ)."""

    utilization = -np.inf
    for cap, used in zip(capacity, load):
        if cap <= 0:
            continue
        utilization = max(utilization, float(used / cap))
    return utilization


def _make_item_sort_keys(job_vectors: Sequence[np.ndarray]) -> List[Tuple[float, ...]]:
    """Pre-compute lexicographic sort keys for each job vector (largest first)."""

    keys: List[Tuple[float, ...]] = []
    for vec in job_vectors:
        flattened = vec.reshape(-1)
        keys.append(tuple((-flattened).tolist()))
    return keys


def _pop_item(
    state: _RepackBinState, idx: int, job_vectors: Sequence[np.ndarray]
) -> int:
    """Remove an item from ``state`` and update its load bookkeeping."""

    job_type = state.items.pop(idx)
    state.item_counts[job_type] -= 1
    state.load -= job_vectors[job_type]
    return job_type


def _push_item(
    state: _RepackBinState,
    job_type: int,
    job_vectors: Sequence[np.ndarray],
    sort_keys: Sequence[Tuple[float, ...]],
) -> None:
    """Insert an item into ``state`` keeping its items sorted in non-increasing order."""

    state.items.append(job_type)
    state.item_counts[job_type] += 1
    state.load += job_vectors[job_type]
    state.items.sort(key=lambda jt: sort_keys[jt])


def _move_items_between_bins(
    source: _RepackBinState,
    target: _RepackBinState,
    job_vectors: Sequence[np.ndarray],
    sort_keys: Sequence[Tuple[float, ...]],
    atol: float,
) -> bool:
    """Attempt to move items from ``source`` to ``target``. Returns True if anything moved."""

    moved = False
    idx = 0
    while idx < len(source.items):
        job_type = source.items[idx]
        demand = job_vectors[job_type]
        if np.all(target.available_capacity() + atol >= demand):
            moved = True
            _pop_item(source, idx, job_vectors)
            _push_item(target, job_type, job_vectors, sort_keys)
            continue
        idx += 1
    return moved


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
            remaining_capacity=capacity,
            item_counts=np.zeros(J, dtype=int),
        )
        bins.append(bin_info)
        return bin_info

    if opened_bins_vec is not None:
        total_cost += float(np.dot(opened_bins_vec, per_bin_costs))
        for bin_type, count in enumerate(opened_bins_vec):
            for _ in range(int(count)):
                _create_bin(bin_type)

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
                    total_cost += float(per_bin_costs[bin_type])
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
    opened_bins: np.ndarray | Sequence[int] | None = None,
) -> BinPackingResult:
    """
    Run first-fit after sorting item requirements per dimension in non-increasing order.

    The sorting uses ``np.sort`` (ascending) followed by ``np.fliplr`` to flip the
    columns, yielding a decreasing order for each row.

    Parameters mirror :func:`first_fit`; ``opened_bins`` is forwarded directly without
    modification.
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

    return first_fit(
        C, R_sorted, purchase_costs, opening_costs, L_array, opened_bins=opened_bins
    )


def repack_jobs(
    bins: Sequence[BinInfo] | BinPackingResult,
    C: np.ndarray,
    R: np.ndarray,
    running_costs: np.ndarray,
    running_counts: Sequence[int] | None = None,
    *,
    atol: float = 1e-9,
) -> Tuple[np.ndarray, List[BinInfo]]:
    """Apply the job re-packing heuristic described in ``method.typ``.

    Parameters
    ----------
    bins : Sequence[BinInfo] | BinPackingResult
        Current packing configuration. If a :class:`BinPackingResult` is supplied, its
        ``bins`` attribute is used.
    C : np.ndarray
        Capacity matrix of shape (K, M). Column ``m`` is the capacity vector for
        machine/bin type ``m``.
    R : np.ndarray
        Requirement matrix of shape (K, J). Column ``j`` is the demand vector for job
        type ``j``.
    running_costs : np.ndarray
        Length M vector with per-time-slot running cost per machine type. This is used
        as the tie-breaker when ordering bins with identical utilization.
    running_counts : Sequence[int], optional
        Vector ``z`` with the number of live instances per machine type. When omitted,
        it is reconstructed directly from ``bins``.
    atol : float, optional
        Numerical tolerance used when comparing remaining capacities.

    Returns
    -------
    tuple[np.ndarray, list[BinInfo]]
        The updated ``z`` vector and new bin descriptions after re-packing.
    """

    capacity_matrix = np.asarray(C, dtype=float)
    requirement_matrix = np.asarray(R, dtype=float)
    if capacity_matrix.ndim != 2 or requirement_matrix.ndim != 2:
        raise ValueError("C and R must be 2D matrices.")

    K, M = capacity_matrix.shape
    K_req, J = requirement_matrix.shape
    if K != K_req:
        raise ValueError(
            f"C and R must represent the same number of resources (got {K} and {K_req})."
        )

    running_costs_vec = _prepare_vector(running_costs, M, "running_costs")

    # Cache per-job demand vectors (one column per job type) so we can reference
    # them quickly when we move items between bins.
    job_vectors = [requirement_matrix[:, j].reshape(-1) for j in range(J)]
    # Sorting keys encode the "non-increasing by dimension" ordering mandated by
    # the algorithmic description.
    item_sort_keys = _make_item_sort_keys(job_vectors)

    if isinstance(bins, BinPackingResult):
        bin_sequence = bins.bins
    else:
        bin_sequence = list(bins)

    # Convert immutable ``BinInfo`` objects into mutable state that tracks load,
    # utilization, and the explicit list of items each bin contains.
    bin_states: List[_RepackBinState] = []
    for idx, bin_info in enumerate(bin_sequence):
        bin_type = int(bin_info.bin_type)
        if bin_type < 0 or bin_type >= M:
            raise ValueError(
                f"Bin {idx} references invalid type {bin_type}; must be within [0, {M})."
            )

        capacity_vec = capacity_matrix[:, bin_type].reshape(-1)
        remaining_vec = np.asarray(bin_info.remaining_capacity, dtype=float).reshape(-1)
        if remaining_vec.shape[0] != K:
            raise ValueError(
                f"Bin {idx} remaining capacity dimension mismatch. Expected {K}, got {remaining_vec.shape[0]}."
            )
        load_vec = np.clip(capacity_vec - remaining_vec, 0.0, None)

        item_counts = np.asarray(bin_info.item_counts, dtype=int).reshape(-1)
        if item_counts.shape[0] != J:
            raise ValueError(
                f"Bin {idx} item_counts length mismatch. Expected {J}, got {item_counts.shape[0]}."
            )

        items: List[int] = []
        for job_type, count in enumerate(item_counts):
            if count <= 0:
                continue
            items.extend([job_type] * int(count))
        items.sort(key=lambda jt: item_sort_keys[jt])

        bin_states.append(
            _RepackBinState(
                index=idx,
                bin_type=bin_type,
                capacity=capacity_vec.copy(),
                load=load_vec,
                items=items,
                item_counts=item_counts.copy(),
                running_cost=float(running_costs_vec[bin_type]),
                utilization=_bin_utilization(load_vec, capacity_vec),
                had_items_initially=len(items) > 0,
            )
        )

    if not bin_states:
        running_counts_vec = (
            np.zeros(M, dtype=int)
            if running_counts is None
            else np.asarray(running_counts, dtype=int).reshape(-1)
        )
        return running_counts_vec, []

    # Step 1: sort bins by ascending utilization, breaking ties by descending running
    # cost so that expensive bins are considered for emptying first.
    bin_states.sort(key=lambda state: (state.utilization, -state.running_cost))

    # Build or validate the ``z`` vector that tracks how many machines of each type
    # are currently powered on.
    if running_counts is None:
        running_counts_vec = np.zeros(M, dtype=int)
        for state in bin_states:
            if state.had_items_initially:
                running_counts_vec[state.bin_type] += 1
    else:
        running_counts_vec = np.asarray(running_counts, dtype=int).reshape(-1)
        if running_counts_vec.shape[0] != M:
            raise ValueError(
                f"running_counts must have length {M}; got {running_counts_vec.shape[0]}."
            )

    i = 0
    j = len(bin_states) - 1

    # Move ``i`` to the first non-empty bin; empty bins to the left cannot donate
    # work and are therefore skipped.
    while i < len(bin_states) and not bin_states[i].items:
        i += 1

    while i < j:
        source = bin_states[i]
        target = bin_states[j]

        # Try to move the largest items from the low-utilization bin ``i`` into the
        # high-utilization bin ``j`` without violating any capacity constraints.
        _move_items_between_bins(source, target, job_vectors, item_sort_keys, atol)

        if not source.items:
            if source.had_items_initially and not source.emptied:
                running_counts_vec[source.bin_type] = max(
                    0, running_counts_vec[source.bin_type] - 1
                )
                source.emptied = True

            i += 1
            while i < len(bin_states) and not bin_states[i].items:
                i += 1
        else:
            # ``j`` is shifted leftwards to try the next-highest-utilization bin.
            j -= 1

    # Convert the mutable state back into ``BinInfo`` objects using each bin's
    # original ordering so callers see a familiar structure.
    updated_by_state: List[BinInfo] = []
    for state in bin_states:
        remaining = np.clip(state.available_capacity(), 0.0, None).reshape(-1, 1)
        updated_by_state.append(
            BinInfo(
                bin_type=state.bin_type,
                remaining_capacity=remaining,
                item_counts=state.item_counts.copy(),
            )
        )

    ordered_bins: List[BinInfo] = [None] * len(bin_states)
    for state, updated in zip(bin_states, updated_by_state):
        ordered_bins[state.index] = updated

    return running_counts_vec, ordered_bins


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
