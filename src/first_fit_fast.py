import numpy as np


def _max_items_per_bin(
    remaining_capacity: np.ndarray, demand: np.ndarray
) -> np.ndarray:
    """
    Compute how many items of a single type fit per bin given remaining capacity.

    Parameters
    ----------
    remaining_capacity : (K, N) array
        Current remaining capacity of each open bin.
    demand : (K,) array
        Resource demand of the item type.

    Returns
    -------
    (N,) array of ints with the per-bin item capacity.
    """

    if remaining_capacity.size == 0:
        return np.empty(0, dtype=np.int64)

    demand = demand.reshape(-1)
    if remaining_capacity.shape[0] != demand.size:
        raise ValueError(
            "remaining_capacity and demand must share the same number of dimensions"
        )

    max_int = np.iinfo(np.int64).max
    demand_safe = np.where(demand == 0, 1, demand)

    per_dim = remaining_capacity // demand_safe[:, None]
    # Dimensions with zero demand do not constrain placement.
    per_dim = np.where(demand[:, None] == 0, max_int, per_dim)

    return per_dim.min(axis=0)


def _allocate_first_fit(capacity_per_bin: np.ndarray, total_items: int) -> np.ndarray:
    """
    Distribute identical items across bins using a vectorized first-fit rule.

    Parameters
    ----------
    capacity_per_bin : (N,) array
        How many of the current item type each bin can still hold.
    total_items : int
        Number of items to place.

    Returns
    -------
    (N,) array with how many items are placed in each bin (in order).
    """

    if total_items <= 0 or capacity_per_bin.size == 0:
        return np.zeros_like(capacity_per_bin, dtype=np.int64)

    capacity_per_bin = capacity_per_bin.astype(np.int64, copy=False)

    prefix_exclusive = np.cumsum(capacity_per_bin) - capacity_per_bin
    remaining_before_bin = np.maximum(total_items - prefix_exclusive, 0)

    return np.minimum(capacity_per_bin, remaining_before_bin)


def first_fit_vectorized(
    C: np.ndarray,
    R: np.ndarray,
    load: np.ndarray,
):
    """
    Vectorized multidimensional first-fit for heterogeneous bins.

    The algorithm processes item types in order, packs identical items of a type in
    bulk using prefix-sums (no per-item Python loops), and opens the first bin type
    that can host the item when necessary.

    Parameters
    ----------
    C : (K, M) array
        Bin capacity matrix; each column is a bin type.
    R : (K, J) array
        Item requirement matrix; each column is an item type.
    l : (J,) array
        Number of items to pack per item type.

    Returns
    -------
    x : (M,) array
        Count of opened bins per bin type.
    B : (J, N) array
        Packed item counts per bin (columns correspond to bins).
    bin_types : (N,) array
        Bin type index for each opened bin.
    """

    C = np.asarray(C, dtype=np.int64)
    R = np.asarray(R, dtype=np.int64)
    load = np.asarray(load, dtype=np.int64).reshape(-1)

    if C.ndim != 2 or R.ndim != 2:
        raise ValueError("C and R must be 2D matrices.")

    K, M = C.shape
    K_items, J = R.shape
    if K_items != K:
        raise ValueError(
            f"Bin and item matrices must share the same number of rows; got {K} and {K_items}."
        )
    if load.size != J:
        raise ValueError(f"l must have length {J}, got {load.size}.")
    if np.any(C < 0) or np.any(R < 0) or np.any(load < 0):
        raise ValueError(
            "Capacities, requirements, and item counts must be non-negative."
        )

    remaining = np.empty((K, 0), dtype=np.int64)
    bin_types = np.empty(0, dtype=np.int64)
    B = np.zeros((J, 0), dtype=np.int64)

    for j in range(J):
        count = int(load[j])
        if count == 0:
            continue

        demand = R[:, j].astype(np.int64, copy=False)
        existing_capacity = _max_items_per_bin(remaining, demand)
        total_existing_capacity = int(existing_capacity.sum())

        feasible_types = np.all(demand[:, None] <= C, axis=0)
        if not np.any(feasible_types):
            raise ValueError(f"Item type {j} does not fit in any bin type.")

        chosen_type = int(np.argmax(feasible_types))  # First feasible bin type.
        capacity_new_bin = int(_max_items_per_bin(C[:, [chosen_type]], demand)[0])
        if capacity_new_bin <= 0:
            raise ValueError(
                f"Item type {j} cannot be placed in bin type {chosen_type}."
            )

        remaining_items = count - total_existing_capacity
        if remaining_items > 0:
            extra_bins = int(np.ceil(remaining_items / capacity_new_bin))
            if extra_bins > 0:
                new_caps = np.repeat(C[:, [chosen_type]], extra_bins, axis=1)
                remaining = np.concatenate([remaining, new_caps], axis=1)
                B = np.concatenate(
                    [B, np.zeros((J, extra_bins), dtype=np.int64)],
                    axis=1,
                )
                bin_types = np.concatenate(
                    [bin_types, np.full(extra_bins, chosen_type, dtype=np.int64)]
                )
                existing_capacity = np.concatenate(
                    [
                        existing_capacity,
                        np.full(extra_bins, capacity_new_bin, dtype=np.int64),
                    ]
                )

        assignments = _allocate_first_fit(existing_capacity, count)
        if assignments.size:
            B[j, : assignments.size] += assignments
            remaining -= demand[:, None] * assignments[None, :]

    x = np.bincount(bin_types, minlength=M).astype(np.int64)
    return x, B, bin_types
