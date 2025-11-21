from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class RandomInstance:
    """Container for a randomly generated, valid scheduling instance."""

    capacities: np.ndarray  # C shape: (K, M)
    requirements: np.ndarray  # R shape: (K, J)
    job_counts: np.ndarray  # L shape: (T, J)
    purchase_costs: np.ndarray  # shape: (M,)
    running_costs: np.ndarray  # shape: (M,)


def generate_random_instance(
    K: int,
    J: int,
    M: int,
    T: int,
    *,
    seed: int | None = None,
    capacity_range: Tuple[int, int] = (5, 18),
    demand_fraction: Tuple[float, float] = (0.2, 0.8),
    job_count_range: Tuple[int, int] = (0, 6),
    purchase_cost_range: Tuple[int, int] = (4, 15),
    running_cost_range: Tuple[int, int] = (1, 10),
) -> RandomInstance:
    """
    Generate a valid random problem instance for the scheduler.

    The output matrices follow the notation used in ``main.py`` and ``algorithms.py``:
    - capacities (C): (K, M) matrix of per-dimension machine capacities.
    - requirements (R): (K, J) matrix of per-dimension job demands.
    - job_counts (L): (T, J) non-negative integer matrix with job counts per slot.
    - purchase_costs and running_costs: length-M vectors of positive costs.

    Parameters let you control value ranges; defaults keep instances small but diverse.
    """

    if K <= 0 or J <= 0 or M <= 0 or T <= 0:
        raise ValueError("K, J, M, and T must be positive integers.")

    low_cap, high_cap = capacity_range
    if low_cap <= 0 or high_cap < low_cap:
        raise ValueError("capacity_range must be positive with low <= high.")

    low_df, high_df = demand_fraction
    if not (0 < low_df <= high_df):
        raise ValueError(
            "demand_fraction must contain positive bounds with low <= high."
        )

    low_jobs, high_jobs = job_count_range
    if low_jobs < 0 or high_jobs < low_jobs:
        raise ValueError("job_count_range must be non-negative with low <= high.")

    pc_low, pc_high = purchase_cost_range
    rc_low, rc_high = running_cost_range
    if pc_low <= 0 or rc_low <= 0 or pc_high < pc_low or rc_high < rc_low:
        raise ValueError("Cost ranges must be positive with low <= high.")

    rng = np.random.default_rng(seed)

    capacities = rng.integers(low_cap, high_cap + 1, size=(K, M), dtype=int)

    # Each job type is guaranteed to fit in at least one machine by sampling
    # demands as a fraction of a randomly chosen machine's capacity.
    requirements = np.zeros((K, J), dtype=int)
    for job_idx in range(J):
        anchor_machine = rng.integers(0, M)
        fractions = rng.uniform(low_df, high_df, size=(K,))
        req = np.maximum(1, np.floor(capacities[:, anchor_machine] * fractions)).astype(
            int
        )
        req = np.minimum(req, capacities[:, anchor_machine])
        requirements[:, job_idx] = req

    job_counts = rng.integers(low_jobs, high_jobs + 1, size=(T, J), dtype=int)
    if not np.any(job_counts):
        # Ensure at least one job exists so downstream code exercises packing logic.
        slot_idx = rng.integers(0, T)
        job_idx = rng.integers(0, J)
        job_counts[slot_idx, job_idx] = 1

    purchase_costs = rng.integers(pc_low, pc_high + 1, size=M, dtype=int)
    running_costs = rng.integers(rc_low, rc_high + 1, size=M, dtype=int)

    return RandomInstance(
        capacities=capacities.astype(float),
        requirements=requirements.astype(float),
        job_counts=job_counts,
        purchase_costs=purchase_costs,
        running_costs=running_costs,
    )
