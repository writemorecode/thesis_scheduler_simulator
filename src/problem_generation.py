from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ProblemInstance:
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
    alpha: np.ndarray | None = None,
    gamma: float = 0.05,
) -> ProblemInstance:
    """
    Generate a valid random problem instance for the scheduler.

    The output matrices follow the notation used in ``main.py`` and ``algorithms.py``:
    - capacities (C): (K, M) matrix of per-dimension machine capacities.
    - requirements (R): (K, J) matrix of per-dimension job demands.
    - job_counts (L): (T, J) non-negative integer matrix with job counts per slot.
    - purchase_costs (c_p): length-M vector computed as C.T @ alpha, where alpha sums to 1.
    - running_costs (c_r): length-M vector computed as gamma * purchase_costs.

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

    if gamma <= 0 or gamma >= 1:
        raise ValueError("gamma must be a percentage in the range (0, 1).")

    rng = np.random.default_rng(seed)

    capacities = rng.integers(low_cap, high_cap + 1, size=(K, M), dtype=int)

    # Dimension weights alpha reflect the relative scarcity/value per resource.
    if alpha is None:
        alpha_raw = rng.random(K)
    else:
        alpha_raw = np.asarray(alpha, dtype=float).reshape(-1)
        if alpha_raw.shape[0] != K:
            raise ValueError(f"alpha must have length {K}.")
        if np.any(alpha_raw < 0):
            raise ValueError("alpha entries must be non-negative.")

    alpha_sum = float(np.sum(alpha_raw))
    if alpha_sum <= 0:
        raise ValueError("alpha must have a positive sum before normalization.")

    alpha = alpha_raw / alpha_sum

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

    purchase_costs = capacities.T @ alpha
    running_costs = gamma * purchase_costs

    return ProblemInstance(
        capacities=capacities.astype(float),
        requirements=requirements.astype(float),
        job_counts=job_counts,
        purchase_costs=purchase_costs,
        running_costs=running_costs,
    )
