from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ProblemInstance:
    """Container for a scheduling instance consumed by simulator algorithms."""

    capacities: np.ndarray  # C shape: (K, M)
    requirements: np.ndarray  # R shape: (K, J)
    job_counts: np.ndarray  # L shape: (J, T)
    purchase_costs: np.ndarray  # shape: (M,)
    running_costs: np.ndarray  # shape: (M,)
    resource_weights: np.ndarray  # shape: (K,)
