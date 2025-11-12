from __future__ import annotations

import numpy as np

from algorithms import first_fit, first_fit_decreasing, machines_upper_bound


def example():
    C = np.array([[10, 15], [8, 12]], dtype=float)
    R = np.array([[4, 6], [3, 4]], dtype=float)

    purchase_costs = np.array([5.0, 7.0])
    opening_costs = np.array([1.0, 1.5])
    L = np.array([2, 3])

    upper_bound = machines_upper_bound(C, R, L)
    print(f"Upper bound on machines per type: {upper_bound}")
    print("-" * 60)

    ff_result = first_fit(C, R, purchase_costs, opening_costs, L)
    print("First-fit")
    print(ff_result)

    print("-" * 60)

    ff_preopened = first_fit(
        C, R, purchase_costs, opening_costs, L, opened_bins=upper_bound
    )
    print("First-fit with pre-opened bins (upper bound)")
    print(ff_preopened)

    print("-" * 60)

    ffd_result = first_fit_decreasing(C, R, purchase_costs, opening_costs, L)
    print("First-fit decreasing:", ffd_result.total_cost)
    print(ffd_result)


if __name__ == "__main__":
    example()
