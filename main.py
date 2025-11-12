from __future__ import annotations

import numpy as np

from algorithms import first_fit, first_fit_decreasing, machines_upper_bound


def example():
    C = np.array([[10, 15], [8, 12]], dtype=float)  # 2 dimensions, 2 bin types
    R = np.array([[4, 6], [3, 4]], dtype=float)  # 2 item types

    print(f"Items:\n{C}")
    print(f"Bins:\n{R}\n")
    print("-" * 60)

    purchase_costs = np.array([5.0, 7.0])
    opening_costs = np.array([1.0, 1.5])
    L = np.array([2, 3])

    ff_result = first_fit(C, R, purchase_costs, opening_costs, L)
    print("First-fit:")
    print(ff_result)

    print("-" * 60)

    ffd_result = first_fit_decreasing(C, R, purchase_costs, opening_costs, L)
    print("First-fit decreasing:")
    print(ffd_result)

    improvement = ffd_result.total_cost / ff_result.total_cost
    print(f"Improvement factor: {improvement}")

    upper_bound = machines_upper_bound(C, R, L)
    print("-" * 60)
    print(f"Upper bound on machines per type: {upper_bound}")


if __name__ == "__main__":
    example()
