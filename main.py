from __future__ import annotations

import numpy as np

from algorithms import (
    BinInfo,
    first_fit,
    first_fit_decreasing,
    machines_upper_bound,
    repack_jobs,
)


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


def repacking_example():
    print("\nRe-packing example")
    print("=" * 60)

    C = np.array([[10.0, 14.0]], dtype=float)  # One resource, two machine types
    R = np.array([[6.0, 4.0, 3.0]], dtype=float)  # Three job types
    running_costs = np.array([1.0, 2.0])  # Type 1 is more expensive to keep running

    def build_bin(bin_type: int, job_sequence: list[int]) -> BinInfo:
        remaining = C[:, [bin_type]].copy()
        counts = np.zeros(R.shape[1], dtype=int)
        for job_type in job_sequence:
            remaining -= R[:, [job_type]]
            counts[job_type] += 1
        return BinInfo(
            bin_type=bin_type,
            remaining_capacity=remaining,
            item_counts=counts,
        )

    # Two low-utilization type-1 machines can donate their work to the busier type-0 bins.
    bins = [
        build_bin(1, [2]),  # 3 units of work on a 14-unit machine
        build_bin(1, [1]),  # 4 units of work on a 14-unit machine
        build_bin(0, [0]),  # 6 units of work on a 10-unit machine
        build_bin(0, [1, 2]),  # 7 units of work on a 10-unit machine
    ]

    initial_counts = np.zeros(C.shape[1], dtype=int)
    for bin_info in bins:
        if np.any(bin_info.item_counts):
            initial_counts[bin_info.bin_type] += 1

    print("Capacity matrix C (rows=resources, cols=machine types):")
    print(C)
    print("Job requirements R (rows=resources, cols=job types):")
    print(R)
    print(f"Initial running counts per machine type: {initial_counts}")
    print("\nBins before re-packing:")
    for idx, bin_info in enumerate(bins):
        print(f"Bin {idx}:")
        print(bin_info)
        print()

    updated_counts, repacked_bins = repack_jobs(
        bins,
        C=C,
        R=R,
        running_costs=running_costs,
    )

    print("-" * 60)
    print(f"Running counts after re-packing: {updated_counts}")
    print("Bins after re-packing:")
    for idx, bin_info in enumerate(repacked_bins):
        print(f"Bin {idx}:")
        print(bin_info)
        print()


if __name__ == "__main__":
    repacking_example()
    #example()
