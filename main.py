from __future__ import annotations

import numpy as np

from algorithms import (
    schedule_jobs,
)


def main():
    # Five resource dimensions (e.g., CPU/GPU/memory/storage/network) and four machine types.
    C = np.array(
        [
            [24, 32, 48, 60],
            [8, 12, 16, 20],
            [64, 96, 128, 160],
            [240, 320, 400, 520],
            [100, 140, 180, 220],
        ],
        dtype=float,
    )
    # Per-job demand for the five resources across four job types.
    R = np.array(
        [
            [16, 12, 8, 22],
            [6, 10, 4, 5],
            [48, 56, 32, 40],
            [180, 220, 360, 200],
            [70, 90, 60, 80],
        ],
        dtype=float,
    )

    purchase_costs = np.array([18.0, 24.0, 32.0, 45.0])
    running_costs = np.array([4.5, 6.0, 7.5, 9.0])

    # Six time slots, each listing counts for the four job types.
    L = np.array(
        [
            [3, 1, 1, 2],
            [4, 2, 2, 3],
            [5, 3, 1, 4],
            [2, 4, 0, 1],
            [3, 2, 2, 2],
            [4, 3, 1, 3],
        ]
    )

    schedule = schedule_jobs(C, R, L, purchase_costs, running_costs)
    print("Schedule cost:", schedule.total_cost)
    print("Machine vector:", schedule.machine_vector)


if __name__ == "__main__":
    main()
