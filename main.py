from __future__ import annotations

import numpy as np

from algorithms import schedule_jobs


def debugging():
    C = np.array(
        [
            [12, 15, 18, 10],
            [10, 14, 18, 12],
        ],
        dtype=float,
    )
    R = np.array(
        [
            [4, 6, 5, 9],
            [3, 5, 7, 6],
        ],
        dtype=float,
    )

    purchase_costs = np.array([6.0, 9.0, 12.0, 15.0])
    running_costs = np.array([1.5, 2.0, 2.5, 3.0])

    # Six time slots with counts per job type.
    L = np.array(
        [
            [4, 2, 1, 0],
        ]
    )
    schedule = schedule_jobs(C, R, L, purchase_costs, running_costs)
    print(schedule)


if __name__ == "__main__":
    debugging()
