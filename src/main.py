from __future__ import annotations

import numpy as np

from algorithms import schedule_jobs, dump_schedule_to_csv
from visualization import visualize_schedule

C = np.array(
    [
        [3, 9, 5, 10],
        [4, 10, 7, 4],
    ],
    dtype=float,
)
R = np.array(
    [
        [4, 10, 9, 3],
        [10, 2, 9, 4],
    ],
    dtype=float,
)

purchase_costs = np.array([7, 6, 12, 10])
running_costs = np.array([1, 8, 9, 11])

running_costs_sorted_indicies = np.argsort(running_costs)
C_sorted = C[:, running_costs_sorted_indicies]


# TODO: put back in 'src/packing.py' for 'first_fit' and etc
def pick_bin_type(item_type: int, C: np.ndarray, R: np.ndarray) -> int:
    demand_vec = R[:, [item_type]].reshape(-1, 1)
    feasible_bin_types = np.all(C_sorted >= demand_vec, axis=0)
    if not np.any(feasible_bin_types):
        raise ValueError(
            f"Item type {item_type} does not fit in any available bin type."
        )
    best_bin_idx = np.argmax(feasible_bin_types)
    bin_vector = C_sorted[:, best_bin_idx].reshape(-1, 1)
    assert np.all(bin_vector >= demand_vec), "invalid bin type"
    return running_costs_sorted_indicies[best_bin_idx]


def main():
    # Six time slots with counts per job type.
    L = np.array(
        [
            [3, 1, 0, 2],
            [4, 2, 1, 0],
            [2, 3, 2, 1],
            [1, 0, 3, 2],
            [5, 2, 2, 1],
            [0, 4, 1, 3],
        ]
    )

    for item_type in range(R.shape[1]):
        bin_type = pick_bin_type(item_type, C, R)
        # print(f"Item type: {item_type} Bin type: {bin_type}")
        print("item", R[:, item_type], "bin", C[:, bin_type])

    # schedule = schedule_jobs(C, R, L, purchase_costs, running_costs)
    # print(schedule.total_cost)

    # print("Avg. unused bin capacity vector:", schedule.average_remaining_capacity())
    # _ = visualize_schedule(schedule, C, R, "out.png")
    # dump_schedule_to_csv(schedule, "data.csv")


if __name__ == "__main__":
    main()
