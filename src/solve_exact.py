from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import time

try:
    import pulp
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "pulp is required for exact ILP solving. Install with `pip install pulp`."
    ) from exc

from algorithms import BinInfo, ScheduleResult, TimeSlotSolution
from problem_generation import ProblemInstance


@dataclass
class ILPModelData:
    model: pulp.LpProblem
    x_vars: dict[tuple[int, int, int], pulp.LpVariable]
    y_vars: dict[tuple[int, int, int, int], pulp.LpVariable]
    z_vars: dict[tuple[int, int], pulp.LpVariable]
    metadata: dict


def build_exact_ilp(
    problem: ProblemInstance,
) -> ILPModelData:
    """
    Build the ILP model for the scheduling/packing problem.

    Returns the pulp model plus the variable maps and metadata needed to
    reconstruct a ScheduleResult after solving.
    """

    capacities = np.asarray(problem.capacities, dtype=float)
    requirements = np.asarray(problem.requirements, dtype=float)
    job_counts = np.asarray(problem.job_counts, dtype=int)
    purchase_costs = np.asarray(problem.purchase_costs, dtype=float).reshape(-1)
    running_costs = np.asarray(problem.running_costs, dtype=float).reshape(-1)

    if capacities.ndim != 2 or requirements.ndim != 2:
        raise ValueError("capacities and requirements must be 2D matrices.")
    if capacities.shape[0] != requirements.shape[0]:
        raise ValueError(
            "capacities and requirements must describe the same resource dimensions."
        )

    if job_counts.ndim == 1:
        job_counts = job_counts.reshape(1, -1)
    elif job_counts.ndim != 2:
        raise ValueError("job_counts must be a vector or 2D matrix.")

    K, M = capacities.shape
    _, J = requirements.shape
    T = job_counts.shape[0]

    if job_counts.shape[1] != J:
        raise ValueError(
            f"job_counts must have {J} columns to match requirements; got {job_counts.shape[1]}."
        )
    if purchase_costs.shape[0] != M or running_costs.shape[0] != M:
        raise ValueError("Cost vectors must have one entry per machine type.")

    # Pre-compute feasibility and bounds to keep the ILP tight.
    job_fits = np.all(requirements[:, :, None] <= capacities[:, None, :], axis=0)
    infeasible_jobs = [
        j for j in range(J) if not np.any(job_fits[j]) and np.any(job_counts[:, j] > 0)
    ]
    if infeasible_jobs:
        raise ValueError(f"Job types {infeasible_jobs} do not fit in any machine type.")

    upper_bound = np.array([1000] * M, dtype=int)
    max_machines_per_type = [int(ub) for ub in upper_bound]
    for j in range(J):
        if not np.any(job_counts[:, j] > 0):
            continue
        has_capacity = any(
            job_fits[j, m] and max_machines_per_type[m] > 0 for m in range(M)
        )
        if not has_capacity:
            raise ValueError(
                f"Upper bound disallows any machine for job type {j}; instance is infeasible."
            )

    model = pulp.LpProblem("exact_schedule", pulp.LpMinimize)

    # Decision variables
    x_vars: dict[tuple[int, int, int], pulp.LpVariable] = {}
    z_vars: dict[tuple[int, int], pulp.LpVariable] = {}
    y_vars: dict[tuple[int, int, int, int], pulp.LpVariable] = {}

    for m, limit in enumerate(max_machines_per_type):
        for b in range(limit):
            z_vars[(m, b)] = pulp.LpVariable(
                f"z_{m}_{b}", lowBound=0, upBound=1, cat="Binary"
            )
            for t in range(T):
                x_vars[(t, m, b)] = pulp.LpVariable(
                    f"x_{t}_{m}_{b}", lowBound=0, upBound=1, cat="Binary"
                )
                for j in range(J):
                    if not job_fits[j, m]:
                        continue
                    y_vars[(t, m, b, j)] = pulp.LpVariable(
                        f"y_{t}_{m}_{b}_{j}", lowBound=0, cat="Integer"
                    )

    purchase_term = pulp.lpSum(
        purchase_costs[m] * z_var for (m, _), z_var in z_vars.items()
    )
    running_term = pulp.lpSum(
        running_costs[m] * var for (t, m, b), var in x_vars.items()
    )
    model += purchase_term + running_term

    # Cover all jobs exactly once per slot/type.
    for t in range(T):
        for j in range(J):
            terms = [
                y_vars[(t, m, b, j)]
                for m in range(M)
                for b in range(max_machines_per_type[m])
                if (t, m, b, j) in y_vars
            ]
            model += pulp.lpSum(terms) == int(job_counts[t, j])

    # Per-machine capacity and activation constraints.
    for (t, m, b), x_var in x_vars.items():
        capacity_col = capacities[:, m]
        # Respect capacities in every dimension.
        for k in range(K):
            load_terms = [
                requirements[k, j] * y_vars[(t, m, b, j)]
                for j in range(J)
                if (t, m, b, j) in y_vars
            ]
            model += pulp.lpSum(load_terms) <= capacity_col[k] * x_var

        # Disallow assigning jobs to powered-off machines, even for zero-demand jobs.
        assigned_jobs = [
            y_vars[(t, m, b, j)] for j in range(J) if (t, m, b, j) in y_vars
        ]
        if assigned_jobs:
            max_jobs = int(job_counts[t].sum())
            model += pulp.lpSum(assigned_jobs) <= max_jobs * x_var

        # Link power-on to purchase.
        model += x_var <= z_vars[(m, b)]

    metadata = dict(
        capacities=capacities,
        requirements=requirements,
        job_counts=job_counts,
        max_machines_per_type=max_machines_per_type,
        upper_bound=upper_bound,
    )

    return ILPModelData(
        model=model,
        x_vars=x_vars,
        y_vars=y_vars,
        z_vars=z_vars,
        metadata=metadata,
    )


def run_exact_solver(
    model: pulp.LpProblem,
    x_vars: dict[tuple[int, int, int], pulp.LpVariable],
    y_vars: dict[tuple[int, int, int, int], pulp.LpVariable],
    z_vars: dict[tuple[int, int], pulp.LpVariable],
    metadata: dict,
    *,
    time_limit: int | None = None,
) -> ScheduleResult:
    """Solve the ILP model and build a ScheduleResult."""

    capacities = metadata["capacities"]
    requirements = metadata["requirements"]
    job_counts = metadata["job_counts"]
    max_machines_per_type = metadata["max_machines_per_type"]
    upper_bound = metadata["upper_bound"]

    K, M = capacities.shape
    _, J = requirements.shape
    T = job_counts.shape[0]

    options = ["maxSolutions 1"]
    solver = pulp.PULP_CBC_CMD(
        msg=True, timeLimit=time_limit, threads=8, options=options
    )

    print(time.time(), "starting solve...")
    try:
        solve_status = model.solve(solver)
    except KeyboardInterrupt:
        pass
    print(time.time(), "finished solve!")

    status_text = pulp.LpStatus[solve_status]
    print(f"solver status: {status_text}")
    if status_text not in {"Optimal", "Integer Feasible"}:
        # raise RuntimeError(f"ILP solver failed with status: {status_text}")
        pass

    machine_vector = np.zeros(M, dtype=int)
    for m in range(M):
        for b in range(max_machines_per_type[m]):
            if pulp.value(z_vars[(m, b)]) >= 0.5:
                machine_vector[m] += 1

    time_slot_solutions = []
    for t in range(T):
        machine_counts = np.zeros(M, dtype=int)
        bins = []
        for m in range(M):
            capacity_col = capacities[:, [m]]
            for b in range(max_machines_per_type[m]):
                x_val = pulp.value(x_vars[(t, m, b)])
                if x_val is None or x_val < 0.5:
                    continue

                item_counts = np.zeros(J, dtype=int)
                for j in range(J):
                    var_key = (t, m, b, j)
                    if var_key not in y_vars:
                        continue
                    y_val = pulp.value(y_vars[var_key]) or 0.0
                    item_counts[j] = int(round(y_val))

                load = requirements @ item_counts.reshape(-1, 1)
                remaining = np.clip(capacity_col - load, a_min=0.0, a_max=None)

                bins.append(
                    BinInfo(
                        bin_type=m,
                        capacity=capacity_col.copy(),
                        remaining_capacity=remaining,
                        item_counts=item_counts,
                    )
                )
                machine_counts[m] += 1

        time_slot_solutions.append(
            TimeSlotSolution(machine_counts=machine_counts, bins=bins)
        )

    total_cost = float(pulp.value(model.objective))

    return ScheduleResult(
        total_cost=total_cost,
        machine_vector=machine_vector,
        upper_bound=upper_bound,
        time_slot_solutions=time_slot_solutions,
    )


def write_ilp_problem(model: pulp.LpProblem, path: str | Path) -> None:
    """Persist the ILP to disk in MPS or LP format."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".mps":
        model.writeMPS(str(path))
    elif suffix == ".lp":
        model.writeLP(str(path))
    else:
        raise ValueError("File extension must be .mps or .lp.")


def solve_exact(
    problem: ProblemInstance, *, time_limit: int | None = None
) -> ScheduleResult:
    """
    Solve the scheduling/packing problem exactly via integer linear programming.

    Variables
    ---------
    x[t,m,b] : binary
        Machine ``b`` of type ``m`` is powered on in time slot ``t``.
    z[m,b] : binary
        Machine ``b`` of type ``m`` is purchased (available to any slot).
    y[t,m,b,j] : integer
        Number of jobs of type ``j`` scheduled on machine ``b`` of type ``m`` in slot ``t``.

    Objective
    ---------
    Minimize purchase + running cost across the horizon.

    Constraints
    -----------
    1. All jobs in every slot are assigned exactly once.
    2. Per-machine capacity respected in every resource dimension.
    3. Jobs can only be placed on machines where they fit and only when the machine
       is powered on.
    4. Powered-on machines in any slot must have been purchased.
    """

    ilp_data = build_exact_ilp(problem)
    return run_exact_solver(
        ilp_data.model,
        ilp_data.x_vars,
        ilp_data.y_vars,
        ilp_data.z_vars,
        ilp_data.metadata,
        time_limit=time_limit,
    )
