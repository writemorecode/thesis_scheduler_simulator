from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
# from typing import tuple

import numpy as np


@dataclass
class ProblemInstance:
    """Container for a randomly generated, valid scheduling instance."""

    capacities: np.ndarray  # C shape: (K, M)
    requirements: np.ndarray  # R shape: (K, J)
    job_counts: np.ndarray  # L shape: (J, T)
    purchase_costs: np.ndarray  # shape: (M,)
    running_costs: np.ndarray  # shape: (M,)
    resource_weights: np.ndarray  # shape: (K,)


def _validate_ratio(name: str, value: float) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value}.")


def _validate_range(name: str, bounds: tuple[float, float]) -> tuple[float, float]:
    low, high = bounds
    if low <= 0 or high < low:
        raise ValueError(
            f"{name} must have positive bounds with low <= high, got {bounds}."
        )
    return low, high


def _choose_primary_resources(
    count: int,
    num_resources: int,
    specialized_ratio: float,
    rng: np.random.Generator,
    prob: np.ndarray | None = None,
) -> np.ndarray:
    """Assign an optional primary resource to each entity (job or machine)."""
    num_specialized = int(round(count * specialized_ratio))
    indices = np.arange(count)
    specialized_indices = (
        rng.choice(indices, size=num_specialized, replace=False)
        if num_specialized
        else np.array([], dtype=int)
    )

    primary = np.full(count, -1, dtype=int)
    if num_specialized:
        probs = prob if prob is not None else None
        primary[specialized_indices] = rng.choice(
            num_resources, size=num_specialized, p=probs
        )
    return primary


def _generate_capacities_and_requirements(
    *,
    K: int,
    M: int,
    J: int,
    base_capacity: int,
    base_demand: int,
    mult_low: int,
    mult_high: int,
    cap_jitter_low: float,
    cap_jitter_high: float,
    dem_jitter_low: float,
    dem_jitter_high: float,
    job_primary: np.ndarray,
    machine_primary: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    capacities = np.full((K, M), base_capacity, dtype=float)
    requirements = np.full((K, J), base_demand, dtype=float)

    capacities *= rng.uniform(cap_jitter_low, cap_jitter_high, size=(K, M))
    requirements *= rng.uniform(dem_jitter_low, dem_jitter_high, size=(K, J))

    for m, primary in enumerate(machine_primary):
        if primary >= 0:
            mult = rng.integers(mult_low, mult_high + 1)
            capacities[primary, m] *= mult

    for j, primary in enumerate(job_primary):
        if primary >= 0:
            mult = rng.integers(mult_low, mult_high + 1)
            requirements[primary, j] *= mult

    capacities = np.maximum(1, np.rint(capacities)).astype(int)
    requirements = np.maximum(1, np.rint(requirements)).astype(int)

    for j in range(J):
        fits = np.all(capacities >= requirements[:, j][:, None], axis=0)
        if fits.any():
            continue

        primary = job_primary[j]
        if primary >= 0 and np.any(machine_primary == primary):
            candidates = np.where(machine_primary == primary)[0]
            target = int(rng.choice(candidates))
        else:
            target = int(np.argmax(np.sum(capacities, axis=0)))

        deficit = np.maximum(0, requirements[:, j] - capacities[:, target])
        capacities[:, target] += deficit

    return capacities, requirements


def _generate_job_counts(
    *,
    K: int,
    J: int,
    T: int,
    base_slot_load: int,
    load_jitter_low: float,
    load_jitter_high: float,
    slot_focus_ratio: float,
    focus_low: int,
    focus_high: int,
    job_primary: np.ndarray,
    slot_primary_correlation: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample job-count matrix L with slot primary resources that bias toward matching jobs."""
    job_hist = (
        np.bincount(job_primary[job_primary >= 0], minlength=K)
        if np.any(job_primary >= 0)
        else np.zeros(K)
    )
    if job_hist.sum() > 0:
        job_dist = job_hist / job_hist.sum()
    else:
        job_dist = np.full(K, 1.0 / K)

    uniform_dist = np.full(K, 1.0 / K)
    slot_pref = (
        slot_primary_correlation * job_dist
        + (1.0 - slot_primary_correlation) * uniform_dist
    )
    slot_pref = slot_pref / slot_pref.sum()

    slot_primary = _choose_primary_resources(
        count=T,
        num_resources=K,
        specialized_ratio=slot_focus_ratio,
        rng=rng,
        prob=slot_pref,
    )

    job_counts = np.zeros((T, J), dtype=int)
    for t in range(T):
        slot_total = max(
            1,
            int(round(base_slot_load * rng.uniform(load_jitter_low, load_jitter_high))),
        )
        weights = rng.uniform(0.5, 1.0, size=J)

        primary = slot_primary[t]
        if primary >= 0 and J > 0:
            matching_jobs = job_primary == primary
            if matching_jobs.any():
                focus_mult = rng.integers(focus_low, focus_high + 1)
                weights[matching_jobs] *= focus_mult

        weights_sum = float(weights.sum())
        probs = weights / weights_sum
        job_counts[t, :] = rng.multinomial(slot_total, probs)

    return job_counts


def _compute_costs(
    capacities: np.ndarray,
    resource_weights: np.ndarray | None,
    gamma: float | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = (
        resource_weights
        if resource_weights is not None
        else rng.random(capacities.shape[0])
    )
    purchase_costs = capacities.T @ weights
    running_costs = gamma * purchase_costs
    return purchase_costs, running_costs, weights


def generate_random_instance(
    K: int,
    M: int,
    J: int,
    T: int,
    *,
    base_capacity: int = 20,
    base_demand: int = 8,
    specialization_multiplier: tuple[int, int] = (2, 4),
    capacity_jitter: tuple[float, float] = (0.8, 1.3),
    demand_jitter: tuple[float, float] = (0.8, 1.2),
    specialized_job_ratio: float = 0.7,
    specialized_machine_ratio: float = 0.7,
    correlation: float = 0.7,
    base_slot_load: int = 12,
    slot_load_jitter: tuple[float, float] = (0.6, 1.4),
    slot_focus_ratio: float = 0.6,
    slot_specialization_correlation: float = 0.7,
    slot_focus_multiplier: tuple[int, int] = (5, 8),
    resource_weights: np.ndarray | None = None,
    gamma: float | None = 0.10,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> ProblemInstance:
    """
    Generate heterogeneous capacity (C) and demand (R) matrices with correlated specializations.

    Each job and machine type may have a primary resource that gets amplified by a multiplier,
    creating CPU-, memory-, or disk-leaning profiles. Machine specialization probabilities are
    biased toward job specializations via ``correlation`` so that CPU-heavy jobs are more likely
    to have CPU-optimized machines available. All outputs are integers and every job type is
    guaranteed to be packable into at least one machine type. The job-count matrix L (J×T)
    assigns time-slot primary resources (via ``slot_focus_ratio``) to emphasize job types whose
    primary resource matches the slot, mirroring the specialization approach used for C and R.

    Parameters
    ----------
    K, M, J, T: dimensionality of resources, machines, jobs, and time slots.
    base_capacity: mean capacity per resource per machine before jitter and specialization.
    base_demand: mean requirement per resource per job before jitter and specialization.
    specialization_multiplier: low/high integer multiplier applied to a primary resource when
        a job or machine is specialized.
    capacity_jitter: multiplicative jitter range applied to base capacities.
    demand_jitter: multiplicative jitter range applied to base demands.
    specialized_job_ratio: fraction of job types that get a primary resource.
    specialized_machine_ratio: fraction of machine types that get a primary resource.
    correlation: weight pulling machine specialization probabilities toward the job histogram
        (1.0 follows jobs exactly, 0.0 is uniform over resources).
    base_slot_load: mean total job count per time slot before jitter.
    slot_load_jitter: multiplicative jitter range applied to per-slot load.
    slot_focus_ratio: fraction of time slots assigned a primary resource (specialized slots).
    slot_specialization_correlation: weight pulling slot specialization probabilities toward the
        job specialization histogram (1.0 follows jobs exactly, 0.0 is uniform over resources).
    slot_focus_multiplier: low/high integer multiplier to boost job types matching a slot's primary
        resource.
    resource_weights: optional resource cost weight vector (K,). Randomly sampled if None.
    gamma: running-cost factor applied to purchase costs.
    rng/seed: random generator or seed used for reproducibility.
    """

    if K <= 0 or M <= 0 or J <= 0 or T <= 0:
        raise ValueError("K, M, J, and T must be positive integers.")

    _validate_ratio("specialized_job_ratio", specialized_job_ratio)
    _validate_ratio("specialized_machine_ratio", specialized_machine_ratio)
    _validate_ratio("correlation", correlation)
    _validate_ratio("slot_focus_ratio", slot_focus_ratio)
    _validate_ratio("slot_specialization_correlation", slot_specialization_correlation)

    if base_capacity <= 0 or base_demand <= 0:
        raise ValueError("base_capacity and base_demand must be positive integers.")

    mult_low, mult_high = specialization_multiplier
    if mult_low < 1 or mult_high < mult_low:
        raise ValueError(
            "specialization_multiplier must have bounds >= 1 with low <= high."
        )

    cap_jitter_low, cap_jitter_high = _validate_range(
        "capacity_jitter", capacity_jitter
    )
    dem_jitter_low, dem_jitter_high = _validate_range("demand_jitter", demand_jitter)
    load_jitter_low, load_jitter_high = _validate_range(
        "slot_load_jitter", slot_load_jitter
    )

    focus_low, focus_high = slot_focus_multiplier
    if focus_low < 1 or focus_high < focus_low:
        raise ValueError(
            "slot_focus_multiplier must have bounds >= 1 with low <= high."
        )

    if base_slot_load <= 0:
        raise ValueError("base_slot_load must be positive.")

    rng = rng or np.random.default_rng(seed)

    # Specialize jobs and machines; machine specialization is biased toward job histogram.
    job_primary = _choose_primary_resources(
        count=J,
        num_resources=K,
        specialized_ratio=specialized_job_ratio,
        rng=rng,
    )

    job_hist = (
        np.bincount(job_primary[job_primary >= 0], minlength=K)
        if np.any(job_primary >= 0)
        else np.zeros(K)
    )
    if job_hist.sum() > 0:
        job_dist = job_hist / job_hist.sum()
    else:
        job_dist = np.full(K, 1.0 / K)

    uniform_dist = np.full(K, 1.0 / K)
    machine_pref = correlation * job_dist + (1.0 - correlation) * uniform_dist
    machine_pref = machine_pref / machine_pref.sum()

    machine_primary = _choose_primary_resources(
        count=M,
        num_resources=K,
        specialized_ratio=specialized_machine_ratio,
        rng=rng,
        prob=machine_pref,
    )

    capacities, requirements = _generate_capacities_and_requirements(
        K=K,
        M=M,
        J=J,
        base_capacity=base_capacity,
        base_demand=base_demand,
        mult_low=mult_low,
        mult_high=mult_high,
        cap_jitter_low=cap_jitter_low,
        cap_jitter_high=cap_jitter_high,
        dem_jitter_low=dem_jitter_low,
        dem_jitter_high=dem_jitter_high,
        job_primary=job_primary,
        machine_primary=machine_primary,
        rng=rng,
    )

    job_counts = _generate_job_counts(
        K=K,
        J=J,
        T=T,
        base_slot_load=base_slot_load,
        load_jitter_low=load_jitter_low,
        load_jitter_high=load_jitter_high,
        slot_focus_ratio=slot_focus_ratio,
        focus_low=focus_low,
        focus_high=focus_high,
        job_primary=job_primary,
        slot_primary_correlation=slot_specialization_correlation,
        rng=rng,
    )

    purchase_costs, running_costs, resource_weights_used = _compute_costs(
        capacities=capacities,
        resource_weights=resource_weights,
        gamma=gamma,
        rng=rng,
    )

    return ProblemInstance(
        capacities=capacities,
        requirements=requirements,
        job_counts=job_counts,
        purchase_costs=purchase_costs,
        running_costs=running_costs,
        resource_weights=resource_weights_used,
    )


def generate_dataset(
    num_instances: int,
    *,
    K_range: tuple[int, int],
    J_range: tuple[int, int],
    M_range: tuple[int, int],
    T_range: tuple[int, int],
    dataset_dir: str | Path = "dataset",
    rng: np.random.Generator,
) -> list[dict[str, int | str]]:
    """
    Generate a dataset of random problem instances and store them on disk.

    Each instance samples K, J, M, and T uniformly within the provided ranges,
    then forwards those values to ``generate_random_instance``.

    Returns a list of metadata rows that also get written to ``dataset/dataset.csv``.
    """

    if num_instances <= 0:
        raise ValueError("num_instances must be a positive integer.")

    def _sample_dimension(bounds: tuple[int, int], name: str) -> int:
        low, high = bounds
        if low <= 0 or high < low:
            raise ValueError(f"{name} must be positive with low <= high.")
        return int(rng.integers(low, high + 1))

    dataset_path = Path(dataset_dir)
    dataset_path.mkdir(parents=True, exist_ok=True)

    metadata: list[dict[str, int | str]] = []

    for idx in range(num_instances):
        K = _sample_dimension(K_range, "K_range")
        J = _sample_dimension(J_range, "J_range")
        M = _sample_dimension(M_range, "M_range")
        T = _sample_dimension(T_range, "T_range")

        instance = generate_random_instance(
            K=K,
            J=J,
            M=M,
            T=T,
            rng=rng,
        )

        filename = f"instance_{idx:04d}.npz"
        file_path = dataset_path / filename

        np.savez(
            file_path,
            capacities=instance.capacities,
            requirements=instance.requirements,
            job_counts=instance.job_counts,
            purchase_costs=instance.purchase_costs,
            running_costs=instance.running_costs,
            resource_weights=instance.resource_weights,
            K=K,
            J=J,
            M=M,
            T=T,
        )

        metadata.append(
            {
                "filename": filename,
                "K": K,
                "J": J,
                "M": M,
                "T": T,
            }
        )

    csv_path = dataset_path / "dataset.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["filename", "K", "J", "M", "T"])
        writer.writeheader()
        writer.writerows(metadata)

    return metadata
