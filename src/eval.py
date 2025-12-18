from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from algorithms import ScheduleResult, ffd_schedule
from problem_generation import ProblemInstance
from ruin_recreate import ruin_recreate_schedule
from simple_scheduler import simple_scheduler


@dataclass
class InstanceResult:
    filename: str
    total_cost: float
    machine_vector: np.ndarray
    runtime_sec: float

    @property
    def total_machines(self) -> int:
        return int(np.sum(self.machine_vector))


def _load_problem(npz_path: Path) -> ProblemInstance:
    """Load a problem instance stored by ``generate_dataset``."""

    with np.load(npz_path) as data:
        required_keys = [
            "capacities",
            "requirements",
            "job_counts",
            "purchase_costs",
            "running_costs",
            "resource_weights",
        ]
        missing = [key for key in required_keys if key not in data]
        if missing:
            raise ValueError(f"{npz_path} is missing required keys: {missing}")

        return ProblemInstance(
            capacities=data["capacities"],
            requirements=data["requirements"],
            job_counts=data["job_counts"],
            purchase_costs=data["purchase_costs"],
            running_costs=data["running_costs"],
            resource_weights=data["resource_weights"],
        )


def _npz_dimensions(npz_path: Path) -> dict[str, int]:
    with np.load(npz_path) as data:
        capacities = np.asarray(data["capacities"])
        requirements = np.asarray(data["requirements"])
        job_counts = np.asarray(data["job_counts"])

        K, M = capacities.shape
        _, J = requirements.shape
        T = int(np.atleast_2d(job_counts).shape[0])

        return {"K": int(K), "J": int(J), "M": int(M), "T": T}


def _dataset_entries(dataset_dir: Path) -> Iterable[tuple[Path, dict[str, int]]]:
    csv_path = dataset_dir / "dataset.csv"
    if csv_path.exists():
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                filename = row["filename"]
                dims = {key: int(row[key]) for key in ("K", "J", "M", "T")}
                yield dataset_dir / filename, dims
        return

    for npz_path in sorted(dataset_dir.glob("*.npz")):
        yield npz_path, _npz_dimensions(npz_path)


def _build_scheduler(
    name: str, *, iterations: int, rng: np.random.Generator
) -> Callable[[ProblemInstance], ScheduleResult]:
    normalized = name.lower().replace("-", "_")

    if normalized in {"ruin_recreate", "rr"}:
        return lambda problem: ruin_recreate_schedule(
            problem, max_iterations=iterations, rng=rng
        )

    if normalized in {"ffd", "first_fit_decreasing"}:

        def _ffd(problem: ProblemInstance) -> ScheduleResult:
            return ffd_schedule(
                C=problem.capacities,
                R=problem.requirements,
                L=problem.job_counts,
                purchase_costs=problem.purchase_costs,
                running_costs=problem.running_costs,
            )

        return _ffd

    if normalized in {"simple"}:
        return lambda problem: simple_scheduler(problem, max_iterations=iterations)

    raise ValueError(
        f"Unknown scheduler '{name}'. Expected one of: ruin_recreate, ffd, simple_scheduler."
    )


def run_on_instance(
    npz_path: Path, scheduler_fn: Callable[[ProblemInstance], ScheduleResult]
) -> InstanceResult:
    problem = _load_problem(npz_path)

    start = time.time()
    schedule = scheduler_fn(problem)
    runtime_sec = time.time() - start

    schedule.validate(problem)

    return InstanceResult(
        filename=npz_path.name,
        total_cost=float(schedule.total_cost),
        machine_vector=np.asarray(schedule.machine_vector, dtype=int),
        runtime_sec=runtime_sec,
    )


def evaluate_dataset(
    dataset_dir: Path,
    scheduler_name: str,
    *,
    iterations: int,
    limit: int | None,
    seed: int | None,
    output_csv: Path,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    scheduler_fn = _build_scheduler(scheduler_name, iterations=iterations, rng=rng)

    rows: list[dict[str, object]] = []
    for idx, (npz_path, dims) in enumerate(_dataset_entries(dataset_dir)):
        if limit is not None and idx >= limit:
            break

        result = run_on_instance(npz_path, scheduler_fn)
        row = {
            "filename": result.filename,
            "K": dims["K"],
            "J": dims["J"],
            "M": dims["M"],
            "T": dims["T"],
            "total_cost": result.total_cost,
            "total_machines": result.total_machines,
            "runtime_sec": result.runtime_sec,
            "machine_vector": " ".join(map(str, result.machine_vector.tolist())),
        }
        rows.append(row)
        print(
            f"{result.filename}: cost={result.total_cost:.4f}, "
            f"machines={result.machine_vector}, runtime={result.runtime_sec:.3f}s"
        )

    if rows:
        with output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "filename",
                    "K",
                    "J",
                    "M",
                    "T",
                    "total_cost",
                    "total_machines",
                    "runtime_sec",
                    "machine_vector",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    return rows


def _print_summary(rows: list[dict[str, object]]) -> None:
    if not rows:
        print("No instances were evaluated.")
        return

    costs = np.array([row["total_cost"] for row in rows], dtype=float)
    runtimes = np.array([row["runtime_sec"] for row in rows], dtype=float)
    machine_counts = np.array([row["total_machines"] for row in rows], dtype=int)

    print(
        "\nDataset summary:"
        f"\n  Instances: {len(rows)}"
        f"\n  Avg cost: {costs.mean():.4f}"
        f"\n  Min/Max cost: {costs.min():.4f} / {costs.max():.4f}"
        f"\n  Avg machines: {machine_counts.mean():.2f}"
        f"\n  Avg runtime: {runtimes.mean():.3f}s"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a scheduler/packing algorithm on a dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset"),
        help="Directory containing dataset.csv and NPZ instances.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="ruin_recreate",
        help="Scheduler to run (ruin_recreate | ffd).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of iterations (used by ruin_recreate).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on how many instances to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the scheduler RNG.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset") / "eval_results.csv",
        help="Where to write the per-instance results CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = evaluate_dataset(
        dataset_dir=args.dataset,
        scheduler_name=args.scheduler,
        iterations=args.iterations,
        limit=args.limit,
        seed=args.seed,
        output_csv=args.output,
    )
    _print_summary(rows)


if __name__ == "__main__":
    main()
