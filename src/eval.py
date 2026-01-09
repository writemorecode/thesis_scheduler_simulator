from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from algorithms import ScheduleResult
from eval_utils import (
    build_scheduler,
    normalize_scheduler_name,
    parse_scheduler_list,
    scheduler_output_filename,
)
from problem_generation import ProblemInstance


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


def run_on_instance(
    npz_path: Path,
    scheduler_fn: Callable[[ProblemInstance], ScheduleResult],
    *,
    validate: bool,
) -> InstanceResult:
    problem = _load_problem(npz_path)

    start = time.time()
    schedule = scheduler_fn(problem)
    runtime_sec = time.time() - start

    if validate:
        schedule.validate(problem)

    return InstanceResult(
        filename=npz_path.name,
        total_cost=float(schedule.total_cost),
        machine_vector=np.asarray(schedule.machine_vector, dtype=int),
        runtime_sec=runtime_sec,
    )


def evaluate_schedulers(
    dataset_dir: Path,
    scheduler_names: list[str],
    *,
    iterations: int,
    limit: int | None,
    seed: int | None,
    output_dir: Path,
    validate: bool,
    verbose: bool,
) -> dict[str, list[dict[str, object]]]:
    entries = list(_dataset_entries(dataset_dir))
    if limit is not None:
        entries = entries[:limit]

    if not entries:
        return {}

    canonical_names: list[str] = []
    for name in scheduler_names:
        canonical = normalize_scheduler_name(name)
        if canonical in canonical_names:
            raise ValueError(f"Duplicate scheduler '{canonical}' in list.")
        canonical_names.append(canonical)

    results: dict[str, list[dict[str, object]]] = {}
    for idx, scheduler_name in enumerate(canonical_names):
        scheduler_seed = None if seed is None else seed + idx
        rng = np.random.default_rng(scheduler_seed)
        scheduler_fn = build_scheduler(scheduler_name, iterations=iterations, rng=rng)

        rows: list[dict[str, object]] = []
        for npz_path, dims in entries:
            result = run_on_instance(npz_path, scheduler_fn, validate=validate)
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
            if verbose:
                print(
                    f"[{scheduler_name}] {result.filename}: "
                    f"cost={result.total_cost:.4f}, "
                    f"machines={result.machine_vector}, "
                    f"runtime={result.runtime_sec:.3f}s"
                )

        if rows:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_csv = output_dir / scheduler_output_filename(scheduler_name)
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

        results[scheduler_name] = rows

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run schedulers on a dataset and write per-instance CSVs."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset"),
        help="Directory containing dataset.csv and NPZ instances.",
    )
    scheduler_group = parser.add_mutually_exclusive_group()
    scheduler_group.add_argument(
        "--scheduler",
        type=str,
        help="Single scheduler name (deprecated; use --schedulers).",
    )
    scheduler_group.add_argument(
        "--schedulers",
        type=str,
        default="ruin_recreate",
        help=(
            "Comma-separated list of scheduler names "
            "(ruin_recreate, ffd, ffd_new, "
            "ffd_sum, ffd_max, ffd_prod, ffd_l2, ffd_with_repack, bfd)."
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of iterations (used by ruin_recreate and ffd_with_repack).",
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
        "--output-dir",
        type=Path,
        default=Path("eval_results"),
        help="Directory to write per-scheduler results CSVs.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate each schedule after solving (default: off).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-instance results (default: off).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.scheduler:
        scheduler_names = [args.scheduler]
    else:
        scheduler_names = parse_scheduler_list(args.schedulers)
    if not scheduler_names:
        raise SystemExit("No schedulers specified. Use --schedulers a,b,c.")

    results = evaluate_schedulers(
        dataset_dir=args.dataset,
        scheduler_names=scheduler_names,
        iterations=args.iterations,
        limit=args.limit,
        seed=args.seed,
        output_dir=args.output_dir,
        validate=args.validate,
        verbose=args.verbose,
    )
    if not results:
        print("No instances were evaluated.")


if __name__ == "__main__":
    main()
