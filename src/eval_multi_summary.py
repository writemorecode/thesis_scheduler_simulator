from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from eval import _build_scheduler, _dataset_entries, run_on_instance


def _parse_schedulers(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _evaluate_scheduler(
    entries: list[tuple[Path, dict[str, int]]],
    scheduler_name: str,
    *,
    iterations: int,
    seed: int | None,
    validate: bool,
    verbose: bool,
) -> dict[str, object] | None:
    rng = np.random.default_rng(seed)
    scheduler_fn = _build_scheduler(scheduler_name, iterations=iterations, rng=rng)

    costs: list[float] = []
    runtimes: list[float] = []
    machine_counts: list[int] = []
    for npz_path, _dims in entries:
        result = run_on_instance(npz_path, scheduler_fn, validate=validate)
        costs.append(result.total_cost)
        runtimes.append(result.runtime_sec)
        machine_counts.append(result.total_machines)

    if not costs:
        return None

    costs_array = np.asarray(costs, dtype=float)
    runtimes_array = np.asarray(runtimes, dtype=float)
    machines_array = np.asarray(machine_counts, dtype=int)

    row = {
        "scheduler": scheduler_name,
        "avg_cost": float(costs_array.mean()),
        "min_cost": float(costs_array.min()),
        "max_cost": float(costs_array.max()),
        "avg_runtime_sec": float(runtimes_array.mean()),
        "avg_machines": float(machines_array.mean()),
    }

    if verbose:
        print(
            f"{scheduler_name}: avg_cost={row['avg_cost']:.4f}, "
            f"min_cost={row['min_cost']:.4f}, max_cost={row['max_cost']:.4f}, "
            f"avg_runtime={row['avg_runtime_sec']:.3f}s, "
            f"avg_machines={row['avg_machines']:.2f}"
        )

    return row


def evaluate_schedulers(
    dataset_dir: Path,
    scheduler_names: list[str],
    *,
    iterations: int,
    limit: int | None,
    seed: int | None,
    validate: bool,
    verbose: bool,
) -> list[dict[str, object]]:
    entries = list(_dataset_entries(dataset_dir))
    if limit is not None:
        entries = entries[:limit]

    if not entries:
        return []

    rows: list[dict[str, object]] = []
    for idx, scheduler_name in enumerate(scheduler_names):
        scheduler_seed = None if seed is None else seed + idx
        row = _evaluate_scheduler(
            entries,
            scheduler_name,
            iterations=iterations,
            seed=scheduler_seed,
            validate=validate,
            verbose=verbose,
        )
        if row is not None:
            rows.append(row)

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate multiple schedulers on a dataset and write summary CSV."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset"),
        help="Directory containing dataset.csv and NPZ instances.",
    )
    parser.add_argument(
        "--schedulers",
        type=str,
        default="ruin_recreate",
        help="Comma-separated list of scheduler names.",
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
        help="Base seed for scheduler RNG (offset per scheduler).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset") / "eval_summary.csv",
        help="Where to write the per-scheduler summary CSV.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate each schedule after solving (default: off).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-scheduler summaries (default: off).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scheduler_names = _parse_schedulers(args.schedulers)
    if not scheduler_names:
        raise SystemExit("No schedulers specified. Use --schedulers a,b,c.")

    rows = evaluate_schedulers(
        dataset_dir=args.dataset,
        scheduler_names=scheduler_names,
        iterations=args.iterations,
        limit=args.limit,
        seed=args.seed,
        validate=args.validate,
        verbose=args.verbose,
    )

    if not rows:
        print("No instances were evaluated.")
        return

    rows.sort(key=lambda row: float(row["avg_cost"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scheduler",
                "avg_cost",
                "min_cost",
                "max_cost",
                "avg_runtime_sec",
                "avg_machines",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
