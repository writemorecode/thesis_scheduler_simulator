from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from eval_utils import (
    normalize_scheduler_name,
    parse_scheduler_list,
    scheduler_output_filename,
)


def _discover_results(results_dir: Path) -> dict[str, Path]:
    results: dict[str, Path] = {}
    for path in sorted(results_dir.glob("eval_*.csv")):
        if not path.is_file():
            continue
        name = path.stem
        if name.startswith("eval_"):
            name = name[5:]
        results[name] = path
    return results


def _summarize_scheduler(
    scheduler_name: str, csv_path: Path, *, verbose: bool
) -> dict[str, object] | None:
    costs: list[float] = []
    runtimes: list[float] = []
    machine_counts: list[int] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            costs.append(float(row["total_cost"]))
            runtimes.append(float(row["runtime_sec"]))
            machine_counts.append(int(row["total_machines"]))

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
            f"{scheduler_name}: avg_cost={row['avg_cost']:.2f}, "
            f"min_cost={row['min_cost']:.2f}, max_cost={row['max_cost']:.2f}, "
            f"avg_runtime={row['avg_runtime_sec']:.2f}s, "
            f"avg_machines={row['avg_machines']:.2f}"
        )

    return row


def evaluate_summaries(
    results_dir: Path, scheduler_names: list[str] | None, *, verbose: bool
) -> list[dict[str, object]]:
    if scheduler_names:
        results: dict[str, Path] = {}
        for name in scheduler_names:
            canonical = normalize_scheduler_name(name)
            if canonical in results:
                raise ValueError(f"Duplicate scheduler '{canonical}' in list.")
            csv_path = results_dir / scheduler_output_filename(canonical)
            results[canonical] = csv_path
    else:
        results = _discover_results(results_dir)

    rows: list[dict[str, object]] = []
    for scheduler_name, csv_path in results.items():
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Missing results file for {scheduler_name}: {csv_path}"
            )
        row = _summarize_scheduler(scheduler_name, csv_path=csv_path, verbose=verbose)
        if row is not None:
            rows.append(row)

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize per-scheduler results from raw per-instance CSVs."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("eval_results"),
        help="Directory containing per-scheduler CSVs.",
    )
    parser.add_argument(
        "--schedulers",
        type=str,
        default=None,
        help="Optional comma-separated list of scheduler names.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_summary_results.csv"),
        help="Where to write the per-scheduler summary CSV.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-scheduler summaries (default: off).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scheduler_names = None
    if args.schedulers:
        scheduler_names = parse_scheduler_list(args.schedulers)
        if not scheduler_names:
            raise SystemExit("No schedulers specified. Use --schedulers a,b,c.")

    rows = evaluate_summaries(
        results_dir=args.results_dir,
        scheduler_names=scheduler_names,
        verbose=args.verbose,
    )

    if not rows:
        print("No instances were evaluated.")
        return

    rows.sort(key=lambda row: float(row["avg_cost"]))

    numeric_fields = {
        "avg_cost",
        "min_cost",
        "max_cost",
        "avg_runtime_sec",
        "avg_machines",
    }

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
        for row in rows:
            formatted_row = dict(row)
            for field in numeric_fields:
                formatted_row[field] = f"{float(row[field]):.2f}"
            writer.writerow(formatted_row)


if __name__ == "__main__":
    main()
