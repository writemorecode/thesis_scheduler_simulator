from __future__ import annotations

import argparse
import bisect
import csv
import itertools
import math
from pathlib import Path

import numpy as np

from eval_utils import (
    normalize_scheduler_name,
    parse_scheduler_list,
    scheduler_output_filename,
)


def _discover_results(
    results_dir: Path, scheduler_names: list[str] | None
) -> dict[str, Path]:
    if scheduler_names:
        results: dict[str, Path] = {}
        for name in scheduler_names:
            canonical = normalize_scheduler_name(name)
            if canonical in results:
                raise ValueError(f"Duplicate scheduler '{canonical}' in list.")
            csv_path = results_dir / scheduler_output_filename(canonical)
            results[canonical] = csv_path
        return results

    results = {}
    for path in sorted(results_dir.glob("eval_*.csv")):
        if not path.is_file():
            continue
        name = path.stem[5:] if path.stem.startswith("eval_") else path.stem
        if name in results:
            raise ValueError(f"Duplicate scheduler '{name}' derived from {path}.")
        results[name] = path
    if not results:
        raise FileNotFoundError(f"No eval_*.csv files found in {results_dir}.")
    return results


def _load_costs(csv_path: Path) -> dict[str, float]:
    costs: dict[str, float] = {}
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{csv_path} has no header row.")
        required = {"filename", "total_cost"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{csv_path} missing required columns: {sorted(missing)}.")
        for row in reader:
            filename = row["filename"]
            if filename in costs:
                raise ValueError(
                    f"{csv_path} has duplicate filename entry: {filename}."
                )
            try:
                costs[filename] = float(row["total_cost"])
            except ValueError as exc:
                raise ValueError(
                    f"{csv_path} has invalid total_cost for {filename}: {row['total_cost']}"
                ) from exc
    if not costs:
        raise ValueError(f"{csv_path} has no rows to evaluate.")
    return costs


def _ensure_matching_instances(
    costs_by_scheduler: dict[str, dict[str, float]],
) -> list[str]:
    iterator = iter(costs_by_scheduler.items())
    try:
        base_name, base_costs = next(iterator)
    except StopIteration:
        raise ValueError("No schedulers provided.") from None
    base_filenames = set(base_costs.keys())
    if not base_filenames:
        raise ValueError(f"No instances found for scheduler '{base_name}'.")
    for name, costs in iterator:
        filenames = set(costs.keys())
        missing = base_filenames - filenames
        extra = filenames - base_filenames
        if missing or extra:
            details = [f"Scheduler '{name}' has mismatched instances vs '{base_name}'."]
            if missing:
                details.append(f"Missing {len(missing)} instance(s).")
            if extra:
                details.append(f"Extra {len(extra)} instance(s).")
            raise ValueError(" ".join(details))
    return sorted(base_filenames)


def compute_tau1_wins(
    costs_by_scheduler: dict[str, dict[str, float]],
    filenames: list[str],
    *,
    rel_tol: float,
    abs_tol: float,
) -> dict[str, int]:
    wins = {name: 0 for name in costs_by_scheduler}
    for filename in filenames:
        instance_costs = {
            name: costs[filename] for name, costs in costs_by_scheduler.items()
        }
        best_cost = min(instance_costs.values())
        for name, cost in instance_costs.items():
            if math.isclose(cost, best_cost, rel_tol=rel_tol, abs_tol=abs_tol):
                wins[name] += 1
    return wins


def _build_tau_values(max_ratio: float, num_points: int = 200) -> list[float]:
    return np.linspace(1.0, max_ratio, num=num_points).tolist()


def compute_performance_profiles(
    costs_by_scheduler: dict[str, dict[str, float]],
    filenames: list[str],
    *,
    num_points: int = 200,
) -> tuple[list[float], dict[str, list[float]]]:
    ratios_by_scheduler = {name: [] for name in costs_by_scheduler}
    for filename in filenames:
        instance_costs = {
            name: costs[filename] for name, costs in costs_by_scheduler.items()
        }
        best_cost = min(instance_costs.values())
        if best_cost <= 0:
            raise ValueError(
                f"Instance '{filename}' has non-positive best cost: {best_cost}."
            )
        for name, cost in instance_costs.items():
            ratios_by_scheduler[name].append(cost / best_cost)

    max_ratio = max(max(ratios) for ratios in ratios_by_scheduler.values())
    tau_values = _build_tau_values(max_ratio, num_points=num_points)

    profiles: dict[str, list[float]] = {}
    for name, ratios in ratios_by_scheduler.items():
        ratios_sorted = sorted(ratios)
        counts = [
            bisect.bisect_right(ratios_sorted, tau) / len(ratios_sorted)
            for tau in tau_values
        ]
        profiles[name] = counts
    return tau_values, profiles


def _plot_performance_profiles(
    tau_values: list[float],
    profiles: dict[str, list[float]],
    output_path: Path,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    fig, ax = plt.subplots()
    scheduler_names = sorted(profiles)
    color_map = plt.get_cmap("tab20", max(len(scheduler_names), 1))
    colors = [color_map(i) for i in range(len(scheduler_names))]
    linestyles = itertools.cycle(["-", "--", "-.", ":"])
    for name, color in zip(scheduler_names, colors, strict=True):
        linestyle = next(linestyles)
        ax.plot(
            tau_values,
            profiles[name],
            label=name,
            color=color,
            linestyle=linestyle,
        )
    ax.set_xlabel("Tau")
    ax.set_ylabel("Fraction of instances")
    ax.set_title("Dolan-More Performance Profiles")
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    formatter.set_useOffset(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Dolan-More performance profile at tau=1 "
            "(win counts for best total_cost per instance)."
        )
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
        default=Path("eval_profile_tau1.csv"),
        help="Where to write the tau=1 win summary CSV.",
    )
    parser.add_argument(
        "--rel-tol",
        type=float,
        default=1e-9,
        help="Relative tolerance for tie detection at best cost.",
    )
    parser.add_argument(
        "--abs-tol",
        type=float,
        default=1e-12,
        help="Absolute tolerance for tie detection at best cost.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the computed win summary (default: off).",
    )
    parser.add_argument(
        "--plot-filename",
        type=Path,
        default=None,
        help="Optional plot output path for performance profile line plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scheduler_names = None
    if args.schedulers:
        scheduler_names = parse_scheduler_list(args.schedulers)
        if not scheduler_names:
            raise SystemExit("No schedulers specified. Use --schedulers a,b,c.")

    mapping = _discover_results(args.results_dir, scheduler_names)
    costs_by_scheduler = {}
    for scheduler_name, csv_path in mapping.items():
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Missing results file for {scheduler_name}: {csv_path}"
            )
        costs_by_scheduler[scheduler_name] = _load_costs(csv_path)

    filenames = _ensure_matching_instances(costs_by_scheduler)
    wins = compute_tau1_wins(
        costs_by_scheduler,
        filenames,
        rel_tol=args.rel_tol,
        abs_tol=args.abs_tol,
    )
    total_instances = len(filenames)

    rows = []
    for scheduler_name in sorted(wins, key=lambda name: (-wins[name], name)):
        count = wins[scheduler_name]
        rows.append(
            {
                "scheduler": scheduler_name,
                "wins": count,
                "total_instances": total_instances,
                "win_fraction": count / total_instances,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scheduler", "wins", "total_instances", "win_fraction"],
        )
        writer.writeheader()
        writer.writerows(rows)

    if args.plot_filename:
        tau_values, profiles = compute_performance_profiles(
            costs_by_scheduler,
            filenames,
        )
        _plot_performance_profiles(tau_values, profiles, args.plot_filename)

    if args.verbose:
        for row in rows:
            print(
                f"{row['scheduler']}: wins={row['wins']}, "
                f"fraction={row['win_fraction']:.3f}"
            )


if __name__ == "__main__":
    main()
