"""
Analysis utilities for comparing algorithm performance results.

Loads per-algorithm CSVs, joins them on ``filename``, computes pairwise log-ratios
of ``total_cost``, and aggregates summary statistics. Designed for quick CLI use:

    python -m src.analysis --results-dir eval_results
    python -m src.analysis --schedulers ffd,ruin_recreate
    python -m src.analysis eval_ffd.csv eval_ruin_recreate.csv
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from statistics import NormalDist

import numpy as np
import polars as pl

from eval_utils import (
    normalize_scheduler_name,
    parse_scheduler_list,
    scheduler_output_filename,
)


@dataclass(frozen=True)
class AlgorithmData:
    name: str
    path: Path
    df: pl.DataFrame


def _t_critical(df: int, confidence: float = 0.95) -> float:
    """Return the two-tailed critical t value; fall back to normal if SciPy missing."""
    if df <= 0:
        raise ValueError("Degrees of freedom must be positive for t critical value")
    alpha = 1.0 - confidence
    prob = 1.0 - alpha / 2.0
    try:
        from scipy import stats  # type: ignore

        return float(stats.t.ppf(prob, df))
    except Exception:
        # Normal approximation as a fallback when SciPy is unavailable.
        warnings.warn(
            "SciPy is not installed; using normal approximation for t critical value.",
            RuntimeWarning,
            stacklevel=2,
        )
        return float(NormalDist().inv_cdf(prob))


def load_algorithm_results(
    results_dir: Path, mapping: dict[str, str]
) -> list[AlgorithmData]:
    """Read each algorithm CSV."""
    data: list[AlgorithmData] = []
    for algo, filename in mapping.items():
        path = (results_dir / filename).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing results file for {algo}: {path}")
        df = (
            pl.read_csv(path)
            .select(
                "filename",
                pl.col("total_cost").alias(f"total_cost_{algo}"),
            )
            .group_by("filename")
            .agg(pl.col(f"total_cost_{algo}").min())
        )
        data.append(AlgorithmData(name=algo, path=path, df=df))
    return data


def _discover_mapping(
    results_dir: Path, scheduler_names: Iterable[str] | None
) -> dict[str, str]:
    if scheduler_names:
        mapping: dict[str, str] = {}
        for name in scheduler_names:
            canonical = normalize_scheduler_name(name)
            if canonical in mapping:
                raise ValueError(f"Duplicate algorithm name derived from {name}")
            filename = scheduler_output_filename(canonical)
            path = results_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing results file for {canonical}: {path}")
            mapping[canonical] = filename
        return mapping

    mapping: dict[str, str] = {}
    for path in sorted(results_dir.glob("eval_*.csv")):
        if not path.is_file():
            continue
        name = path.stem[5:] if path.stem.startswith("eval_") else path.stem
        if name in mapping:
            raise ValueError(f"Duplicate algorithm name derived from {path}")
        mapping[name] = path.name

    if not mapping:
        raise FileNotFoundError(f"No eval_*.csv files found in {results_dir}")
    return mapping


def _mapping_from_csv_paths(paths: Iterable[Path]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in paths:
        stem = path.stem
        name = stem[5:] if stem.startswith("eval_") else stem
        if name in mapping:
            raise ValueError(f"Duplicate algorithm name derived from {path}")
        mapping[name] = path.name
    return mapping


def join_on_filename(datasets: Iterable[AlgorithmData]) -> pl.DataFrame:
    """Inner-join all algorithm result frames on filename."""
    iterator = iter(datasets)
    try:
        base = next(iterator).df
    except StopIteration:
        raise ValueError("No datasets provided for join") from None

    joined = base
    for algo in iterator:
        joined = joined.join(algo.df, on="filename", how="inner")
    return joined


def add_log_ratio_columns(
    df: pl.DataFrame, algos: Iterable[str]
) -> tuple[pl.DataFrame, list[str]]:
    """Add pairwise log(total_cost ratios) for all ordered pairs of algorithms."""
    # Ensure costs are valid for logarithms.
    for algo in algos:
        col = f"total_cost_{algo}"
        min_val = df[col].min()  # type: ignore[index]
        if min_val is None or min_val <= 0:
            raise ValueError(
                f"total_cost for {algo} contains non-positive values; cannot take log ratios"
            )

    names = list(algos)
    log_ratio_columns: list[str] = []
    result = df
    for num in names:
        for den in names:
            if num == den:
                continue
            numerator = f"total_cost_{num}"
            denominator = f"total_cost_{den}"
            ratio_name = f"log_ratio_{num}_over_{den}"
            result = result.with_columns(
                (pl.col(numerator).log() - pl.col(denominator).log()).alias(ratio_name)
            )
            log_ratio_columns.append(ratio_name)
    return result, log_ratio_columns


def summarize_ratios(df: pl.DataFrame, ratio_columns: Iterable[str]) -> pl.DataFrame:
    """Compute summary statistics and 95% paired t CIs for each log-ratio column."""
    summaries = []
    for col in ratio_columns:
        series = df[col]
        count = series.len()
        mean = series.mean()
        median = series.median()
        std = series.std()
        min_val = series.min()
        max_val = series.max()
        ci_low = ci_high = None
        if mean is not None and std is not None and count >= 2:
            se = std / sqrt(count)
            t_crit = _t_critical(count - 1, confidence=0.95)
            half_width = t_crit * se
            ci_low = np.exp(mean - half_width)
            ci_high = np.exp(mean + half_width)
        summaries.append(
            {
                "ratio": col,
                "count": count,
                "mean": mean,
                "median": median,
                "std": std,
                "min": min_val,
                "max": max_val,
                "ci_low_95": ci_low,
                "ci_high_95": ci_high,
            }
        )
    return pl.DataFrame(summaries)


def run_analysis(
    results_dir: Path = Path("eval_results"),
    mapping: dict[str, str] | None = None,
    scheduler_names: Iterable[str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load results, compute log-ratios, and return (joined_df, summary_df)."""
    mapping = mapping or _discover_mapping(results_dir, scheduler_names)
    data = load_algorithm_results(results_dir, mapping)
    joined = join_on_filename(data)
    joined_with_log_ratios, log_ratio_cols = add_log_ratio_columns(
        joined, mapping.keys()
    )
    summary = summarize_ratios(joined_with_log_ratios, log_ratio_cols)
    return joined_with_log_ratios, summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze algorithm total_cost log-ratios."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("eval_results"),
        help="Directory containing per-algorithm CSVs.",
    )
    parser.add_argument(
        "--schedulers",
        type=str,
        default=None,
        help="Optional comma-separated list of scheduler names.",
    )
    parser.add_argument(
        "csv_paths",
        nargs="*",
        type=Path,
        help="Optional list of CSV files to analyze instead of --results-dir.",
    )
    parser.add_argument(
        "--export-joined",
        type=Path,
        help="Optional path to write the joined data with log-ratio columns as CSV.",
    )
    parser.add_argument(
        "--export-summary",
        type=Path,
        help="Optional path to write the log-ratio summary statistics as CSV.",
    )
    args = parser.parse_args()

    if args.csv_paths:
        mapping = _mapping_from_csv_paths(args.csv_paths)
        parents = {path.parent.resolve() for path in args.csv_paths}
        if len(parents) != 1:
            raise ValueError(
                "All CSV paths must live in the same directory when passed as arguments."
            )
        results_dir = parents.pop()
        joined, summary = run_analysis(results_dir, mapping=mapping)
    else:
        scheduler_names = None
        if args.schedulers:
            scheduler_names = parse_scheduler_list(args.schedulers)
            if not scheduler_names:
                raise SystemExit("No schedulers specified. Use --schedulers a,b,c.")
        joined, summary = run_analysis(
            args.results_dir, scheduler_names=scheduler_names
        )

    print("\nLog-ratio summaries:")
    print(summary.sort("ratio"))

    if args.export_joined:
        joined.write_csv(args.export_joined)
        print(f"\nWrote joined data to {args.export_joined}")
    if args.export_summary:
        summary.write_csv(args.export_summary)
        print(f"Wrote summary to {args.export_summary}")


if __name__ == "__main__":
    main()
