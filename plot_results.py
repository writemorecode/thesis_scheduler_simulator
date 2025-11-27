from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def _apply_latex_style() -> None:
    """Configure Matplotlib for a LaTeX-like serif appearance."""
    plt.style.use("seaborn-v0_8-paper")
    if shutil.which("latex"):
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
            }
        )
    else:
        plt.rcParams.update(
            {
                "text.usetex": False,
                "mathtext.fontset": "cm",
                "font.family": "serif",
            }
        )


def _load_results(path: Path) -> Dict[str, List[Tuple[int, float]]]:
    """Read algorithm-step-cost rows keyed by algorithm name."""
    series: Dict[str, List[Tuple[int, float]]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            algo = str(row["algorithm"])
            step = int(row["step"])
            cost = float(row["cost"])
            series.setdefault(algo, []).append((step, cost))

    # Ensure points are plotted in step order.
    for algo, points in series.items():
        points.sort(key=lambda entry: entry[0])
    return series


def plot_results(
    csv_path: str | Path, output_path: str | Path = "results_plot.png"
) -> Path:
    csv_file = Path(csv_path)
    results = _load_results(csv_file)
    print(results)

    _apply_latex_style()

    colors = plt.get_cmap("tab10").colors
    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    for idx, (algo, points) in enumerate(
        sorted(results.items(), key=lambda item: item[0])
    ):
        steps, costs = zip(*points)
        color = colors[idx % len(colors)]
        ax.plot(
            steps,
            costs,
            label=f"Algorithm {algo}",
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=6,
        )

    ax.set_title("Cost by Scheduler and Iteration")
    ax.set_xlabel("Iteration step")
    ax.set_ylabel("Cost")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.legend(title="Scheduler", frameon=False)

    fig.tight_layout()
    output_file = Path(output_path)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_file


if __name__ == "__main__":
    output = plot_results(Path("results.csv"))
    print(f"Wrote plot to {output}")
