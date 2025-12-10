from __future__ import annotations
import sys

import numpy as np
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


def _load_results(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read algorithm-step-cost rows keyed by algorithm name."""
    X, y = np.loadtxt(path, delimiter=",", unpack=True, skiprows=1)
    return X, y


def plot_results(
    csv_path: str | Path, output_path: str | Path = "results_plot.png"
) -> Path:
    csv_file = Path(csv_path)
    X, y = _load_results(csv_file)

    _apply_latex_style()

    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    ax.plot(
        X,
        y,
        linewidth=2.2,
        marker="o",
        markersize=6,
    )

    ax.set_title("Cost")
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
    filename = sys.argv[1]
    output = plot_results(Path(filename))
    print(f"Wrote plot to {output}")
