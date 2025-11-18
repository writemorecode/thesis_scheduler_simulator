from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from algorithms import BinInfo, ScheduleResult


@dataclass
class _FreeRectangle:
    x: float
    y: float
    width: float
    height: float


@dataclass
class _PlacedItem:
    x: float
    y: float
    width: float
    height: float
    job_type: int
    size: Tuple[float, float]


def _bin_item_shapes(
    bin_info: BinInfo, requirements: np.ndarray
) -> List[Tuple[float, float, int]]:
    items: List[Tuple[float, float, int]] = []
    counts = bin_info.item_counts.reshape(-1)
    for job_type, count in enumerate(counts):
        num = int(count)
        if num <= 0:
            continue
        item = requirements[:, job_type].astype(float).reshape(-1)
        if item.shape[0] != 2:
            raise ValueError(
                "Visualization only supports two-dimensional requirements."
            )
        width, height = float(item[0]), float(item[1])
        if width < 0 or height < 0:
            raise ValueError("Job requirements must be non-negative.")
        for _ in range(num):
            items.append((width, height, job_type))
    return items


def _select_free_rect(
    free_rects: List[_FreeRectangle], width: float, height: float, eps: float
) -> int | None:
    best_idx: int | None = None
    best_score = float("inf")
    for idx, rect in enumerate(free_rects):
        if width <= rect.width + eps and height <= rect.height + eps:
            leftover = (rect.width - width) * (rect.height - height)
            if leftover < best_score:
                best_score = leftover
                best_idx = idx
    return best_idx


def _prune_free_rectangles(
    free_rects: List[_FreeRectangle], eps: float
) -> List[_FreeRectangle]:
    pruned: List[_FreeRectangle] = []
    for i, rect in enumerate(free_rects):
        contained = False
        rect_x2 = rect.x + rect.width
        rect_y2 = rect.y + rect.height
        for j, other in enumerate(free_rects):
            if i == j:
                continue
            other_x2 = other.x + other.width
            other_y2 = other.y + other.height
            if (
                rect.x >= other.x - eps
                and rect.y >= other.y - eps
                and rect_x2 <= other_x2 + eps
                and rect_y2 <= other_y2 + eps
            ):
                contained = True
                break
        if not contained:
            pruned.append(rect)
    return pruned


def _layout_bin_items(
    bin_info: BinInfo, capacities: np.ndarray, requirements: np.ndarray
) -> List[_PlacedItem]:
    capacity_vec = capacities[:, bin_info.bin_type].astype(float).reshape(-1)
    if capacity_vec.shape[0] != 2:
        raise ValueError("Visualization only supports two-dimensional bin capacities.")
    cap_width, cap_height = capacity_vec
    if cap_width <= 0 or cap_height <= 0:
        raise ValueError(
            "Bin capacities must be positive in both dimensions for visualization."
        )

    items = _bin_item_shapes(bin_info, requirements)
    if not items:
        return []

    # Large items first increases the likelihood of finding a placement.
    items.sort(
        key=lambda entry: (entry[0] * entry[1], max(entry[0], entry[1])), reverse=True
    )

    eps = 1e-9
    free_rects = [_FreeRectangle(x=0.0, y=0.0, width=cap_width, height=cap_height)]
    placed: List[_PlacedItem] = []

    for width, height, job_type in items:
        idx = _select_free_rect(free_rects, width, height, eps)
        if idx is None:
            raise ValueError(
                "Unable to lay out items for visualization; the generated schedule uses a configuration "
                "that the drawer cannot embed without overlap."
            )
        rect = free_rects.pop(idx)
        x, y = rect.x, rect.y
        placed.append(
            _PlacedItem(
                x=x / cap_width,
                y=y / cap_height,
                width=width / cap_width,
                height=height / cap_height,
                job_type=job_type,
                size=(width, height),
            )
        )

        remaining_width = rect.width - width
        remaining_height = rect.height - height

        if remaining_width > eps:
            free_rects.append(
                _FreeRectangle(
                    x=x + width,
                    y=y,
                    width=remaining_width,
                    height=height,
                )
            )
        if remaining_height > eps:
            free_rects.append(
                _FreeRectangle(
                    x=x,
                    y=y + height,
                    width=rect.width,
                    height=remaining_height,
                )
            )

        free_rects = _prune_free_rectangles(free_rects, eps)

    return placed


def visualize_schedule(
    schedule: ScheduleResult,
    capacities: np.ndarray,
    requirements: np.ndarray,
    output_path: str | Path,
    *,
    slot_labels: Sequence[str] | None = None,
    title: str | None = None,
    dpi: int = 200,
    legend: bool = True,
) -> Path:
    """Render a schedule as a 2D image for ``K = 2`` instances.

    Parameters
    ----------
    schedule : ScheduleResult
        Result returned by :func:`schedule_jobs`.
    capacities : np.ndarray
        ``(2, M)`` matrix that was used for the scheduler.
    requirements : np.ndarray
        ``(2, J)`` matrix describing per-job resource demand.
    output_path : str or pathlib.Path
        Target file path. The suffix controls the output type; defaults to ``.png``.
    slot_labels : sequence of str, optional
        Custom per-slot labels. Falls back to ``Slot i``.
    title : str, optional
        Title placed at the top of the figure.
    dpi : int, optional
        Resolution used when saving the figure.
    legend : bool, optional
        If ``True`` (default) draw a color legend per job type.
    """

    capacities = np.asarray(capacities, dtype=float)
    requirements = np.asarray(requirements, dtype=float)
    if capacities.ndim != 2 or requirements.ndim != 2:
        raise ValueError("C and R must be 2D matrices for visualization.")
    if capacities.shape[0] != 2 or requirements.shape[0] != 2:
        raise ValueError(
            "Visualization is only supported for two-dimensional instances (K = 2)."
        )

    time_slots = schedule.time_slot_solutions
    if not time_slots:
        raise ValueError("Schedule does not contain any time slots to visualize.")

    num_slots = len(time_slots)
    num_job_types = requirements.shape[1]

    if slot_labels is not None and len(slot_labels) != num_slots:
        raise ValueError(
            f"slot_labels must have {num_slots} entries; got {len(slot_labels)} instead."
        )
    labels = slot_labels or [f"Slot {idx + 1}" for idx in range(num_slots)]

    max_bins = max((len(slot.bins) for slot in time_slots), default=0)
    max_bins = max(max_bins, 1)

    # Geometry for laying out the figure on a single Matplotlib axes.
    bin_width = 1.4
    bin_height = 1.1
    horizontal_gap = 0.4
    vertical_gap = 0.7
    margin_x = 0.8
    margin_y = 0.6

    plot_width = margin_x * 2 + max_bins * bin_width + (max_bins - 1) * horizontal_gap
    plot_height = margin_y * 2 + num_slots * bin_height + (num_slots - 1) * vertical_gap

    output_path = Path(output_path)
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_format = output_path.suffix[1:]

    import matplotlib.pyplot as plt
    from matplotlib import patches as mpatches

    fig = plt.figure(figsize=(plot_width * 1.6, plot_height * 1.6))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, plot_width)
    ax.set_ylim(plot_height, 0)
    ax.axis("off")

    cmap = plt.get_cmap("tab20")
    colors = [cmap(idx % cmap.N) for idx in range(max(num_job_types, 1))]

    for slot_idx, slot in enumerate(time_slots):
        y0 = margin_y + slot_idx * (bin_height + vertical_gap)
        ax.text(
            margin_x - 0.3,
            y0 + bin_height / 2,
            labels[slot_idx],
            ha="right",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

        job_count = int(
            np.sum([np.sum(bin_info.item_counts) for bin_info in slot.bins])
        )
        ax.text(
            plot_width - margin_x + 0.1,
            y0 + bin_height / 2,
            f"jobs: {job_count}",
            ha="left",
            va="center",
            fontsize=10,
            color="#444444",
        )

        if not slot.bins:
            ax.text(
                margin_x,
                y0 + bin_height / 2,
                "No active bins",
                ha="left",
                va="center",
                fontsize=11,
                color="#666666",
            )
            continue

        for bin_idx, bin_info in enumerate(slot.bins):
            x0 = margin_x + bin_idx * (bin_width + horizontal_gap)
            rect = mpatches.Rectangle(
                (x0, y0),
                bin_width,
                bin_height,
                linewidth=1.5,
                edgecolor="#333333",
                facecolor="#fafafa",
            )
            ax.add_patch(rect)

            capacity_vec = capacities[:, bin_info.bin_type].astype(float).reshape(-1)
            usage = (requirements @ bin_info.item_counts.reshape(-1, 1)).reshape(-1)
            remaining = bin_info.remaining_capacity.reshape(-1)

            header = f"Type {bin_info.bin_type}"
            ax.text(
                x0 + bin_width / 2,
                y0 - 0.1,
                header,
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )
            cap_text = f"Cap: ({capacity_vec[0]:g}, {capacity_vec[1]:g})"
            ax.text(
                x0 + bin_width / 2,
                y0 + bin_height + 0.05,
                cap_text,
                ha="center",
                va="bottom",
                fontsize=9,
                color="#555555",
            )
            use_text = f"Used: ({usage[0]:g}, {usage[1]:g})"
            ax.text(
                x0 + bin_width / 2,
                y0 + bin_height + 0.2,
                use_text,
                ha="center",
                va="bottom",
                fontsize=9,
                color="#555555",
            )
            rem_text = f"Rem: ({remaining[0]:g}, {remaining[1]:g})"
            ax.text(
                x0 + bin_width / 2,
                y0 + bin_height + 0.35,
                rem_text,
                ha="center",
                va="bottom",
                fontsize=9,
                color="#555555",
            )

            placement_items = []
            try:
                placement_items = _layout_bin_items(bin_info, capacities, requirements)
            except ValueError as exc:
                ax.text(
                    x0 + bin_width / 2,
                    y0 + bin_height / 2,
                    str(exc),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#cc0000",
                    wrap=True,
                )

            for placed_item in placement_items:
                color = colors[placed_item.job_type % len(colors)]
                px = x0 + placed_item.x * bin_width
                py = y0 + placed_item.y * bin_height
                patch = mpatches.Rectangle(
                    (px, py),
                    placed_item.width * bin_width,
                    placed_item.height * bin_height,
                    facecolor=color,
                    edgecolor="#222222",
                    linewidth=0.8,
                    alpha=0.85,
                )
                ax.add_patch(patch)
                label = f"J{placed_item.job_type}\n({placed_item.size[0]:g}, {placed_item.size[1]:g})"
                ax.text(
                    px + placed_item.width * bin_width / 2,
                    py + placed_item.height * bin_height / 2,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )

    if title:
        ax.set_title(title, fontsize=14, pad=10)

    if legend and num_job_types > 0:
        legend_handles = [
            mpatches.Patch(
                color=colors[idx % len(colors)],
                label=f"Job {idx} ({requirements[0, idx]:g}, {requirements[1, idx]:g})",
            )
            for idx in range(num_job_types)
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(num_job_types, 3),
            frameon=False,
        )

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", format=file_format)
    plt.close(fig)
    return output_path
