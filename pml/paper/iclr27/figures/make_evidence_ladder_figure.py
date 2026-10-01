#!/usr/bin/env python3
"""Create horizontal and single-column PML evidence-ladder figures."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[2] / "results" / "paper_tables" / "path_summary.csv"


def load_rows() -> list[dict[str, str]]:
    with RESULTS.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def percent(row: dict[str, str], key: str) -> float:
    return 100.0 * float(row[key])


def draw_panels(axes, coarse, agop) -> None:
    methods = ["mean_difference", "logistic", "linear", "random"]
    labels = ["Mean diff.", "Logistic", "Linear", "Random"]
    positions = np.arange(len(methods))
    width = 0.34

    axes[0].bar(
        positions - width / 2,
        [percent(coarse[method], "target_any") for method in methods],
        width,
        color="#0072B2",
        label="Target-any",
    )
    axes[0].bar(
        positions + width / 2,
        [percent(coarse[method], "clean_window") for method in methods],
        width,
        color="#E69F00",
        hatch="//",
        label="Clean window",
    )
    axes[0].set_xticks(positions, labels, rotation=15, ha="right")
    axes[0].set_ylabel("Records (%)")
    axes[0].set_ylim(0, 72)
    axes[0].set_title("(a) Leverage exceeds clean control", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, ncol=2, loc="upper right")
    axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.5)

    points = [
        ("Mean diff.", coarse["mean_difference"], "o", "#0072B2"),
        ("Logistic", coarse["logistic"], "s", "#56B4E9"),
        ("Linear", coarse["linear"], "^", "#009E73"),
        ("Random", coarse["random"], "D", "#777777"),
        ("AGOP top-1", agop["agop_top1"], "P", "#CC79A7"),
        ("AGOP top-k", agop["agop_topk_project"], "X", "#D55E00"),
    ]
    offsets = {
        "Mean diff.": (7, -10),
        "Logistic": (-40, 7),
        "Linear": (7, 7),
        "Random": (-38, -10),
        "AGOP top-1": (-44, -10),
        "AGOP top-k": (-50, 7),
    }
    for label, row, marker, color in points:
        x_value = percent(row, "target_any")
        y_value = percent(row, "neighbor_damage_any")
        axes[1].scatter(
            x_value,
            y_value,
            s=48,
            marker=marker,
            color=color,
            edgecolor="black",
            linewidth=0.4,
        )
        x_offset, y_offset = offsets[label]
        axes[1].annotate(
            label,
            (x_value, y_value),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=7.0,
        )

    axes[1].plot([35, 95], [35, 95], linestyle="--", color="#A6A6A6", linewidth=0.8)
    axes[1].set_xlim(35, 95)
    axes[1].set_ylim(30, 98)
    axes[1].set_xlabel("Target-any (%)")
    axes[1].set_ylabel("Neighbor damage (%)")
    axes[1].set_title("(b) More leverage can mean more damage", loc="left", fontweight="bold")
    axes[1].grid(color="#E6E6E6", linewidth=0.5)


def save_figure(coarse, agop, layout: str) -> None:
    if layout == "horizontal":
        figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.75), constrained_layout=True)
        output = HERE / "evidence_ladder"
    else:
        figure, axes = plt.subplots(2, 1, figsize=(3.25, 4.35), constrained_layout=True)
        output = HERE / "evidence_ladder_vertical"
    draw_panels(axes, coarse, agop)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.2,
            "axes.titlesize": 8.8,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    rows = load_rows()
    coarse = {
        row["method"]: row
        for row in rows
        if row["dataset"] == "commonsense_3000_baselines"
    }
    agop = {
        row["method"]: row
        for row in rows
        if row["dataset"] == "heldout_144_agop_direct"
    }
    save_figure(coarse, agop, "horizontal")
    save_figure(coarse, agop, "vertical")


if __name__ == "__main__":
    main()
