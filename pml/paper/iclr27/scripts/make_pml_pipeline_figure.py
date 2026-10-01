#!/usr/bin/env python3
"""Render the paper-facing PML representation, path, and decision overview."""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PAPER_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PAPER_ROOT / "figures"

NAVY = "#264653"
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#6B7280"
LIGHT_BLUE = "#EAF3F8"
LIGHT_GREEN = "#E7F5EF"
LIGHT_ORANGE = "#FFF3D6"
LIGHT_PURPLE = "#F7EAF2"
LIGHT_GRAY = "#F3F4F5"

mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
    }
)


def arrow(axis, start, end, color=GRAY, linewidth=1.1, dashed=False, mutation=10):
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation,
            linewidth=linewidth,
            color=color,
            linestyle="--" if dashed else "-",
            shrinkA=1,
            shrinkB=1,
        )
    )


def rounded_box(axis, xy, width, height, text, facecolor, edgecolor, fontsize=7.0):
    axis.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.015,rounding_size=0.025",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.0,
        )
    )
    axis.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=NAVY,
        linespacing=1.12,
    )


def representation_panel(axis, show_title=True):
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    if show_title:
        axis.set_title("(a) Internal representation", loc="left", fontweight="bold")

    axis.text(0.01, 0.93, "prompt", fontsize=7.0, color=NAVY, va="center")
    arrow(axis, (0.10, 0.93), (0.20, 0.93))
    block_x = [0.21, 0.31, 0.41, 0.51, 0.61]
    labels = ["0", r"$\cdots$", "7", "11", r"$\cdots$"]
    for index, (x_pos, label) in enumerate(zip(block_x, labels)):
        selected = index in {2, 3}
        axis.add_patch(
            FancyBboxPatch(
                (x_pos, 0.84),
                0.075,
                0.16,
                boxstyle="round,pad=0.008,rounding_size=0.014",
                facecolor=LIGHT_BLUE if selected else LIGHT_GRAY,
                edgecolor=BLUE if selected else "#9CA3AF",
                linewidth=1.35 if selected else 0.9,
            )
        )
        axis.text(x_pos + 0.0375, 0.92, label, ha="center", va="center", fontsize=7.4)
        if index < len(block_x) - 1:
            arrow(axis, (x_pos + 0.076, 0.92), (block_x[index + 1], 0.92), mutation=8)
    arrow(axis, (0.686, 0.92), (0.74, 0.92))
    axis.text(0.76, 0.92, "next-token logits", fontsize=7.0, color=NAVY, va="center")

    axis.plot([0.448, 0.448], [0.83, 0.69], color=BLUE, linewidth=1.1)
    arrow(axis, (0.448, 0.70), (0.43, 0.62), color=BLUE, mutation=9)
    axis.text(0.47, 0.71, r"read $\mathbf{h}_{i\ell}$", fontsize=7.0, color=BLUE)

    rng = np.random.default_rng(113)
    desired = rng.normal(loc=(0.23, 0.34), scale=(0.035, 0.036), size=(8, 2))
    contrast = rng.normal(loc=(0.43, 0.21), scale=(0.035, 0.03), size=(8, 2))
    axis.scatter(desired[:, 0], desired[:, 1], s=18, color=BLUE, edgecolor="white", linewidth=0.35)
    axis.scatter(contrast[:, 0], contrast[:, 1], s=20, marker="^", color=ORANGE, edgecolor="white", linewidth=0.35)
    axis.text(0.10, 0.46, "desired answer states", fontsize=7.0, color=BLUE)
    axis.text(0.34, 0.13, "contrast answer states", fontsize=7.0, color="#9A5C00")
    arrow(axis, (0.39, 0.23), (0.29, 0.33), color=PURPLE, linewidth=1.8, mutation=12)
    axis.text(0.32, 0.35, r"direction $\mathbf{v}_{i\ell s}$", fontsize=7.3, color=PURPLE, fontweight="bold")

    axis.scatter([0.68], [0.23], s=38, color=NAVY, zorder=4)
    arrow(axis, (0.68, 0.23), (0.84, 0.36), color=GREEN, linewidth=1.8, mutation=12)
    axis.scatter([0.84], [0.36], s=38, facecolor="white", edgecolor=GREEN, linewidth=1.3, zorder=4)
    axis.text(0.55, 0.055, r"$\mathbf{h}'_\ell=\mathbf{h}_\ell+\alpha\mathbf{v}_{i\ell s}$", fontsize=7.2, color=NAVY)
    axis.text(0.73, 0.43, "controlled residual state", fontsize=7.0, color=GREEN)
    axis.text(0.57, 0.28, "add at every token", fontsize=7.0, color=GRAY)


def path_panel(axis, show_title=True):
    if show_title:
        axis.set_title("(b) Intervention path", loc="left", fontweight="bold")
    alpha = np.array([-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5])
    target = np.array([-0.88, -0.54, -0.13, 0.0, 0.16, 0.58, 0.93])
    neighbor = np.array([-0.58, -0.21, -0.04, 0.0, -0.03, -0.13, -0.52])
    capability = np.array([-0.29, -0.10, -0.02, 0.0, -0.01, -0.06, -0.22])
    axis.axhline(0, color="#9CA3AF", linewidth=0.75, zorder=0)
    axis.axhline(0.42, color=BLUE, linewidth=0.85, linestyle=(0, (3, 2)), zorder=0)
    axis.axhline(-0.38, color=VERMILLION, linewidth=0.85, linestyle=(0, (3, 2)), zorder=0)
    axis.axvspan(0.205, 0.365, color=LIGHT_GREEN, zorder=-1)
    axis.plot(alpha, target, color=BLUE, marker="o", markersize=4.0, linewidth=1.8, label="Target")
    axis.plot(alpha, neighbor, color=VERMILLION, marker="s", markersize=3.6, linewidth=1.45, label="Neighbor")
    axis.plot(alpha, capability, color=GRAY, marker="^", markersize=3.7, linewidth=1.35, label="Capability")
    axis.text(0.305, 1.04, "clean window", ha="center", va="center", fontsize=6.8, color=GREEN, fontweight="bold")
    axis.annotate(
        "target onset",
        xy=(0.25, 0.58),
        xytext=(-0.07, 1.06),
        fontsize=7.2,
        color=BLUE,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8),
    )
    axis.annotate(
        "collateral onset",
        xy=(0.5, -0.52),
        xytext=(0.13, -0.93),
        fontsize=7.2,
        color=VERMILLION,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=VERMILLION, lw=0.8),
    )
    axis.text(-0.515, 0.46, r"$\tau_T$", fontsize=7.0, color=BLUE)
    axis.text(-0.515, -0.34, r"$-\tau_{N/C}$", fontsize=7.0, color=VERMILLION)
    axis.set_xlim(-0.53, 0.53)
    axis.set_ylim(-1.03, 1.13)
    axis.set_xlabel(r"intervention strength $\alpha$")
    axis.set_ylabel("margin change")
    axis.set_xticks(alpha, ["-.5", "-.25", "-.1", "0", ".1", ".25", ".5"])
    axis.set_yticks([])
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.5)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=3,
        frameon=False,
        fontsize=6.5,
        handlelength=1.25,
        columnspacing=0.55,
        handletextpad=0.28,
    )


def decision_panel(axis, show_title=True):
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    if show_title:
        axis.set_title("(c) Forecast utility and decide", loc="left", fontweight="bold")
    rounded_box(axis, (0.01, 0.72), 0.24, 0.14, "Static prior\n$B+M+L+G$", LIGHT_GRAY, GRAY, 7.0)
    rounded_box(axis, (0.01, 0.48), 0.24, 0.14, "Weak probe\n" + r"$R$ at $|\alpha|=.1$", LIGHT_BLUE, BLUE, 7.0)
    rounded_box(axis, (0.33, 0.56), 0.22, 0.20, "Predict\nlater path\n" + r"for each $\alpha$", LIGHT_PURPLE, PURPLE, 7.0)
    arrow(axis, (0.25, 0.79), (0.33, 0.69), color=GRAY)
    arrow(axis, (0.25, 0.55), (0.33, 0.62), color=BLUE)
    arrow(axis, (0.55, 0.66), (0.64, 0.66), color=PURPLE)

    candidate = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])
    utility = np.array([-0.05, 0.01, 0.0, 0.09, 0.03])
    x = 0.80 + 0.18 * candidate
    y = 0.57 + 1.35 * utility
    axis.axhline(0.57, xmin=0.68, xmax=0.98, color="#9CA3AF", linewidth=0.7)
    axis.plot(x, y, color=GREEN, marker="o", markersize=3.4, linewidth=1.5)
    axis.scatter([x[3]], [y[3]], s=62, facecolor=LIGHT_GREEN, edgecolor=GREEN, linewidth=1.2, zorder=4)
    axis.text(0.63, 0.87, "predicted utility", fontsize=7.0, color=NAVY)
    axis.text(0.76, 0.76, r"choose $\alpha=.25$", fontsize=7.0, color=GREEN, fontweight="bold")
    axis.text(0.67, 0.47, "candidate strength", fontsize=7.0, color=GRAY)
    axis.text(0.62, 0.29, r"if $\max_\alpha \hat U(\alpha)\leq0$", fontsize=7.0, color=NAVY)
    rounded_box(axis, (0.73, 0.12), 0.22, 0.12, "abstain", LIGHT_ORANGE, ORANGE, 7.2)
    arrow(axis, (0.84, 0.28), (0.84, 0.24), color=ORANGE)
    axis.text(0.02, 0.20, "Later labels use only\n" + r"$|\alpha|\in\{.25,.5\}$", fontsize=7.0, color=GRAY)


def save_panel(filename, drawer, figsize, margins):
    figure, axis = plt.subplots(figsize=figsize)
    drawer(axis, show_title=False)
    figure.subplots_adjust(**margins)
    output = FIGURE_DIR / filename
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    figure.savefig(output.with_suffix(".png"), dpi=320, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all-panels",
        action="store_true",
        help="Regenerate the legacy A/C panels and combined preview as well as panel B.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    save_panel(
        "figure1_intervention_path",
        path_panel,
        (2.55, 2.35),
        {"left": 0.20, "right": 0.97, "bottom": 0.31, "top": 0.98},
    )
    if not args.all_panels:
        print("Wrote Figure 1(b); preserved externally edited panels (a) and (c)")
        return

    save_panel(
        "figure1_internal_representation",
        representation_panel,
        (3.05, 2.35),
        {"left": 0.01, "right": 0.99, "bottom": 0.03, "top": 0.99},
    )
    save_panel(
        "figure1_forecast_decide",
        decision_panel,
        (2.65, 2.35),
        {"left": 0.01, "right": 0.99, "bottom": 0.03, "top": 0.99},
    )

    # Retain a combined preview for local visual QA; the manuscript uses the
    # three independent files above so panels can be replaced separately.
    figure, axes = plt.subplots(1, 3, figsize=(7.05, 3.05), gridspec_kw={"width_ratios": [1.40, 1.10, 1.12]})
    representation_panel(axes[0])
    path_panel(axes[1])
    decision_panel(axes[2])
    figure.subplots_adjust(left=0.015, right=0.995, bottom=0.16, top=0.90, wspace=0.34)
    output = FIGURE_DIR / "pml_pipeline"
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    figure.savefig(output.with_suffix(".png"), dpi=320, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    print("Wrote independent Figure 1 panels and combined preview")


if __name__ == "__main__":
    main()
