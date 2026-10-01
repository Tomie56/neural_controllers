"""Format frozen paper assets for ICLR; never regenerate manuscript prose."""
from pathlib import Path
import re

PAPER = Path(__file__).resolve().parents[1]


def normalize_tables():
    for path in sorted((PAPER / "tables").glob("*.tex")):
        text = path.read_text()
        if r"\begin{tabular" not in text:
            continue
        # Captions may contain nested math/formatting braces. Scan balanced
        # braces instead of a non-greedy regex so none of the caption is lost.
        match = re.search(r"\\caption(?:of\*?\{table\}|\*)?\{", text)
        if not match:
            continue
        depth, end = 1, match.end()
        while depth:
            char = text[end]
            if char in "{}" and text[end - 1] != "\\":
                depth += 1 if char == "{" else -1
            end += 1
        label = re.match(r"\s*\\label\{[^}]+\}", text[end:])
        if label:
            end += label.end()
        caption = text[match.start():end]
        body = text[:match.start()] + text[end:]
        tabular = body.index(r"\begin{tabular")
        wrappers = [body.find(command) for command in (r"\resizebox", r"\scalebox")]
        tabular = min([tabular] + [position for position in wrappers if position >= 0])
        # A caption and its label precede the entire (possibly two-panel) table.
        text = body[:tabular].rstrip() + "\n" + caption + "\n" + body[tabular:]
        if path.name in {"appendix_split_macro.tex", "appendix_geometry_ablation.tex"}:
            text = text.replace(r"\begin{center}", r"\begin{table}[t]")
            text = text.replace(r"\end{center}", r"\end{table}")
            text = text.replace(r"\captionof{table}", r"\caption")
        text = re.sub(r"\n{3,}", "\n\n", text)
        if text != path.read_text():
            path.write_text(text)


def render_heatmap():
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    frame = pd.read_csv(PAPER / "data/cross_model_prediction_heatmap.csv")
    models = list(frame.model.unique())
    targets = list(frame.target.unique())
    specs = [("B", "random_forest"), ("B+M", "random_forest"),
             ("B+M+L", "random_forest"), ("B+M+L+R", "logistic"),
             ("B+M+L+R", "random_forest")]
    assert len(models) == 3 and len(targets) == 8 and len(frame) == 120
    assert not frame.duplicated(["model", "target", "feature_set", "predictor"]).any()
    mpl.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans",
                         "font.size": 7, "axes.linewidth": 0.4})
    # Restore the original wide Figure 4 layout. The compact redraw made the
    # model panels and rotated labels harder to read at paper scale.
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.38), sharey=True)
    for ax, model in zip(axes, models):
        data = frame[frame.model.eq(model)].set_index(["target", "feature_set", "predictor"])
        values = np.array([[data.loc[(t, f, p), "auroc"] for f, p in specs] for t in targets])
        im = ax.imshow(values, vmin=.50, vmax=.90, cmap="Blues", aspect="auto")
        for (y, x), value in np.ndenumerate(values):
            ax.text(x, y, f"{value:.3f}", ha="center", va="center",
                    fontsize=5.7, color="white" if value >= .76 else "#1F2937")
        ax.set_title(model.replace("-Base-2512", "").replace("-Base", ""),
                     fontweight="bold", fontsize=8.3)
        ax.set_xticks(range(5), ["B", "B+M", "B+M+L", "Final (linear)", "Final (RF)"],
                      rotation=27, ha="right", fontsize=5.9)
        ax.tick_params(length=0, pad=2)
        ax.set_yticks(range(8), ["Supp.", "Enh.", "Target", "N-dmg.", "C-dmg.", "Cl.-S", "Cl.-E", "Clean"], fontsize=6.8)
        ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 8, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=.35)
        ax.tick_params(which="minor", length=0)
    fig.subplots_adjust(left=.085, right=.90, bottom=.20, top=.78, wspace=.18)
    cbax = fig.add_axes([.925, .20, .013, .67])
    cb = fig.colorbar(im, cax=cbax, ticks=[.5,.7,.9])
    cb.ax.tick_params(labelsize=7, length=2, pad=1)
    cb.ax.set_ylabel("AUROC", fontsize=8, labelpad=5)
    for suffix in ["pdf", "png"]:
        fig.savefig(PAPER / f"figures/cross_model_prediction_heatmap.{suffix}", dpi=320)
    plt.close(fig)


if __name__ == "__main__":
    normalize_tables()
    render_heatmap()
    print("ICLR captions and frozen-data heatmap prepared")
